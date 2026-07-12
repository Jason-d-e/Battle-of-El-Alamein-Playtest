import { createAlphaAiClient } from "./ai-alpha-controller.js";
import {
  alphaModelEnvironmentFingerprint,
  validateAlphaModelEnvironment,
  validateAlphaModelFeatureContract,
  validateReleasedAlphaModelArtifact,
} from "./ai-alpha-model.js";
import { alphaTrainingFeatureContract } from "./ai-alpha-training.js";

export const DEFAULT_ALPHA_BROWSER_SEARCH = Object.freeze({
  iterations: 6,
  maxDepth: 1,
  actionLimit: 6,
  preApplyLimit: 18,
  timeBudgetMs: 45,
  responsePaddingMs: 160,
  progressInterval: 2,
  progressLimit: 3,
  modelPolicyWeight: 2.4,
});
export const DEFAULT_ALPHA_BROWSER_MODEL_VALIDATION = Object.freeze({
  minSuiteGames: 2,
  minSuiteSides: 2,
  minAnalyzedActions: 1,
  minAverageRootVisits: 1,
  minTrainingSamples: 1,
  minTrainingSides: 1,
  minTrainingSources: 1,
});
const DEFAULT_ALPHA_BROWSER_CACHE_ENTRIES = 8;
const DEFAULT_ALPHA_ANALYSIS_SCHEDULER_INTERVAL_MS = 250;

export function createBrowserAlphaAi({
  rawModel = null,
  scenario = null,
  rules = null,
  workerFactory = null,
  timeoutMs = 90,
  logger = console,
  cacheEntries = DEFAULT_ALPHA_BROWSER_CACHE_ENTRIES,
  modelValidationOptions = DEFAULT_ALPHA_BROWSER_MODEL_VALIDATION,
} = {}) {
  const client = createAlphaAiClient({
    workerFactory,
    directFallback: false,
    timeoutMs,
  });
  const analysisClient = createAlphaAiClient({
    workerFactory: null,
    directFallback: true,
    timeoutMs,
  });
  const expectedEnvironment = modelValidationOptions.expectedEnvironment
    || alphaModelEnvironmentFingerprint({ scenario, rules });
  const expectedFeatureContract = modelValidationOptions.expectedFeatureContract
    || alphaTrainingFeatureContract();
  const validation = validateReleasedAlphaModelArtifact(rawModel, {
    ...modelValidationOptions,
    expectedEnvironment,
    expectedFeatureContract,
  });
  const environmentValidation = validation.environment
    || validateAlphaModelEnvironment(validation.model || rawModel, expectedEnvironment, {
      requireEnvironmentFingerprint: modelValidationOptions.requireEnvironmentFingerprint,
    });
  const featureContractValidation = validation.featureContract
    || validateAlphaModelFeatureContract(validation.model || rawModel, expectedFeatureContract, {
      requireFeatureContract: modelValidationOptions.requireFeatureContract,
    });
  const validationWithEnvironment = {
    ...validation,
    environment: environmentValidation,
    featureContract: featureContractValidation,
  };
  const model = validation.ok ? validation.model : null;
  const modelStatus = summarizeBrowserAlphaModelValidation(validationWithEnvironment);
  if (rawModel && !validation.ok) {
    logger?.warn?.("Ignoring untrusted Alpha AI model; falling back to scripted AI weights.", validation.reason);
  }
  return {
    client,
    analysisClient,
    model,
    modelValidation: validationWithEnvironment,
    modelStatus,
    analysisCache: createBrowserAlphaAnalysisCache({ maxEntries: cacheEntries }),
    lastAnalysisMeta: null,
    lastAnalysis: null,
    lastSummary: null,
  };
}

export async function chooseBrowserAlphaAction({
  alpha = null,
  scenario = null,
  rules = null,
  state = null,
  side = null,
  legalActions = null,
  searchOptions = {},
  logger = console,
} = {}) {
  if (!alpha || !state || !scenario || !rules) return null;
  try {
    const options = buildBrowserAlphaSearchOptions({
      model: alpha.model,
      side,
      searchOptions,
    });
    const useCache = options.useCache !== false;
    const cacheKey = useCache
      ? browserAlphaAnalysisCacheKey({
        scenario,
        rules,
        state,
        side,
        model: alpha.model,
        searchOptions: options,
      })
      : null;
    const cached = useCache ? getBrowserAlphaCachedAnalysis(alpha, cacheKey) : null;
    if (cached?.analysis) {
      updateBrowserAlphaSummary(alpha, cached.analysis, { cacheHit: true, cacheKey });
      return cached.analysis.requiresChance ? null : legalBrowserAlphaActionOrNull(cached.analysis.bestAction, legalActions, logger);
    }
    if (!alpha.client?.chooseAction) return null;
    const payload = {
      scenario,
      rules,
      state,
      model: alpha.model,
      searchOptions: options,
    };
    const action = await alpha.client.chooseAction(payload, {
      timeoutMs: browserAlphaTimeoutMs(options),
    });
    const analysis = alpha.client.getLastAnalysis?.() || null;
    updateBrowserAlphaSummary(alpha, analysis || await fallbackBrowserAlphaAnalysis(alpha, payload, options), {
      cacheHit: false,
      cacheKey,
    });
    return legalBrowserAlphaActionOrNull(action, legalActions, logger);
  } catch (error) {
    logger?.warn?.("Alpha AI analysis failed; falling back to scripted AI.", error);
    return null;
  }
}

export async function analyzeBrowserAlphaPosition({
  alpha = null,
  scenario = null,
  rules = null,
  state = null,
  side = null,
  searchOptions = {},
  logger = console,
} = {}) {
  if (!alpha?.client?.analyze || !state || !scenario || !rules) return null;
  try {
    const options = buildBrowserAlphaSearchOptions({
      model: alpha.model,
      side,
      searchOptions,
    });
    const useCache = options.useCache !== false;
    const cacheKey = useCache
      ? browserAlphaAnalysisCacheKey({
        scenario,
        rules,
        state,
        side,
        model: alpha.model,
        searchOptions: options,
      })
      : null;
    const cached = useCache ? getBrowserAlphaCachedAnalysis(alpha, cacheKey) : null;
    if (cached?.analysis) {
      updateBrowserAlphaSummary(alpha, cached.analysis, { cacheHit: true, cacheKey });
      return {
        analysis: alpha.lastAnalysis,
        summary: alpha.lastSummary,
      };
    }
    const payload = {
      scenario,
      rules,
      state,
      model: alpha.model,
      searchOptions: options,
    };
    const analysis = await alpha.client.analyze(payload, {
      timeoutMs: browserAlphaTimeoutMs(options),
    });
    updateBrowserAlphaSummary(alpha, analysis || await fallbackBrowserAlphaAnalysis(alpha, payload, options), {
      cacheHit: false,
      cacheKey,
    });
    return alpha.lastAnalysis
      ? {
        analysis: alpha.lastAnalysis,
        summary: alpha.lastSummary,
      }
      : null;
  } catch (error) {
    logger?.warn?.("Alpha AI analysis failed.", error);
    return null;
  }
}

export function createBrowserAlphaAnalysisScheduler({
  analyze = analyzeBrowserAlphaPosition,
  minIntervalMs = DEFAULT_ALPHA_ANALYSIS_SCHEDULER_INTERVAL_MS,
  defaultSearchOptions = {},
  now = () => Date.now(),
  setTimer = (callback, delay) => globalThis.setTimeout?.(callback, delay),
  clearTimer = (timerId) => globalThis.clearTimeout?.(timerId),
  onResult = null,
  onError = null,
} = {}) {
  let timer = null;
  let queued = null;
  let inFlight = false;
  let inFlightKey = null;
  let lastStartedAt = 0;
  let lastCompletedKey = null;
  let requestId = 0;

  function request(context = {}) {
    const job = makeScheduledAlphaAnalysisJob(context, {
      defaultSearchOptions,
      requestId: requestId + 1,
    });
    if (!job.key) return { scheduled: false, reason: "missing_position_key", key: null };
    if (!context.force && (job.key === lastCompletedKey || job.key === inFlightKey || job.key === queued?.key)) {
      return { scheduled: false, reason: "duplicate_position", key: job.key };
    }
    requestId += 1;
    job.id = requestId;
    queued = job;
    schedule();
    return { scheduled: true, reason: "scheduled", key: job.key };
  }

  function schedule() {
    if (timer || inFlight || !queued) return;
    const delay = Math.max(0, Math.ceil(Number(minIntervalMs || 0) - (Number(now()) - lastStartedAt)));
    timer = setTimer(run, delay);
  }

  async function run() {
    timer = null;
    if (!queued || inFlight) return null;
    const job = queued;
    queued = null;
    inFlight = true;
    inFlightKey = job.key;
    lastStartedAt = Number(now());
    try {
      const result = await analyze({
        alpha: job.alpha,
        scenario: job.scenario,
        rules: job.rules,
        state: job.state,
        side: job.side,
        searchOptions: job.searchOptions,
        logger: job.logger,
      });
      if (result && job.id === requestId) {
        lastCompletedKey = job.key;
        onResult?.(result, job);
      }
      return result;
    } catch (error) {
      onError?.(error, job);
      return null;
    } finally {
      inFlight = false;
      inFlightKey = null;
      schedule();
    }
  }

  function cancel() {
    if (timer) clearTimer?.(timer);
    timer = null;
    queued = null;
  }

  function status() {
    return {
      scheduled: Boolean(timer || queued),
      inFlight,
      queuedKey: queued?.key || null,
      inFlightKey,
      lastCompletedKey,
    };
  }

  return {
    request,
    cancel,
    status,
  };
}

function makeScheduledAlphaAnalysisJob(context = {}, options = {}) {
  const searchOptions = {
    ...(options.defaultSearchOptions || {}),
    ...(context.searchOptions || {}),
  };
  const state = stableCopy(context.state);
  const key = browserAlphaAnalysisCacheKey({
    scenario: context.scenario,
    rules: context.rules,
    state,
    side: context.side,
    model: context.alpha?.model,
    searchOptions,
  });
  return {
    id: options.requestId || 0,
    key,
    alpha: context.alpha || null,
    scenario: context.scenario || null,
    rules: context.rules || null,
    state,
    side: context.side || null,
    searchOptions,
    logger: context.logger || console,
  };
}

async function fallbackBrowserAlphaAnalysis(alpha, payload, options) {
  if (!alpha?.analysisClient?.analyze) return null;
  const fallbackOptions = buildBrowserAlphaFallbackSearchOptions(options);
  try {
    return await alpha.analysisClient.analyze({
      ...payload,
      searchOptions: fallbackOptions,
    }, {
      timeoutMs: browserAlphaTimeoutMs(fallbackOptions),
    });
  } catch {
    return null;
  }
}

export function updateBrowserAlphaSummary(alpha, analysis, metadata = {}) {
  if (!alpha || typeof alpha !== "object") return null;
  alpha.lastAnalysis = analysis || null;
  alpha.lastAnalysisMeta = analysis
    ? {
      cacheHit: Boolean(metadata.cacheHit),
      cacheKey: metadata.cacheKey || null,
    }
    : null;
  if (analysis && metadata.cacheKey && !metadata.cacheHit) {
    rememberBrowserAlphaAnalysis(alpha, metadata.cacheKey, analysis);
  }
  alpha.lastSummary = analysis
    ? summarizeBrowserAlphaAnalysis(analysis, {
      modelStatus: alpha.modelStatus,
      cache: metadata.cacheKey
        ? {
          hit: Boolean(metadata.cacheHit),
        }
        : null,
    })
    : null;
  return alpha.lastSummary;
}

export function summarizeBrowserAlphaAnalysis(analysis, options = {}) {
  if (!analysis || typeof analysis !== "object") return null;
  const policyLimit = Math.max(1, Number(options.policyLimit || 4));
  const featureLimit = Math.max(1, Number(options.featureLimit || 8));
  const variationLimit = Math.max(0, Number(options.variationLimit || 4));
  const progressLimit = Math.max(0, Number(options.progressLimit ?? 3));
  const policy = (analysis.policy || [])
    .slice(0, policyLimit)
    .map((entry) => ({
      action: compactBrowserAction(entry.action),
      visits: finiteNumber(entry.visits, 0),
      visitShare: rounded(entry.visitShare),
      q: rounded(entry.q),
      prior: rounded(entry.prior),
    }));
  const summary = {
    schema: "zizi-el-alamein-alpha-browser-summary-v1",
    side: analysis.side || null,
    stateHash: analysis.stateHash || null,
    turn: analysis.situation?.turn ?? null,
    phaseId: analysis.situation?.phaseId || null,
    rootValue: rounded(analysis.rootValue),
    bestAction: compactBrowserAction(analysis.bestAction),
    topPolicy: policy,
    principalVariation: (analysis.principalVariation || [])
      .slice(0, variationLimit)
      .map(compactBrowserVariationStep),
    candidateLines: (analysis.candidateLines || [])
      .slice(0, policyLimit)
      .map((line) => ({
        action: compactBrowserAction(line.action),
        visits: finiteNumber(line.visits, 0),
        visitShare: rounded(line.visitShare),
        q: rounded(line.q),
        prior: rounded(line.prior),
        value: rounded(line.value),
        principalVariation: (line.principalVariation || [])
          .slice(0, variationLimit)
          .map(compactBrowserVariationStep),
      })),
    features: summarizeSituationFeatures(analysis.situation?.features, featureLimit),
    search: {
      iterations: finiteNumber(analysis.search?.iterations, 0),
      rootVisits: finiteNumber(analysis.search?.rootVisits, 0),
      rootChildren: finiteNumber(analysis.search?.rootChildren, 0),
      maxDepth: finiteNumber(analysis.search?.maxDepth, 0),
      actionLimit: finiteNumber(analysis.search?.actionLimit, 0),
      preApplyLimit: finiteNumber(analysis.search?.preApplyLimit, 0),
      elapsedMs: finiteNumber(analysis.search?.elapsedMs, 0),
    },
    progress: progressLimit > 0
      ? (analysis.progress || [])
        .slice(-progressLimit)
        .map((snapshot) => compactBrowserProgressSnapshot(snapshot, {
          policyLimit,
          variationLimit,
        }))
      : [],
    requiresChance: Boolean(analysis.requiresChance),
  };
  if (options.modelStatus) {
    summary.model = summarizeBrowserAlphaModelValidation(options.modelStatus);
  }
  if (options.cache) {
    summary.cache = {
      hit: Boolean(options.cache.hit),
      status: options.cache.hit ? "hit" : "fresh",
    };
  }
  return summary;
}

export function summarizeBrowserAlphaModelValidation(validation) {
  const status = validation?.schema === "zizi-el-alamein-alpha-browser-model-status-v1"
    ? validation
    : browserAlphaModelStatusFromValidation(validation);
  return {
    schema: "zizi-el-alamein-alpha-browser-model-status-v1",
    ok: Boolean(status?.ok),
    reason: status?.reason || null,
    sampleCount: finiteOrNull(status?.sampleCount),
    sourceArtifact: status?.sourceArtifact || null,
    sourceHash: status?.sourceHash || null,
    releasedAt: status?.releasedAt || null,
    trainingSamples: finiteOrNull(status?.trainingSamples),
    trainingSides: finiteOrNull(status?.trainingSides),
    trainingSources: finiteOrNull(status?.trainingSources),
    reanalysisSamples: finiteOrNull(status?.reanalysisSamples),
    stateSnapshots: finiteOrNull(status?.stateSnapshots),
    validationSamples: finiteOrNull(status?.validationSamples),
    validationGroupBy: status?.validationGroupBy || null,
    validationValueMse: finiteOrNull(status?.validationValueMse),
    validationPolicyCrossEntropy: finiteOrNull(status?.validationPolicyCrossEntropy),
    environmentFingerprint: status?.environmentFingerprint || null,
    expectedEnvironmentFingerprint: status?.expectedEnvironmentFingerprint || null,
    environmentMatches: typeof status?.environmentMatches === "boolean" ? status.environmentMatches : null,
    featureContractFingerprint: status?.featureContractFingerprint || null,
    expectedFeatureContractFingerprint: status?.expectedFeatureContractFingerprint || null,
    featureContractMatches: typeof status?.featureContractMatches === "boolean" ? status.featureContractMatches : null,
    analyzedActions: finiteOrNull(status?.analyzedActions),
    averageRootVisits: finiteOrNull(status?.averageRootVisits),
    runtimeTarget: status?.runtimeTarget || null,
    runtimeModelFile: status?.runtimeModelFile || null,
    runtimeInstallApproved: Boolean(status?.runtimeInstallApproved),
  };
}

function browserAlphaModelStatusFromValidation(validation) {
  const model = validation?.model || null;
  const release = model?.release || null;
  const training = model?.training?.data || null;
  const validationEvidence = model?.training?.validation || null;
  const decisionEvidence = release?.decisionEvidence || null;
  const environment = validation?.environment || null;
  const modelEnvironment = environment?.environment || model?.environment || null;
  const expectedEnvironment = environment?.expected || null;
  const featureContract = validation?.featureContract || null;
  const modelFeatureContract = featureContract?.featureContract || model?.featureContract || null;
  const expectedFeatureContract = featureContract?.expected || null;
  return {
    ok: Boolean(validation?.ok),
    reason: validation?.reason || null,
    sampleCount: model?.sampleCount ?? null,
    sourceArtifact: release?.sourceArtifact || null,
    sourceHash: release?.sourceHash || null,
    releasedAt: release?.releasedAt || null,
    trainingSamples: training?.sampleCount ?? null,
    trainingSides: training ? Object.keys(training.sides || {}).length : null,
    trainingSources: Array.isArray(training?.sources) ? training.sources.length : null,
    reanalysisSamples: training?.reanalysisSamples ?? null,
    stateSnapshots: training?.samplesWithStateSnapshot ?? null,
    validationSamples: validationEvidence?.sampleCount ?? null,
    validationGroupBy: validationEvidence?.validationGroupBy || null,
    validationValueMse: validationEvidence?.value?.mse ?? null,
    validationPolicyCrossEntropy: validationEvidence?.policy?.crossEntropy ?? null,
    environmentFingerprint: modelEnvironment?.fingerprint || null,
    expectedEnvironmentFingerprint: expectedEnvironment?.fingerprint || null,
    environmentMatches: typeof environment?.match === "boolean" ? environment.match : null,
    featureContractFingerprint: modelFeatureContract?.fingerprint || null,
    expectedFeatureContractFingerprint: expectedFeatureContract?.fingerprint || null,
    featureContractMatches: typeof featureContract?.match === "boolean" ? featureContract.match : null,
    analyzedActions: decisionEvidence?.analyzedActions ?? null,
    averageRootVisits: decisionEvidence?.averageRootVisits ?? null,
    runtimeTarget: release?.runtime?.target || null,
    runtimeModelFile: release?.runtime?.modelFile || null,
    runtimeInstallApproved: Boolean(release?.runtime?.installApproved),
  };
}

export function createBrowserAlphaAnalysisCache({
  maxEntries = DEFAULT_ALPHA_BROWSER_CACHE_ENTRIES,
} = {}) {
  const entries = new Map();
  const limit = Math.max(1, Number(maxEntries || DEFAULT_ALPHA_BROWSER_CACHE_ENTRIES));
  return {
    get size() {
      return entries.size;
    },
    get(key) {
      if (!key || !entries.has(key)) return null;
      const value = entries.get(key);
      entries.delete(key);
      entries.set(key, value);
      return value;
    },
    set(key, analysis) {
      if (!key || !analysis) return null;
      entries.delete(key);
      entries.set(key, { analysis });
      while (entries.size > limit) entries.delete(entries.keys().next().value);
      return entries.get(key);
    },
    clear() {
      entries.clear();
    },
  };
}

export function browserAlphaAnalysisCacheKey({
  scenario = null,
  rules = null,
  state = null,
  side = null,
  model = null,
  searchOptions = {},
} = {}) {
  if (!state) return null;
  return stableStringify({
    schema: "zizi-el-alamein-alpha-browser-analysis-cache-key-v1",
    scenario: scenarioFingerprint(scenario),
    rules: rulesFingerprint(rules),
    side: side || searchOptions.side || null,
    model: modelFingerprint(model),
    searchOptions: normalizeSearchOptionsForCache(searchOptions),
    state: browserAlphaAnalysisPositionState(state),
  });
}

export function browserAlphaAnalysisPositionState(state = null) {
  if (!state || typeof state !== "object") return null;
  return {
    turn: state.turn ?? null,
    phaseIndex: state.phaseIndex ?? null,
    combatMode: state.combatMode || null,
    movedUnits: sortedStrings(state.movedUnits),
    usedAttackers: sortedStrings(state.usedAttackers),
    usedDefenders: sortedStrings(state.usedDefenders),
    retreatTask: compactBrowserRetreatTask(state.retreatTask),
    advanceTask: compactBrowserAdvanceTask(state.advanceTask),
    declaredCombats: (state.declaredCombats || []).map(compactBrowserBattleForPosition),
    winner: state.winner
      ? {
        side: state.winner.side || null,
        reason: state.winner.reason || null,
        type: state.winner.type || null,
        turn: state.winner.turn || state.turn || null,
      }
      : null,
    units: (state.units || [])
      .map((unit) => ({
        id: unit.id,
        side: unit.side,
        hexId: unit.hexId,
        disrupted: Boolean(unit.disrupted),
        eliminated: Boolean(unit.eliminated),
      }))
      .sort((left, right) => String(left.id).localeCompare(String(right.id))),
  };
}

export function isBrowserAlphaActionLegal(action = null, legalActions = null) {
  if (!action?.type || !Array.isArray(legalActions)) return false;
  const target = browserAlphaActionIdentity(action);
  return legalActions.some((legalAction) => stableStringify(browserAlphaActionIdentity(legalAction)) === stableStringify(target));
}

export function getBrowserAlphaCachedAnalysis(alpha, cacheKey) {
  if (!cacheKey) return null;
  return ensureBrowserAlphaAnalysisCache(alpha)?.get(cacheKey) || null;
}

export function rememberBrowserAlphaAnalysis(alpha, cacheKey, analysis) {
  if (!cacheKey || !analysis) return null;
  return ensureBrowserAlphaAnalysisCache(alpha)?.set(cacheKey, analysis) || null;
}

export function buildBrowserAlphaSearchOptions({
  model = null,
  side = null,
  searchOptions = {},
} = {}) {
  return {
    side,
    iterations: DEFAULT_ALPHA_BROWSER_SEARCH.iterations,
    maxDepth: DEFAULT_ALPHA_BROWSER_SEARCH.maxDepth,
    actionLimit: DEFAULT_ALPHA_BROWSER_SEARCH.actionLimit,
    preApplyLimit: DEFAULT_ALPHA_BROWSER_SEARCH.preApplyLimit,
    timeBudgetMs: DEFAULT_ALPHA_BROWSER_SEARCH.timeBudgetMs,
    progressInterval: DEFAULT_ALPHA_BROWSER_SEARCH.progressInterval,
    progressLimit: DEFAULT_ALPHA_BROWSER_SEARCH.progressLimit,
    policyWeight: model ? DEFAULT_ALPHA_BROWSER_SEARCH.modelPolicyWeight : 0,
    ...searchOptions,
  };
}

export function buildBrowserAlphaFallbackSearchOptions(options = {}) {
  return {
    ...options,
    iterations: Math.min(2, Math.max(1, Number(options.iterations || 1))),
    maxDepth: 1,
    actionLimit: Math.min(4, Math.max(1, Number(options.actionLimit || 4))),
    preApplyLimit: Math.min(12, Math.max(4, Number(options.preApplyLimit || 12))),
    timeBudgetMs: Math.min(24, Math.max(8, Number(options.timeBudgetMs || 20))),
    progressInterval: Math.min(1, Math.max(0, Number(options.progressInterval ?? DEFAULT_ALPHA_BROWSER_SEARCH.progressInterval))),
    progressLimit: Math.min(1, Math.max(0, Number(options.progressLimit ?? DEFAULT_ALPHA_BROWSER_SEARCH.progressLimit))),
  };
}

function browserAlphaTimeoutMs(options = {}) {
  return Number(options.timeBudgetMs || DEFAULT_ALPHA_BROWSER_SEARCH.timeBudgetMs)
    + Number(options.responsePaddingMs || DEFAULT_ALPHA_BROWSER_SEARCH.responsePaddingMs);
}

function ensureBrowserAlphaAnalysisCache(alpha) {
  if (!alpha || typeof alpha !== "object") return null;
  if (!alpha.analysisCache || typeof alpha.analysisCache.get !== "function" || typeof alpha.analysisCache.set !== "function") {
    alpha.analysisCache = createBrowserAlphaAnalysisCache();
  }
  return alpha.analysisCache;
}

function scenarioFingerprint(scenario) {
  if (!scenario || typeof scenario !== "object") return null;
  const units = Array.isArray(scenario.units) ? scenario.units : [];
  return stableCopy({
    id: scenario.id || scenario.name || scenario.title || null,
    boardImage: scenario.board?.image || null,
    unitCount: units.length,
    objectiveKeys: Object.keys(scenario.objectives || {}).sort(),
  });
}

function rulesFingerprint(rules) {
  if (!rules || typeof rules !== "object") return null;
  return stableCopy({
    schema: rules.schema || null,
    version: rules.version || null,
    phaseCount: Array.isArray(rules.phases) ? rules.phases.length : 0,
    combatResults: Object.keys(rules.combatResults || rules.crt || {}).sort(),
  });
}

function modelFingerprint(model) {
  if (!model || typeof model !== "object") return null;
  return stableCopy({
    schema: model.schema || null,
    generatedAt: model.generatedAt || null,
    sampleCount: model.sampleCount ?? null,
    release: {
      sourceArtifact: model.release?.sourceArtifact || null,
      sourceHash: model.release?.sourceHash || null,
      releasedAt: model.release?.releasedAt || null,
    },
    environment: model.environment?.fingerprint || null,
    featureContract: model.featureContract?.fingerprint || null,
    hasValue: Boolean(model.value),
    hasPolicy: Boolean(model.policy),
  });
}

function sortedStrings(values) {
  return Array.isArray(values) ? values.slice().sort() : [];
}

function compactBrowserRetreatTask(task) {
  if (!task) return null;
  return {
    battleId: task.battleId || null,
    unitIds: (task.unitIds || []).slice(),
    index: task.index || 0,
    steps: task.steps,
    result: task.result,
    origins: stableCopy(task.origins || {}),
    disruptAfterRetreat: Boolean(task.disruptAfterRetreat),
    advanceAfter: Boolean(task.advanceAfter),
  };
}

function compactBrowserAdvanceTask(task) {
  if (!task) return null;
  return {
    battleId: task.battleId || null,
    targetHexId: task.targetHexId,
    attackerIds: (task.attackerIds || []).slice(),
  };
}

function compactBrowserBattleForPosition(battle) {
  return {
    id: battle.id,
    defenderId: battle.defenderId,
    attackerIds: (battle.attackerIds || []).slice(),
    resolved: Boolean(battle.resolved),
    result: battle.resultCode || battle.result || null,
    roll: battle.roll || null,
  };
}

function legalBrowserAlphaActionOrNull(action, legalActions, logger = console) {
  if (!action) return null;
  if (!Array.isArray(legalActions)) return action;
  if (isBrowserAlphaActionLegal(action, legalActions)) return action;
  logger?.warn?.("Ignoring illegal Alpha AI action for current position.", browserAlphaActionIdentity(action));
  return null;
}

function browserAlphaActionIdentity(action) {
  if (!action?.type) return null;
  const type = action.type;
  if (type === "MOVE_UNIT" || type === "RETREAT_UNIT") {
    return {
      type,
      unitId: action.unitId || null,
      toHexId: action.toHexId || null,
      battleId: action.battleId || null,
    };
  }
  if (type === "DECLARE_COMBAT") {
    return {
      type,
      defenderId: action.defenderId || null,
      attackerIds: sortedStrings(action.attackerIds),
    };
  }
  if (type === "ADVANCE_UNIT") {
    return {
      type,
      unitId: action.unitId || null,
      battleId: action.battleId || null,
      targetHexId: action.targetHexId || null,
    };
  }
  if (type === "SKIP_ADVANCE") {
    return {
      type,
      battleId: action.battleId || null,
      targetHexId: action.targetHexId || null,
    };
  }
  if (type === "RESOLVE_COMBAT") {
    return {
      type,
      battleId: action.battleId || null,
      dieRoll: action.dieRoll ?? null,
    };
  }
  return { type };
}

function normalizeSearchOptionsForCache(options = {}) {
  const copy = stableCopy(options || {});
  if (!copy || typeof copy !== "object" || Array.isArray(copy)) return copy;
  delete copy.model;
  delete copy.random;
  delete copy.responsePaddingMs;
  delete copy.useCache;
  return copy;
}

function stableStringify(value) {
  return JSON.stringify(stableCopy(value));
}

function stableCopy(value, seen = new WeakSet()) {
  if (value === null) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "undefined" || typeof value === "function" || typeof value === "symbol") return undefined;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value.toISOString();
  if (Array.isArray(value)) return value.map((item) => stableCopy(item, seen)).filter((item) => item !== undefined);
  if (value instanceof Map) {
    return [...value.entries()]
      .map(([key, entryValue]) => [String(key), stableCopy(entryValue, seen)])
      .filter(([, entryValue]) => entryValue !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
  }
  if (value instanceof Set) {
    return [...value.values()]
      .map((item) => stableCopy(item, seen))
      .filter((item) => item !== undefined)
      .sort((left, right) => stableStringify(left).localeCompare(stableStringify(right)));
  }
  if (typeof value !== "object") return null;
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  const output = {};
  for (const key of Object.keys(value).sort()) {
    const next = stableCopy(value[key], seen);
    if (next !== undefined) output[key] = next;
  }
  seen.delete(value);
  return output;
}

function summarizeSituationFeatures(features, limit) {
  if (!features || typeof features !== "object") return [];
  return Object.entries(features)
    .map(([key, value]) => ({
      key,
      value: rounded(value),
      magnitude: Math.abs(Number(value) || 0),
    }))
    .filter((entry) => Number.isFinite(entry.value))
    .sort((a, b) => b.magnitude - a.magnitude || a.key.localeCompare(b.key))
    .slice(0, limit)
    .map(({ key, value }) => ({ key, value }));
}

function compactBrowserAction(action) {
  if (!action || typeof action !== "object") return null;
  const compact = {};
  for (const key of ["type", "unitId", "fromHexId", "toHexId", "targetHexId", "defenderId", "battleId", "dieRoll"]) {
    if (action[key] !== undefined) compact[key] = action[key];
  }
  if (Array.isArray(action.attackerIds)) compact.attackerIds = action.attackerIds.slice();
  if (action.route && typeof action.route === "object") {
    compact.route = {
      remaining: finiteNumber(action.route.remaining, 0),
      path: Array.isArray(action.route.path) ? action.route.path.slice() : [],
    };
  }
  return compact;
}

function compactBrowserVariationStep(step) {
  if (!step || typeof step !== "object") return null;
  if (step.action) {
    return {
      action: compactBrowserAction(step.action),
      visits: finiteNumber(step.visits, 0),
      q: rounded(step.q),
    };
  }
  return {
    action: compactBrowserAction(step),
    visits: 0,
    q: 0,
  };
}

function compactBrowserProgressSnapshot(snapshot, options = {}) {
  const policyLimit = Math.max(1, Number(options.policyLimit || 4));
  const variationLimit = Math.max(0, Number(options.variationLimit || 4));
  return {
    schema: "zizi-el-alamein-alpha-progress-v1",
    iterations: finiteNumber(snapshot?.iterations, 0),
    elapsedMs: finiteNumber(snapshot?.elapsedMs, 0),
    rootVisits: finiteNumber(snapshot?.rootVisits, 0),
    rootChildren: finiteNumber(snapshot?.rootChildren, 0),
    bestAction: compactBrowserAction(snapshot?.bestAction),
    topPolicy: (snapshot?.topPolicy || [])
      .slice(0, policyLimit)
      .map((entry) => ({
        action: compactBrowserAction(entry.action),
        visits: finiteNumber(entry.visits, 0),
        visitShare: rounded(entry.visitShare),
        q: rounded(entry.q),
        prior: rounded(entry.prior),
      })),
    principalVariation: (snapshot?.principalVariation || [])
      .slice(0, variationLimit)
      .map(compactBrowserVariationStep),
    candidateLines: (snapshot?.candidateLines || [])
      .slice(0, policyLimit)
      .map((line) => ({
        action: compactBrowserAction(line.action),
        visits: finiteNumber(line.visits, 0),
        visitShare: rounded(line.visitShare),
        q: rounded(line.q),
        prior: rounded(line.prior),
        principalVariation: (line.principalVariation || [])
          .slice(0, variationLimit)
          .map(compactBrowserVariationStep),
      })),
  };
}

function rounded(value) {
  const next = Number(value);
  if (!Number.isFinite(next)) return 0;
  return Number(next.toFixed(6));
}

function finiteNumber(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

export function createAlphaWorkerFactory({
  workerUrl = "./src/app/ai-alpha-worker.js?v=20260709-alpha-worker-2",
  WorkerCtor = globalThis.Worker,
} = {}) {
  return () => {
    if (typeof WorkerCtor !== "function") return null;
    return new WorkerCtor(workerUrl, { type: "module" });
  };
}
