import { createBoard, createEnvironment } from "../core/index.js";
import { canonicalSerialize } from "../../../shared/wargame-alpha/environment-contract.js";
import { createElAlameinAlphaEnvironmentAdapter } from "./ai-alpha-environment-adapter.js";
import { analyzePosition } from "./ai-alpha-search.js";
import { searchElAlameinAlpha } from "./ai-alpha-search-bridge.js";
import { normalizeAlphaModelArtifact } from "./ai-alpha-model.js";
import { analyzeSituation } from "./ai-situation.js";

export const ALPHA_RUNTIME_ENGINE = Object.freeze({
  LEGACY: "legacy",
  GENERIC: "generic-alpha-v1",
});

export const GENERIC_ALPHA_RUNTIME_LIMITS = Object.freeze({
  simulations: 4,
  maxSimulations: 32,
  maxDepth: 1,
  maxAllowedDepth: 2,
  actionLimit: 32,
  maxActionLimit: 32,
  exploration: 1.35,
  policyTemperature: 1,
});

export const ALPHA_RUNTIME_MESSAGE = Object.freeze({
  ANALYZE: "ANALYZE_ALPHA_POSITION",
  CHOOSE_ACTION: "CHOOSE_ALPHA_ACTION",
});

export function alphaSearchOptionsFromModel(model = null, options = {}) {
  return {
    ...options,
    model: normalizeAlphaModel(model),
  };
}

export function analyzeAlphaPosition({
  scenario,
  rules,
  board = null,
  state,
  model = null,
  searchOptions = {},
} = {}) {
  const environment = createEnvironment({
    scenario,
    rules,
    board: board || createBoard(scenario),
    state,
  });
  if (searchOptions.engine === ALPHA_RUNTIME_ENGINE.GENERIC) {
    return analyzeGenericAlphaPosition(environment, model, searchOptions);
  }
  return analyzePosition(environment, alphaSearchOptionsFromModel(model, searchOptions));
}

export function genericAlphaSearchOptionsFromModel(model = null, options = {}) {
  const normalizedModel = normalizeAlphaModel(model);
  return {
    simulations: boundedInteger(options.simulations, GENERIC_ALPHA_RUNTIME_LIMITS.simulations, 1, GENERIC_ALPHA_RUNTIME_LIMITS.maxSimulations),
    maxDepth: boundedInteger(options.maxDepth, GENERIC_ALPHA_RUNTIME_LIMITS.maxDepth, 0, GENERIC_ALPHA_RUNTIME_LIMITS.maxAllowedDepth),
    actionLimit: boundedInteger(options.actionLimit, GENERIC_ALPHA_RUNTIME_LIMITS.actionLimit, 1, GENERIC_ALPHA_RUNTIME_LIMITS.maxActionLimit),
    exploration: boundedNumber(options.exploration, GENERIC_ALPHA_RUNTIME_LIMITS.exploration, 0, 4),
    policyTemperature: boundedNumber(options.policyTemperature, GENERIC_ALPHA_RUNTIME_LIMITS.policyTemperature, 0.1, 4),
    valueModel: normalizedModel?.value || null,
    policyModel: normalizedModel?.policy || null,
    hexGraphModel: normalizedModel?.hexGraph || null,
  };
}

export function analyzeGenericAlphaPosition(environment, model = null, searchOptions = {}) {
  const adapter = createElAlameinAlphaEnvironmentAdapter({
    scenario: environment.scenario,
    rules: environment.rules,
    board: environment.board,
  });
  const options = genericAlphaSearchOptionsFromModel(model, searchOptions);
  const started = runtimeNow();
  const result = searchElAlameinAlpha(environment, { ...options, adapter });
  const elapsedMs = Math.max(0, runtimeNow() - started);
  const side = result.nodeKind === "decision" ? adapter.currentPlayer(environment.state) : null;
  const rootValue = result.rootValues.find((entry) => entry.playerId === side)?.value ?? null;
  const rootVisits = result.policy.reduce((sum, entry) => sum + Number(entry.visits || 0), 0);
  const situation = side ? analyzeSituation(environment, { side }) : null;
  const policy = result.policy.map((entry) => {
    const value = entry.values?.find((item) => item.playerId === side)?.value ?? null;
    return {
      actionKey: entry.actionKey,
      action: entry.action,
      prior: entry.prior,
      visits: entry.visits,
      visitShare: rootVisits > 0 ? entry.visits / rootVisits : 0,
      q: value,
      value,
    };
  }).sort((left, right) => (
    right.visits - left.visits
    || Number(right.q || 0) - Number(left.q || 0)
    || right.prior - left.prior
    || codeUnitCompare(canonicalSerialize(left.actionKey), canonicalSerialize(right.actionKey))
  ));
  return {
    schema: "zizi-el-alamein-alpha-generic-analysis-v1",
    engine: ALPHA_RUNTIME_ENGINE.GENERIC,
    side,
    stateHash: adapter.stateHash(environment.state),
    nodeKind: result.nodeKind,
    rootStateKey: result.rootStateKey,
    rootValue,
    rootValues: result.rootValues,
    bestAction: result.selectedAction,
    selectedActionKey: result.selectedActionKey,
    policy,
    chanceOutcomes: result.chanceOutcomes,
    situation,
    recommendation: null,
    principalVariation: [],
    candidateLines: [],
    requiresChance: result.nodeKind === "chance",
    search: {
      engine: ALPHA_RUNTIME_ENGINE.GENERIC,
      iterations: result.simulations,
      simulations: result.simulations,
      rootVisits,
      rootChildren: result.policy.length,
      maxDepth: result.maxDepth,
      actionLimit: result.actionLimit,
      preApplyLimit: 0,
      elapsedMs,
    },
  };
}

export function chooseAlphaRuntimeAction({
  scenario,
  rules,
  board = null,
  state,
  model = null,
  searchOptions = {},
} = {}) {
  return chooseAlphaRuntimeDecision({
    scenario,
    rules,
    board,
    state,
    model,
    searchOptions,
  }).action;
}

export function chooseAlphaRuntimeDecision({
  scenario,
  rules,
  board = null,
  state,
  model = null,
  searchOptions = {},
} = {}) {
  const analysis = analyzeAlphaPosition({
    scenario,
    rules,
    board,
    state,
    model,
    searchOptions,
  });
  return {
    action: analysis.requiresChance ? null : analysis.bestAction,
    analysis,
    decision: summarizeAlphaRuntimeDecision(analysis),
  };
}

export function summarizeAlphaRuntimeDecision(analysis, options = {}) {
  const candidateLimit = Math.max(1, Number(options.candidateLimit || 4));
  const pvLimit = Math.max(0, Number(options.pvLimit || 4));
  const featureLimit = Math.max(0, Number(options.featureLimit || 6));
  const reason = alphaRuntimeDecisionReason(analysis);
  const recommendation = compactAlphaRuntimeRecommendation(analysis?.recommendation);
  return {
    schema: "zizi-el-alamein-alpha-runtime-decision-v1",
    engine: analysis?.engine || ALPHA_RUNTIME_ENGINE.LEGACY,
    ok: reason === null,
    reason,
    side: analysis?.side || null,
    stateHash: analysis?.stateHash || null,
    turn: analysis?.situation?.turn ?? null,
    phaseId: analysis?.situation?.phaseId || null,
    action: reason === null ? compactAlphaRuntimeAction(analysis.bestAction) : null,
    rootValue: rounded(analysis?.rootValue),
    recommendation,
    confidence: recommendation?.confidence ?? null,
    candidates: summarizeAlphaRuntimeCandidates(analysis, {
      candidateLimit,
      pvLimit,
    }),
    principalVariation: (analysis?.principalVariation || [])
      .slice(0, pvLimit)
      .map(compactAlphaRuntimeVariationStep)
      .filter(Boolean),
    features: summarizeAlphaRuntimeFeatures(analysis?.situation?.features, featureLimit),
    search: {
      iterations: finiteNumber(analysis?.search?.iterations, 0),
      rootVisits: finiteNumber(analysis?.search?.rootVisits, 0),
      rootChildren: finiteNumber(analysis?.search?.rootChildren, 0),
      maxDepth: finiteNumber(analysis?.search?.maxDepth, 0),
      actionLimit: finiteNumber(analysis?.search?.actionLimit, 0),
      preApplyLimit: finiteNumber(analysis?.search?.preApplyLimit, 0),
      elapsedMs: finiteNumber(analysis?.search?.elapsedMs, 0),
    },
    requiresChance: Boolean(analysis?.requiresChance),
  };
}

export function handleAlphaRuntimeMessage(message) {
  try {
    if (!message || typeof message !== "object") return errorResponse(message, "invalid_message");
    if (message.type === ALPHA_RUNTIME_MESSAGE.ANALYZE) {
      const analysis = analyzeAlphaPosition(message.payload || {});
      return {
        id: message.id || null,
        type: `${message.type}_RESULT`,
        ok: true,
        analysis,
        decision: summarizeAlphaRuntimeDecision(analysis),
      };
    }
    if (message.type === ALPHA_RUNTIME_MESSAGE.CHOOSE_ACTION) {
      const decision = chooseAlphaRuntimeDecision(message.payload || {});
      return {
        id: message.id || null,
        type: `${message.type}_RESULT`,
        ok: true,
        action: decision.action,
        analysis: decision.analysis,
        decision: decision.decision,
      };
    }
    return errorResponse(message, "unknown_message_type");
  } catch (error) {
    return errorResponse(message, "alpha_runtime_failed", error);
  }
}

function alphaRuntimeDecisionReason(analysis) {
  if (!analysis || typeof analysis !== "object") return "missing_analysis";
  if (analysis.requiresChance) return "requires_chance";
  if (!analysis.bestAction) return "missing_best_action";
  return null;
}

function summarizeAlphaRuntimeCandidates(analysis, options = {}) {
  const candidateLimit = Math.max(1, Number(options.candidateLimit || 4));
  const pvLimit = Math.max(0, Number(options.pvLimit || 4));
  const source = Array.isArray(analysis?.candidateLines) && analysis.candidateLines.length
    ? analysis.candidateLines
    : (analysis?.policy || []);
  return source
    .slice(0, candidateLimit)
    .map((entry, index) => ({
      rank: index + 1,
      action: compactAlphaRuntimeAction(entry.action),
      visits: finiteNumber(entry.visits, 0),
      visitShare: rounded(entry.visitShare),
      q: rounded(entry.q),
      prior: rounded(entry.prior),
      value: entry.value === undefined ? null : rounded(entry.value),
      principalVariation: (entry.principalVariation || [])
        .slice(0, pvLimit)
        .map(compactAlphaRuntimeVariationStep)
        .filter(Boolean),
    }))
    .filter((entry) => entry.action);
}

function compactAlphaRuntimeRecommendation(recommendation) {
  if (!recommendation || typeof recommendation !== "object") return null;
  return {
    schema: "zizi-el-alamein-alpha-runtime-recommendation-v1",
    action: compactAlphaRuntimeAction(recommendation.action),
    label: typeof recommendation.label === "string" ? recommendation.label : "unknown",
    confidence: rounded(recommendation.confidence),
    bestVisitShare: rounded(recommendation.bestVisitShare),
    runnerUpVisitShare: rounded(recommendation.runnerUpVisitShare),
    visitMargin: rounded(recommendation.visitMargin),
    qMargin: recommendation.qMargin === null || recommendation.qMargin === undefined
      ? null
      : rounded(recommendation.qMargin),
    priorMargin: recommendation.priorMargin === null || recommendation.priorMargin === undefined
      ? null
      : rounded(recommendation.priorMargin),
    entropy: rounded(recommendation.entropy),
    choices: finiteNumber(recommendation.choices, 0),
  };
}

function compactAlphaRuntimeVariationStep(step) {
  if (!step || typeof step !== "object") return null;
  return {
    action: compactAlphaRuntimeAction(step.action || step),
    visits: finiteNumber(step.visits, 0),
    q: rounded(step.q),
  };
}

function compactAlphaRuntimeAction(action) {
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
  return compact.type ? compact : null;
}

function summarizeAlphaRuntimeFeatures(features, limit) {
  if (!features || typeof features !== "object" || limit <= 0) return [];
  return Object.entries(features)
    .map(([key, value]) => ({
      key,
      value: rounded(value),
      magnitude: Math.abs(Number(value) || 0),
    }))
    .filter((entry) => Number.isFinite(entry.value))
    .sort((left, right) => right.magnitude - left.magnitude || left.key.localeCompare(right.key))
    .slice(0, limit)
    .map(({ key, value }) => ({ key, value }));
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

function normalizeAlphaModel(model) {
  return normalizeAlphaModelArtifact(model);
}

function boundedInteger(value, fallback, minimum, maximum) {
  const number = Number(value);
  const normalized = Number.isInteger(number) ? number : fallback;
  return Math.max(minimum, Math.min(maximum, normalized));
}

function boundedNumber(value, fallback, minimum, maximum) {
  const number = Number(value);
  const normalized = Number.isFinite(number) ? number : fallback;
  return Math.max(minimum, Math.min(maximum, normalized));
}

function runtimeNow() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function codeUnitCompare(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function errorResponse(message, reason, error = null) {
  return {
    id: message?.id || null,
    type: `${message?.type || "ALPHA_RUNTIME"}_RESULT`,
    ok: false,
    reason,
    message: error ? String(error?.message || error) : undefined,
  };
}
