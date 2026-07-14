import {
  evaluateAlphaPolicyModel,
  evaluateAlphaValueModel,
  flattenAlphaSelfPlaySamples,
  policyRowsFromSamples,
  trainAlphaModelFromSamples,
  validateAlphaTrainingSampleFeatureContracts,
} from "./ai-alpha-training.js";
import {
  buildAlphaReplayBuffer,
  samplesFromAlphaReplayBuffer,
  selectAlphaReplaySamples,
  summarizeAlphaReplayBuffer,
  summarizeAlphaReplaySampleSelection,
} from "./ai-alpha-replay-buffer.js";
import {
  ALPHA_REANALYSIS_BATCH_SCHEMA,
  alphaReanalysisSamplesFromInputs,
  runAlphaReanalysisBatch,
} from "./ai-alpha-reanalysis.js";
import {
  makeAlphaSelfPlayBatch,
  runAlphaSelfPlayBatch,
} from "./ai-self-play.js";

export function runAlphaTrainingIteration({
  scenario,
  rules,
  board = null,
  initialState = null,
  initialStates = null,
  baseModel = null,
  seed = 1942,
  games = 1,
  maxPlies = 80,
  searchOptions = {},
  selfPlayOptions = {},
  selfPlayBatch = null,
  replayInputs = [],
  replayBufferOptions = {},
  reanalysisOptions = {},
  reanalysisBatch = null,
  trainingOptions = {},
} = {}) {
  const reanalysisEnabled = isReanalysisEnabled(reanalysisOptions);
  const effectiveSelfPlayOptions = reanalysisEnabled
    ? { ...selfPlayOptions, includeStateSnapshots: true }
    : selfPlayOptions;
  const injectedSelfPlay = selfPlayBatch
    ? validateInjectedAlphaSelfPlayBatch(selfPlayBatch, {
      seed,
      games,
      maxPlies,
      requireStateSnapshots: Boolean(effectiveSelfPlayOptions.includeStateSnapshots),
    })
    : null;
  if (injectedSelfPlay && !injectedSelfPlay.ok) {
    throw new Error(`Injected Alpha self-play batch check failed: ${injectedSelfPlay.reason}`);
  }
  const selfPlay = injectedSelfPlay?.batch || runAlphaSelfPlayBatch({
    scenario,
    rules,
    board,
    initialState,
    initialStates,
    model: baseModel,
    seed,
    games,
    maxPlies,
    searchOptions,
    ...effectiveSelfPlayOptions,
  });
  const prepared = prepareAlphaTrainingIterationData({
    selfPlay,
    replayInputs,
    replayBufferOptions,
  });
  const {
    replayBuffer,
    sampleSelectionOptions,
    samples,
    reanalysisInputs,
  } = prepared;
  const reanalysisMaxSamples = Number(reanalysisOptions.maxSamples || samples.length || 1);
  if (reanalysisBatch && !reanalysisEnabled) {
    throw new Error("Injected Alpha reanalysis batch check failed: reanalysis_not_enabled");
  }
  const injectedReanalysis = reanalysisBatch
    ? validateInjectedAlphaReanalysisBatch(reanalysisBatch, {
      inputs: reanalysisInputs,
      seed,
      maxSamples: reanalysisMaxSamples,
      sides: reanalysisOptions.sides || reanalysisOptions.side,
      balanceBy: reanalysisOptions.balanceBy || replayBufferOptions.trainingSampleBalance || "none",
      priorityBy: reanalysisOptions.priorityBy || replayBufferOptions.trainingSamplePriority || "policyEntropy",
      includeStateSnapshots: reanalysisOptions.includeStateSnapshots !== false,
      searchOptions: reanalysisOptions.searchOptions || searchOptions,
    })
    : null;
  if (injectedReanalysis && !injectedReanalysis.ok) {
    throw new Error(`Injected Alpha reanalysis batch check failed: ${injectedReanalysis.reason}`);
  }
  const reanalysis = reanalysisEnabled
    ? injectedReanalysis?.batch || runAlphaReanalysisBatch({
      scenario,
      rules,
      board,
      inputs: reanalysisInputs,
      model: reanalysisOptions.model || baseModel,
      maxSamples: reanalysisMaxSamples,
      sides: reanalysisOptions.sides || reanalysisOptions.side,
      balanceBy: reanalysisOptions.balanceBy || replayBufferOptions.trainingSampleBalance || "none",
      priorityBy: reanalysisOptions.priorityBy || replayBufferOptions.trainingSamplePriority || "policyEntropy",
      seed,
      includeStateSnapshots: reanalysisOptions.includeStateSnapshots !== false,
      searchOptions: reanalysisOptions.searchOptions || searchOptions,
    })
    : null;
  const combinedTrainingSamples = reanalysis
    ? combineTrainingSamplesWithReanalysis(
      samples,
      applyAlphaReanalysisTrainingWeights(flattenAlphaSelfPlaySamples([reanalysis]), reanalysisOptions),
      reanalysisOptions.trainingMode,
    )
    : samples;
  const dualTrainingBalance = replayBufferOptions.trainingSampleValueWeightBalance
    || replayBufferOptions.trainingSamplePolicyWeightBalance;
  const valueTrainingBalance = applyAlphaTrainingBalanceWeights(combinedTrainingSamples, {
    balanceBy: dualTrainingBalance
      ? replayBufferOptions.trainingSampleValueWeightBalance || "none"
      : replayBufferOptions.trainingSampleWeightBalance || "none",
    weightField: dualTrainingBalance ? "valueTrainingWeight" : "trainingWeight",
    strength: replayBufferOptions.trainingSampleBalanceStrength,
    minMultiplier: replayBufferOptions.trainingSampleMinWeightMultiplier,
    maxMultiplier: replayBufferOptions.trainingSampleMaxWeightMultiplier,
  });
  const policyTrainingBalance = dualTrainingBalance
    ? applyAlphaTrainingBalanceWeights(valueTrainingBalance.samples, {
      balanceBy: replayBufferOptions.trainingSamplePolicyWeightBalance || "none",
      weightField: "policyTrainingWeight",
      strength: replayBufferOptions.trainingSampleBalanceStrength,
      minMultiplier: replayBufferOptions.trainingSampleMinWeightMultiplier,
      maxMultiplier: replayBufferOptions.trainingSampleMaxWeightMultiplier,
    })
    : null;
  const trainingBalance = dualTrainingBalance
    ? {
      schema: "zizi-el-alamein-alpha-training-dual-balance-v1",
      value: valueTrainingBalance.summary,
      policy: policyTrainingBalance.summary,
    }
    : valueTrainingBalance.summary;
  const trainingSamples = policyTrainingBalance?.samples || valueTrainingBalance.samples;
  const trainingReplayBuffer = reanalysis
    ? buildAlphaReplayBuffer([
      replayBuffer || { samples },
      reanalysis,
    ], {
      maxSamples: Number(replayBufferOptions.maxSamples || replayBuffer?.maxSamples || 4096),
      duplicateMode: normalizeReanalysisTrainingMode(reanalysisOptions.trainingMode) === "augment" ? "merge" : "latest",
      requireFeatureContract: replayBufferOptions.requireFeatureContract,
    })
    : replayBuffer;
  const model = trainAlphaModelFromSamples(trainingSamples, {
    scenario,
    rules,
    parentModel: baseModel,
    ...trainingOptionsWithBaseModel(trainingOptions, baseModel),
  });
  return {
    schema: "zizi-el-alamein-alpha-iteration-v1",
    generatedAt: new Date().toISOString(),
    seed,
    games: Number(games || 1),
    maxPlies,
    searchOptions: compactSearchOptions(searchOptions),
    selfPlayOptions: compactSelfPlayOptions({
      ...selfPlayOptions,
      initialStates: initialStates || selfPlayOptions.initialStates,
    }),
    reanalysisOptions: compactReanalysisOptions(reanalysisOptions, reanalysisEnabled),
    selfPlay: summarizeAlphaSelfPlayBatch(selfPlay),
    replayBuffer,
    replayBufferSummary: replayBuffer ? summarizeAlphaReplayBuffer(replayBuffer) : null,
    trainingReplayBuffer,
    trainingReplayBufferSummary: trainingReplayBuffer ? summarizeAlphaReplayBuffer(trainingReplayBuffer) : null,
    trainingSampleSelection: replayBuffer ? summarizeAlphaReplaySampleSelection(samples, sampleSelectionOptions) : null,
    trainingBalance,
    reanalysis,
    reanalysisSummary: reanalysis ? summarizeAlphaReanalysisBatch(reanalysis, reanalysisOptions) : null,
    training: summarizeAlphaTraining(trainingSamples, model, scenario),
    model,
  };
}

export function prepareAlphaTrainingIterationData({
  selfPlay,
  replayInputs = [],
  replayBufferOptions = {},
} = {}) {
  if (!selfPlay || selfPlay.schema !== "zizi-el-alamein-alpha-self-play-batch-v1") {
    throw new Error("Alpha iteration preparation requires a self-play batch");
  }
  const replayEnabled = replayInputs.length > 0 || replayBufferOptions.enabled === true;
  const replayBuffer = replayEnabled
    ? buildAlphaReplayBuffer([...replayInputs, { ...selfPlay, source: "current-self-play" }], replayBufferOptions)
    : null;
  if (replayBuffer?.quality && replayBufferOptions.requireFeatureContract && !replayBuffer.quality.ok) {
    const firstIssue = replayBuffer.quality.issues?.[0]?.reason;
    throw new Error(`Alpha replay buffer quality check failed: ${replayBuffer.quality.reason}${firstIssue ? ` (${firstIssue})` : ""}`);
  }
  const replaySamples = replayBuffer ? samplesFromAlphaReplayBuffer(replayBuffer) : null;
  const sampleSelectionOptions = {
    maxSamples: Number(replayBufferOptions.trainingSampleLimit || 0),
    sides: replayBufferOptions.trainingSampleSides || replayBufferOptions.trainingSampleSide,
    balanceBy: replayBufferOptions.trainingSampleBalance || "none",
    priorityBy: replayBufferOptions.trainingSamplePriority || "none",
    minSelectedVisitShare: optionalFiniteOrNull(replayBufferOptions.trainingSampleMinSelectedVisitShare),
  };
  const samples = replayBuffer
    ? selectAlphaReplaySamples(replaySamples, sampleSelectionOptions)
    : flattenAlphaSelfPlaySamples([selfPlay]);
  if (replayBuffer && !samples.length) {
    throw new Error("Alpha replay sample selection produced no training samples");
  }
  return {
    replayEnabled,
    replayBuffer,
    replaySamples,
    sampleSelectionOptions,
    samples,
    reanalysisInputs: replayBuffer ? [replayBuffer] : [{ ...selfPlay, source: "current-self-play" }],
  };
}

export function applyAlphaTrainingBalanceWeights(samples = [], options = {}) {
  const rows = Array.isArray(samples) ? samples : [];
  const balanceBy = options.balanceBy || "none";
  const weightField = normalizeTrainingWeightField(options.weightField);
  if (!rows.length || balanceBy === "none") {
    return {
      samples: rows,
      summary: trainingBalanceSummary(rows, balanceBy, false, null, weightField),
    };
  }
  const groups = new Map();
  for (const sample of rows) {
    const key = trainingBalanceKey(sample, balanceBy);
    groups.set(key, (groups.get(key) || 0) + 1);
  }
  const strength = clampFinite(options.strength, 0.5, 0, 1);
  const minMultiplier = clampFinite(options.minMultiplier, 0.25, 0.01, 4);
  const maxMultiplier = clampFinite(options.maxMultiplier, 4, minMultiplier, 16);
  const parentGroups = new Map();
  const parentCounts = new Map();
  for (const sample of rows) {
    const parent = trainingBalanceParentKey(sample, balanceBy);
    if (parent === null) continue;
    if (!parentGroups.has(parent)) parentGroups.set(parent, new Set());
    parentGroups.get(parent).add(trainingBalanceKey(sample, balanceBy));
    parentCounts.set(parent, (parentCounts.get(parent) || 0) + 1);
  }
  const targetCount = rows.length / Math.max(1, groups.size);
  const rawMultipliers = rows.map((sample) => (
    (trainingBalanceTargetCount(sample, balanceBy, targetCount, parentGroups, parentCounts)
      / groups.get(trainingBalanceKey(sample, balanceBy))) ** strength
  ));
  const rawAverages = trainingBalanceRawAverages(rows, rawMultipliers, balanceBy);
  const multipliers = rawMultipliers.map((value, index) => {
    const parent = trainingBalanceParentKey(rows[index], balanceBy);
    const rawAverage = rawAverages.get(parent) ?? rawAverages.get(null) ?? 1;
    return Math.min(maxMultiplier, Math.max(minMultiplier, value / Math.max(Number.EPSILON, rawAverage)));
  });
  const balanced = rows.map((sample, index) => ({
    ...sample,
    [weightField]: round(multipliers[index], 6),
  }));
  return {
    samples: balanced,
    summary: trainingBalanceSummary(balanced, balanceBy, true, groups, weightField),
  };
}

export function validateInjectedAlphaSelfPlayBatch(batch, options = {}) {
  if (!batch || typeof batch !== "object" || batch.schema !== "zizi-el-alamein-alpha-self-play-batch-v1") {
    return invalidInjectedBatch("invalid_self_play_batch_schema");
  }
  if (!Array.isArray(batch.results) || !batch.results.length) {
    return invalidInjectedBatch("missing_self_play_results");
  }
  const expectedGames = Math.max(1, Math.floor(Number(options.games || batch.results.length)));
  const expectedSeed = Number(options.seed ?? batch.seed ?? 1942);
  const expectedMaxPlies = Math.max(1, Math.floor(Number(options.maxPlies || batch.results[0]?.maxPlies || 1)));
  if (Number(batch.seed) !== expectedSeed) return invalidInjectedBatch("self_play_batch_seed_mismatch");
  if (batch.results.length !== expectedGames || Number(batch.games) !== expectedGames) {
    return invalidInjectedBatch("self_play_game_count_mismatch");
  }
  for (let index = 0; index < batch.results.length; index += 1) {
    const result = batch.results[index];
    if (!result || result.schema !== "zizi-el-alamein-alpha-self-play-v1") {
      return invalidInjectedBatch("invalid_self_play_game_schema", index);
    }
    if (Number(result.seed) !== expectedSeed + index) {
      return invalidInjectedBatch("self_play_seed_mismatch", index);
    }
    if (Number(result.maxPlies) !== expectedMaxPlies) {
      return invalidInjectedBatch("self_play_max_plies_mismatch", index);
    }
    if (!Array.isArray(result.samples) || !Array.isArray(result.actions) || !Array.isArray(result.errors)) {
      return invalidInjectedBatch("invalid_self_play_game_collections", index);
    }
    if (result.errors.length) return invalidInjectedBatch("self_play_game_has_errors", index);
    if (
      options.requireStateSnapshots
      && (
        result.includeStateSnapshots !== true
        || result.samples.some((sample) => !sample?.initialState)
      )
    ) {
      return invalidInjectedBatch("self_play_state_snapshots_required", index);
    }
  }
  const canonical = makeAlphaSelfPlayBatch(batch.results, {
    seed: expectedSeed,
    generatedAt: batch.generatedAt,
  });
  for (const key of ["sampleCount", "actionCount", "errorCount", "averageOutcomeWeight"]) {
    if (Number(batch[key]) !== Number(canonical[key])) {
      return invalidInjectedBatch(`self_play_${key}_mismatch`);
    }
  }
  if (
    batch.uniqueInitialPositions !== undefined
    && Number(batch.uniqueInitialPositions || 0) !== Number(canonical.uniqueInitialPositions || 0)
  ) {
    return invalidInjectedBatch("self_play_unique_initial_positions_mismatch");
  }
  if (stableSummaryValue(batch.wins) !== stableSummaryValue(canonical.wins)) {
    return invalidInjectedBatch("self_play_win_summary_mismatch");
  }
  if (stableSummaryValue(batch.sampleOutcomeSources) !== stableSummaryValue(canonical.sampleOutcomeSources)) {
    return invalidInjectedBatch("self_play_outcome_summary_mismatch");
  }
  if (
    batch.initialSides !== undefined
    && stableSummaryValue(batch.initialSides) !== stableSummaryValue(canonical.initialSides)
  ) {
    return invalidInjectedBatch("self_play_initial_side_summary_mismatch");
  }
  if (
    batch.initialPhases !== undefined
    && stableSummaryValue(batch.initialPhases) !== stableSummaryValue(canonical.initialPhases)
  ) {
    return invalidInjectedBatch("self_play_initial_phase_summary_mismatch");
  }
  return { ok: true, reason: null, gameIndex: null, batch: canonical };
}

export function validateInjectedAlphaReanalysisBatch(batch, options = {}) {
  if (!batch || typeof batch !== "object" || batch.schema !== ALPHA_REANALYSIS_BATCH_SCHEMA) {
    return invalidInjectedReanalysis("invalid_reanalysis_batch_schema");
  }
  const inputs = Array.isArray(options.inputs) ? options.inputs : [];
  const eligible = alphaReanalysisSamplesFromInputs(inputs);
  const maxSamples = Math.max(0, Math.floor(Number(options.maxSamples || 0)));
  const sides = normalizeReanalysisSides(options.sides || options.side);
  const balanceBy = options.balanceBy || "none";
  const priorityBy = options.priorityBy || "policyEntropy";
  const selected = selectAlphaReplaySamples(eligible, { maxSamples, sides, balanceBy, priorityBy });
  if (Number(batch.seed) !== Number(options.seed ?? batch.seed ?? 1942)) {
    return invalidInjectedReanalysis("reanalysis_batch_seed_mismatch");
  }
  if (Number(batch.inputCount) !== inputs.length) return invalidInjectedReanalysis("reanalysis_input_count_mismatch");
  if (Number(batch.eligibleSamples) !== eligible.length) return invalidInjectedReanalysis("reanalysis_eligible_sample_count_mismatch");
  if (Number(batch.maxSamples) !== maxSamples) return invalidInjectedReanalysis("reanalysis_max_samples_mismatch");
  if (JSON.stringify(batch.sides || []) !== JSON.stringify(sides)) {
    return invalidInjectedReanalysis("reanalysis_sides_mismatch");
  }
  if (Number(batch.selectedSamples) !== selected.length) return invalidInjectedReanalysis("reanalysis_selected_sample_count_mismatch");
  if (batch.balanceBy !== balanceBy) return invalidInjectedReanalysis("reanalysis_balance_mode_mismatch");
  if (batch.priorityBy !== priorityBy) return invalidInjectedReanalysis("reanalysis_priority_mode_mismatch");
  if (Boolean(batch.includeStateSnapshots) !== Boolean(options.includeStateSnapshots !== false)) {
    return invalidInjectedReanalysis("reanalysis_snapshot_mode_mismatch");
  }
  if (!reanalysisSearchMatches(batch.search, options.searchOptions || {})) {
    return invalidInjectedReanalysis("reanalysis_search_options_mismatch");
  }
  if (!Array.isArray(batch.samples) || !Array.isArray(batch.errors)) {
    return invalidInjectedReanalysis("invalid_reanalysis_batch_collections");
  }
  if (Number(batch.sampleCount) !== batch.samples.length) {
    return invalidInjectedReanalysis("reanalysis_sample_count_mismatch");
  }
  if (batch.samples.length + batch.errors.length !== selected.length) {
    return invalidInjectedReanalysis("reanalysis_result_count_mismatch");
  }
  const expectedKeys = selected.map(trainingSampleKey);
  const resultKeys = batch.samples.map(reanalysisSourceSampleKey);
  for (const error of batch.errors) {
    const index = Number(error?.index);
    if (!Number.isInteger(index) || index < 0 || index >= expectedKeys.length) {
      return invalidInjectedReanalysis("reanalysis_error_index_invalid");
    }
    resultKeys.push(expectedKeys[index]);
  }
  if (stableKeyCounts(expectedKeys) !== stableKeyCounts(resultKeys)) {
    return invalidInjectedReanalysis("reanalysis_source_samples_mismatch");
  }
  for (const sample of batch.samples) {
    if (
      sample?.schema !== "zizi-el-alamein-alpha-training-sample-v1"
      || sample?.reanalysis?.schema !== "zizi-el-alamein-alpha-reanalysis-sample-v1"
      || sample.reanalysis.stateHashMatches === false
    ) {
      return invalidInjectedReanalysis("invalid_reanalysis_sample");
    }
    if (options.includeStateSnapshots !== false && !sample.initialState) {
      return invalidInjectedReanalysis("reanalysis_state_snapshots_required");
    }
  }
  return { ok: true, reason: null, batch };
}

function invalidInjectedReanalysis(reason) {
  return { ok: false, reason, batch: null };
}

function normalizeReanalysisSides(value) {
  const values = Array.isArray(value) ? value : String(value || "").split(",");
  return [...new Set(values.map((side) => String(side).trim().toLowerCase()).filter((side) => (
    side === "axis" || side === "allied"
  )))].sort();
}

function invalidInjectedBatch(reason, gameIndex = null) {
  return { ok: false, reason, gameIndex, batch: null };
}

function stableSummaryValue(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "{}";
  return JSON.stringify(Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right))));
}

function reanalysisSearchMatches(actual = {}, expected = {}) {
  const nullableKeys = [
    "iterations",
    "maxDepth",
    "actionLimit",
    "preApplyLimit",
    "policyWeight",
    "phasePlanWeight",
    "phasePlanNodeLimit",
    "phasePlanBeamWidth",
    "gumbelScale",
    "gumbelPriorScale",
    "gumbelValueScale",
  ];
  for (const key of nullableKeys) {
    const actualValue = optionalFiniteOrNull(actual?.[key]);
    const expectedValue = optionalFiniteOrNull(expected?.[key]);
    if (actualValue !== expectedValue) return false;
  }
  if (String(actual?.rootSelectionMode || "puct") !== String(expected?.rootSelectionMode || "puct")) return false;
  if (String(actual?.gumbelSeed ?? "") !== String(expected?.gumbelSeed ?? "")) return false;
  return finiteOrDefault(actual?.rootNoiseWeight, 0) === finiteOrDefault(expected?.rootNoiseWeight, 0);
}

function reanalysisSourceSampleKey(sample) {
  return [
    sample?.reanalysis?.sourceStateHash || "no-state",
    sample?.reanalysis?.sourceSide || "no-side",
    sample?.reanalysis?.sourcePhaseId || "no-phase",
    sample?.reanalysis?.sourceTurn ?? "no-turn",
  ].join("|");
}

function stableKeyCounts(keys = []) {
  const counts = {};
  for (const key of keys) counts[key] = (counts[key] || 0) + 1;
  return JSON.stringify(Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right))));
}

export function summarizeAlphaReanalysisBatch(batch, options = {}) {
  return {
    schema: "zizi-el-alamein-alpha-reanalysis-summary-v1",
    enabled: Boolean(batch),
    eligibleSamples: Number(batch?.eligibleSamples || 0),
    selectedSamples: Number(batch?.selectedSamples || 0),
    sampleCount: Number(batch?.sampleCount || 0),
    errors: Array.isArray(batch?.errors) ? batch.errors.length : 0,
    balanceBy: batch?.balanceBy || options.balanceBy || "none",
    priorityBy: batch?.priorityBy || options.priorityBy || "policyEntropy",
    trainingMode: normalizeReanalysisTrainingMode(options.trainingMode),
    policyWeightMultiplier: reanalysisWeightMultiplier(options.policyWeightMultiplier, 1),
    valueWeightMultiplier: reanalysisWeightMultiplier(options.valueWeightMultiplier, 1),
    averagePolicyEntropy: averageDecisionPolicyEntropy(batch?.samples || []),
  };
}

export function applyAlphaReanalysisTrainingWeights(samples = [], options = {}) {
  const policyMultiplier = reanalysisWeightMultiplier(options.policyWeightMultiplier, 1);
  const valueMultiplier = reanalysisWeightMultiplier(options.valueWeightMultiplier, 1);
  return (samples || []).map((sample) => ({
    ...sample,
    policyTrainingWeight: reanalysisWeightedSampleValue(sample, "policyTrainingWeight", policyMultiplier),
    valueTrainingWeight: reanalysisWeightedSampleValue(sample, "valueTrainingWeight", valueMultiplier),
  }));
}

export function evaluateAlphaModelOnSelfPlay(model, selfPlay, options = {}) {
  const samples = flattenAlphaSelfPlaySamples([selfPlay]);
  return summarizeAlphaTraining(samples, model, options.scenario || null);
}

export function summarizeAlphaSelfPlayBatch(batch) {
  const results = batch?.results || [];
  const plies = results.map((result) => Number(result.plies || 0));
  return {
    schema: "zizi-el-alamein-alpha-self-play-summary-v1",
    games: Number(batch?.games || results.length || 0),
    wins: { ...(batch?.wins || {}) },
    initialSides: { ...(batch?.initialSides || {}) },
    initialPhases: { ...(batch?.initialPhases || {}) },
    uniqueInitialPositions: Number(batch?.uniqueInitialPositions || 0),
    sampleCount: Number(batch?.sampleCount || 0),
    sampleOutcomeSources: { ...(batch?.sampleOutcomeSources || {}) },
    averageOutcomeWeight: Number(batch?.averageOutcomeWeight || 0),
    includeStateSnapshots: Boolean(batch?.includeStateSnapshots || results.some((result) => result.includeStateSnapshots)),
    actionCount: Number(batch?.actionCount || 0),
    errorCount: Number(batch?.errorCount || 0),
    guardHits: results.filter((result) => result.guardHit).length,
    averagePlies: plies.length ? round(plies.reduce((sum, value) => sum + value, 0) / plies.length, 3) : 0,
    maxPlies: plies.length ? Math.max(...plies) : 0,
  };
}

export function summarizeAlphaTraining(samples, model, scenario = null) {
  const rows = policyRowsFromSamples(samples, { scenario });
  const featureContracts = validateAlphaTrainingSampleFeatureContracts(samples);
  return {
    schema: "zizi-el-alamein-alpha-training-summary-v1",
    samples: samples.length,
    samplesWithFeatureContract: featureContracts.samplesWithFeatureContract,
    featureContractMissingSamples: featureContracts.missingSampleCount,
    featureContractMismatchedSamples: featureContracts.mismatchedSampleCount,
    featureContracts: featureContracts.featureContracts,
    expectedFeatureContractFingerprint: featureContracts.expectedFingerprint,
    value: evaluateAlphaValueModel(samples, model?.value || {}),
    policy: evaluateAlphaPolicyModel(rows, model?.policy || {}),
  };
}

function compactSearchOptions(options = {}) {
  return {
    iterations: options.iterations ?? null,
    maxDepth: options.maxDepth ?? options.depth ?? null,
    actionLimit: options.actionLimit ?? null,
    preApplyLimit: options.preApplyLimit ?? null,
    policyWeight: options.policyWeight ?? null,
    rootSelectionMode: options.rootSelectionMode ?? "puct",
    gumbelScale: options.gumbelScale ?? null,
    gumbelPriorScale: options.gumbelPriorScale ?? null,
    gumbelValueScale: options.gumbelValueScale ?? null,
    gumbelSeed: options.gumbelSeed ?? null,
    phasePlanWeight: options.phasePlanWeight ?? null,
    phasePlanNodeLimit: options.phasePlanNodeLimit ?? null,
    phasePlanBeamWidth: options.phasePlanBeamWidth ?? null,
    phasePlanCandidateLimit: options.phasePlanCandidateLimit ?? null,
    phasePlanMaxActions: options.phasePlanMaxActions ?? null,
    phasePlanProjectionBeams: options.phasePlanProjectionBeams ?? null,
  };
}

function compactSelfPlayOptions(options = {}) {
  return {
    temperature: options.temperature ?? null,
    temperaturePlies: options.temperaturePlies ?? null,
    rootNoiseWeight: options.rootNoiseWeight ?? null,
    rootDirichletAlpha: options.rootDirichletAlpha ?? null,
    rootNoisePlies: options.rootNoisePlies ?? null,
    guardOutcomeMode: options.guardOutcomeMode ?? null,
    guardOutcomeWeight: options.guardOutcomeWeight ?? null,
    guardOutcomeDiscount: options.guardOutcomeDiscount ?? null,
    includeStateSnapshots: Boolean(options.includeStateSnapshots),
    initialStateCount: Array.isArray(options.initialStates) ? options.initialStates.length : 0,
  };
}

function compactReanalysisOptions(options = {}, enabled = false) {
  return {
    enabled,
    maxSamples: options.maxSamples ?? null,
    sides: normalizeReanalysisSides(options.sides || options.side),
    balanceBy: options.balanceBy ?? null,
    priorityBy: options.priorityBy ?? null,
    trainingMode: normalizeReanalysisTrainingMode(options.trainingMode),
    policyWeightMultiplier: reanalysisWeightMultiplier(options.policyWeightMultiplier, 1),
    valueWeightMultiplier: reanalysisWeightMultiplier(options.valueWeightMultiplier, 1),
    includeStateSnapshots: options.includeStateSnapshots !== false,
    searchOptions: options.searchOptions ? compactSearchOptions(options.searchOptions) : null,
  };
}

function trainingOptionsWithBaseModel(options = {}, baseModel = null) {
  return {
    ...options,
    value: {
      baseWeights: baseModel?.value?.weights,
      baseNetwork: baseModel?.value?.network,
      architecture: baseModel?.value?.architecture,
      ...(options.value || {}),
    },
    policy: {
      baseWeights: baseModel?.policy?.weights,
      baseNetwork: baseModel?.policy?.network,
      architecture: baseModel?.policy?.architecture,
      ...(options.policy || {}),
    },
    hexGraph: {
      baseModel: baseModel?.hexGraph,
      ...(options.hexGraph || {}),
    },
  };
}

function isReanalysisEnabled(options = {}) {
  if (options.maxSamples !== undefined && Number(options.maxSamples) <= 0) return false;
  return options.enabled === true || Number(options.maxSamples || 0) > 0;
}

function combineTrainingSamplesWithReanalysis(samples, reanalysisSamples, modeValue = "replace") {
  const mode = normalizeReanalysisTrainingMode(modeValue);
  if (!reanalysisSamples.length) return samples;
  if (mode === "augment") return [...samples, ...reanalysisSamples];
  const replacements = new Map(reanalysisSamples.map((sample) => [trainingSampleKey(sample), sample]));
  const used = new Set();
  const combined = samples.map((sample) => {
    const key = trainingSampleKey(sample);
    const replacement = replacements.get(key);
    if (!replacement) return sample;
    used.add(key);
    return replacement;
  });
  for (const sample of reanalysisSamples) {
    const key = trainingSampleKey(sample);
    if (!used.has(key)) combined.push(sample);
  }
  return combined;
}

function reanalysisWeightedSampleValue(sample, field, multiplier) {
  const base = Number(sample?.[field] ?? sample?.trainingWeight ?? 1);
  return round(Math.min(4, Math.max(0, (Number.isFinite(base) ? base : 1) * multiplier)), 6);
}

function reanalysisWeightMultiplier(value, fallback) {
  if (value === null || value === undefined || value === "") return fallback;
  const next = Number(value);
  return Math.min(4, Math.max(0, Number.isFinite(next) ? next : fallback));
}

function trainingSampleKey(sample) {
  return [
    sample?.stateHash || "no-state",
    sample?.side || "no-side",
    sample?.phaseId || "no-phase",
    sample?.turn ?? "no-turn",
  ].join("|");
}

function trainingBalanceKey(sample, balanceBy) {
  const side = sample?.side || "unknown-side";
  const phase = sample?.phaseId || "unknown-phase";
  const turn = trainingTurnBucket(sample?.turn);
  const outcome = Number(sample?.outcome);
  const outcomeBucket = outcome > 0 ? "positive" : outcome < 0 ? "negative" : "zero";
  if (balanceBy === "side") return side;
  if (balanceBy === "phase") return phase;
  if (balanceBy === "outcome") return outcomeBucket;
  if (balanceBy === "sidePhase") return `${side}|${phase}`;
  if (balanceBy === "sideOutcome") return `${side}|${outcomeBucket}`;
  if (balanceBy === "sidePhaseOutcome") return `${side}|${phase}|${outcomeBucket}`;
  if (balanceBy === "turnWithinPhase") return `${phase}|${turn}`;
  if (balanceBy === "turnWithinSidePhase") return `${side}|${phase}|${turn}`;
  return "all";
}

function trainingBalanceParentKey(sample, balanceBy) {
  const phase = sample?.phaseId || "unknown-phase";
  if (balanceBy === "turnWithinPhase") return phase;
  if (balanceBy === "turnWithinSidePhase") return `${sample?.side || "unknown-side"}|${phase}`;
  return null;
}

function trainingBalanceTargetCount(sample, balanceBy, fallback, parentGroups, parentCounts) {
  const parent = trainingBalanceParentKey(sample, balanceBy);
  if (parent === null) return fallback;
  return Number(parentCounts.get(parent) || 0) / Math.max(1, parentGroups.get(parent)?.size || 0);
}

function trainingBalanceRawAverages(samples, rawMultipliers, balanceBy) {
  const totals = new Map();
  const counts = new Map();
  for (let index = 0; index < samples.length; index += 1) {
    const parent = trainingBalanceParentKey(samples[index], balanceBy);
    totals.set(parent, (totals.get(parent) || 0) + rawMultipliers[index]);
    counts.set(parent, (counts.get(parent) || 0) + 1);
  }
  return new Map([...totals.entries()].map(([parent, total]) => [
    parent,
    total / Math.max(1, counts.get(parent) || 0),
  ]));
}

function trainingTurnBucket(value) {
  const turn = Number(value);
  return Number.isFinite(turn) ? `turn-${Math.floor(turn)}` : "unknown-turn";
}

function trainingBalanceSummary(samples, balanceBy, enabled, sourceGroups = null, weightField = "trainingWeight") {
  const weights = (samples || []).map((sample) => Number(sample?.[weightField] ?? 1)).filter(Number.isFinite);
  const weightedGroups = {};
  for (const sample of samples || []) {
    const key = trainingBalanceKey(sample, balanceBy);
    weightedGroups[key] = round((weightedGroups[key] || 0) + Number(sample?.[weightField] ?? 1), 6);
  }
  return {
    schema: "zizi-el-alamein-alpha-training-balance-v1",
    enabled: Boolean(enabled),
    balanceBy,
    weightField,
    samples: samples.length,
    groups: sourceGroups ? Object.fromEntries([...sourceGroups.entries()].sort()) : {},
    weightedGroups,
    averageMultiplier: weights.length ? round(weights.reduce((sum, value) => sum + value, 0) / weights.length, 6) : 0,
    minMultiplier: weights.length ? Math.min(...weights) : 0,
    maxMultiplier: weights.length ? Math.max(...weights) : 0,
  };
}

function normalizeTrainingWeightField(value) {
  return ["trainingWeight", "valueTrainingWeight", "policyTrainingWeight"].includes(value)
    ? value
    : "trainingWeight";
}

function clampFinite(value, fallback, minimum, maximum) {
  if (value === null || value === undefined || value === "") return fallback;
  const next = Number(value);
  return Math.min(maximum, Math.max(minimum, Number.isFinite(next) ? next : fallback));
}

function normalizeReanalysisTrainingMode(value) {
  return value === "augment" ? "augment" : "replace";
}

function averageDecisionPolicyEntropy(samples) {
  const values = (samples || [])
    .map((sample) => Number(sample?.decision?.policyEntropy))
    .filter(Number.isFinite);
  if (!values.length) return 0;
  return round(values.reduce((sum, value) => sum + value, 0) / values.length, 6);
}

function optionalFiniteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function finiteOrDefault(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
