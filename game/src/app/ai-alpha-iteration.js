import {
  evaluateAlphaPolicyModel,
  evaluateAlphaValueModel,
  flattenAlphaSelfPlaySamples,
  policyRowsFromSamples,
  trainAlphaModelFromSamples,
} from "./ai-alpha-training.js";
import {
  buildAlphaReplayBuffer,
  samplesFromAlphaReplayBuffer,
  selectAlphaReplaySamples,
  summarizeAlphaReplayBuffer,
  summarizeAlphaReplaySampleSelection,
} from "./ai-alpha-replay-buffer.js";
import {
  runAlphaReanalysisBatch,
} from "./ai-alpha-reanalysis.js";
import { runAlphaSelfPlayBatch } from "./ai-self-play.js";

export function runAlphaTrainingIteration({
  scenario,
  rules,
  board = null,
  initialState = null,
  baseModel = null,
  seed = 1942,
  games = 1,
  maxPlies = 80,
  searchOptions = {},
  selfPlayOptions = {},
  replayInputs = [],
  replayBufferOptions = {},
  reanalysisOptions = {},
  trainingOptions = {},
} = {}) {
  const reanalysisEnabled = isReanalysisEnabled(reanalysisOptions);
  const effectiveSelfPlayOptions = reanalysisEnabled
    ? { ...selfPlayOptions, includeStateSnapshots: true }
    : selfPlayOptions;
  const selfPlay = runAlphaSelfPlayBatch({
    scenario,
    rules,
    board,
    initialState,
    model: baseModel,
    seed,
    games,
    maxPlies,
    searchOptions,
    ...effectiveSelfPlayOptions,
  });
  const replayEnabled = replayInputs.length > 0 || replayBufferOptions.enabled === true;
  const replayBuffer = replayEnabled
    ? buildAlphaReplayBuffer([...replayInputs, { ...selfPlay, source: "current-self-play" }], replayBufferOptions)
    : null;
  const replaySamples = replayBuffer ? samplesFromAlphaReplayBuffer(replayBuffer) : null;
  const sampleSelectionOptions = {
    maxSamples: Number(replayBufferOptions.trainingSampleLimit || 0),
    balanceBy: replayBufferOptions.trainingSampleBalance || "none",
    priorityBy: replayBufferOptions.trainingSamplePriority || "none",
  };
  const samples = replayBuffer
    ? selectAlphaReplaySamples(replaySamples, sampleSelectionOptions)
    : flattenAlphaSelfPlaySamples([selfPlay]);
  const reanalysis = reanalysisEnabled
    ? runAlphaReanalysisBatch({
      scenario,
      rules,
      board,
      inputs: replayBuffer ? [replayBuffer] : [{ ...selfPlay, source: "current-self-play" }],
      model: reanalysisOptions.model || baseModel,
      maxSamples: Number(reanalysisOptions.maxSamples || samples.length || 1),
      balanceBy: reanalysisOptions.balanceBy || replayBufferOptions.trainingSampleBalance || "none",
      priorityBy: reanalysisOptions.priorityBy || replayBufferOptions.trainingSamplePriority || "policyEntropy",
      seed,
      includeStateSnapshots: reanalysisOptions.includeStateSnapshots !== false,
      searchOptions: reanalysisOptions.searchOptions || searchOptions,
    })
    : null;
  const trainingSamples = reanalysis
    ? combineTrainingSamplesWithReanalysis(samples, flattenAlphaSelfPlaySamples([reanalysis]), reanalysisOptions.trainingMode)
    : samples;
  const model = trainAlphaModelFromSamples(trainingSamples, {
    scenario,
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
    selfPlayOptions: compactSelfPlayOptions(selfPlayOptions),
    reanalysisOptions: compactReanalysisOptions(reanalysisOptions, reanalysisEnabled),
    selfPlay: summarizeAlphaSelfPlayBatch(selfPlay),
    replayBuffer,
    replayBufferSummary: replayBuffer ? summarizeAlphaReplayBuffer(replayBuffer) : null,
    trainingSampleSelection: replayBuffer ? summarizeAlphaReplaySampleSelection(samples, sampleSelectionOptions) : null,
    reanalysis,
    reanalysisSummary: reanalysis ? summarizeAlphaReanalysisBatch(reanalysis, reanalysisOptions) : null,
    training: summarizeAlphaTraining(trainingSamples, model, scenario),
    model,
  };
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
    averagePolicyEntropy: averageDecisionPolicyEntropy(batch?.samples || []),
  };
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
  return {
    schema: "zizi-el-alamein-alpha-training-summary-v1",
    samples: samples.length,
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
    includeStateSnapshots: Boolean(options.includeStateSnapshots),
  };
}

function compactReanalysisOptions(options = {}, enabled = false) {
  return {
    enabled,
    maxSamples: options.maxSamples ?? null,
    balanceBy: options.balanceBy ?? null,
    priorityBy: options.priorityBy ?? null,
    trainingMode: normalizeReanalysisTrainingMode(options.trainingMode),
    includeStateSnapshots: options.includeStateSnapshots !== false,
    searchOptions: options.searchOptions ? compactSearchOptions(options.searchOptions) : null,
  };
}

function trainingOptionsWithBaseModel(options = {}, baseModel = null) {
  return {
    ...options,
    value: {
      baseWeights: baseModel?.value?.weights,
      ...(options.value || {}),
    },
    policy: {
      baseWeights: baseModel?.policy?.weights,
      ...(options.policy || {}),
    },
  };
}

function isReanalysisEnabled(options = {}) {
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

function trainingSampleKey(sample) {
  return [
    sample?.stateHash || "no-state",
    sample?.side || "no-side",
    sample?.phaseId || "no-phase",
    sample?.turn ?? "no-turn",
  ].join("|");
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

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
