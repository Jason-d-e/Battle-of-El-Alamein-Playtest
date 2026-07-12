import {
  activeSide,
  createBoard,
  createEnvironment,
  stateHash,
} from "../core/index.js";
import {
  analyzePosition,
  makeSearchTrainingSample,
} from "./ai-alpha-search.js";
import {
  samplesFromAlphaReplayBuffer,
  selectAlphaReplaySamples,
} from "./ai-alpha-replay-buffer.js";
import { flattenAlphaSelfPlaySamples } from "./ai-alpha-training.js";

export const ALPHA_REANALYSIS_BATCH_SCHEMA = "zizi-el-alamein-alpha-reanalysis-batch-v1";
export const ALPHA_REANALYSIS_SAMPLE_METADATA_SCHEMA = "zizi-el-alamein-alpha-reanalysis-sample-v1";

export function alphaReanalysisSamplesFromInputs(inputs = []) {
  return (inputs || [])
    .flatMap((input) => (
      input?.schema === "zizi-el-alamein-alpha-replay-buffer-v1"
        ? samplesFromAlphaReplayBuffer(input)
        : flattenAlphaSelfPlaySamples([input])
    ))
    .filter((sample) => sample?.schema === "zizi-el-alamein-alpha-training-sample-v1")
    .filter((sample) => sample.initialState && typeof sample.initialState === "object" && !Array.isArray(sample.initialState));
}

export function runAlphaReanalysisBatch(options = {}) {
  const scenario = options.scenario;
  const rules = options.rules;
  if (!scenario || !rules) throw new Error("Alpha reanalysis requires scenario and rules data");

  const board = options.board || createBoard(scenario);
  const sourceSamples = alphaReanalysisSamplesFromInputs(options.inputs || []);
  const maxSamples = Math.max(0, Math.floor(Number(options.maxSamples ?? 16)));
  const selectedSamples = selectAlphaReplaySamples(sourceSamples, {
    maxSamples,
    balanceBy: options.balanceBy || "none",
    priorityBy: options.priorityBy || "policyEntropy",
  });
  const samples = [];
  const errors = [];

  for (const [index, sourceSample] of selectedSamples.entries()) {
    try {
      const sample = reanalyzeSample(sourceSample, {
        scenario,
        rules,
        board,
        model: options.model || null,
        searchOptions: options.searchOptions || {},
        includeStateSnapshots: options.includeStateSnapshots !== false,
      });
      samples.push(sample);
    } catch (error) {
      errors.push({
        index,
        stateHash: sourceSample?.stateHash || null,
        side: sourceSample?.side || null,
        phaseId: sourceSample?.phaseId || null,
        reason: "reanalysis_failed",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return {
    schema: ALPHA_REANALYSIS_BATCH_SCHEMA,
    generatedAt: typeof options.generatedAt === "string" ? options.generatedAt : new Date().toISOString(),
    seed: finiteOrDefault(options.seed, 1942),
    inputCount: Array.isArray(options.inputs) ? options.inputs.length : 0,
    eligibleSamples: sourceSamples.length,
    selectedSamples: selectedSamples.length,
    sampleCount: samples.length,
    maxSamples,
    balanceBy: options.balanceBy || "none",
    priorityBy: options.priorityBy || "policyEntropy",
    includeStateSnapshots: options.includeStateSnapshots !== false,
    search: summarizeSearchOptions(options.searchOptions || {}),
    errors,
    samples,
  };
}

function reanalyzeSample(sourceSample, options) {
  const environment = createEnvironment({
    scenario: options.scenario,
    rules: options.rules,
    board: options.board,
    state: cloneJsonLike(sourceSample.initialState),
  });
  const side = sourceSample.side || activeSide(environment);
  const analysis = analyzePosition(environment, {
    ...options.searchOptions,
    side,
    model: options.model,
    includeStateSnapshot: options.includeStateSnapshots,
  });
  const sample = makeSearchTrainingSample(analysis, null);
  const environmentHash = stateHash(environment);
  const outcome = finiteOrNull(sourceSample.outcome);
  const outcomeWeight = finiteOrDefault(sourceSample.outcomeWeight, 1);
  return {
    ...sample,
    outcome,
    outcomeSource: typeof sourceSample.outcomeSource === "string" ? sourceSample.outcomeSource : "reanalysis_unlabeled",
    outcomeWeight,
    decision: makeReanalysisDecision(analysis, options.searchOptions),
    reanalysis: {
      schema: ALPHA_REANALYSIS_SAMPLE_METADATA_SCHEMA,
      sourceStateHash: sourceSample.stateHash || null,
      reanalyzedStateHash: analysis.stateHash || environmentHash,
      stateHashMatches: !sourceSample.stateHash || sourceSample.stateHash === environmentHash,
      sourceSide: sourceSample.side || null,
      sourceTurn: sourceSample.turn ?? null,
      sourcePhaseId: sourceSample.phaseId || null,
      sourceRootValue: finiteOrNull(sourceSample.rootValue),
      sourceOutcome: outcome,
      sourceOutcomeSource: typeof sourceSample.outcomeSource === "string" ? sourceSample.outcomeSource : null,
      sourceOutcomeWeight: outcomeWeight,
      sourceReplay: sourceSample.replay && typeof sourceSample.replay === "object"
        ? cloneJsonLike(sourceSample.replay)
        : null,
    },
  };
}

function makeReanalysisDecision(analysis, searchOptions = {}) {
  return {
    schema: "zizi-el-alamein-alpha-self-play-decision-v1",
    selectedAction: analysis.bestAction || null,
    selectionMode: "reanalyzed",
    temperature: 0,
    rootNoiseWeight: finiteOrDefault(searchOptions.rootNoiseWeight, 0),
    policyEntropy: policyEntropy(analysis.policy || []),
    policySize: Array.isArray(analysis.policy) ? analysis.policy.length : 0,
    searchIterations: finiteOrDefault(analysis.search?.iterations, 0),
    rootVisits: finiteOrDefault(analysis.search?.rootVisits, 0),
  };
}

function summarizeSearchOptions(searchOptions = {}) {
  return {
    iterations: finiteOrNull(searchOptions.iterations),
    maxDepth: finiteOrNull(searchOptions.maxDepth),
    actionLimit: finiteOrNull(searchOptions.actionLimit),
    preApplyLimit: finiteOrNull(searchOptions.preApplyLimit),
    policyWeight: finiteOrNull(searchOptions.policyWeight),
    rootNoiseWeight: finiteOrDefault(searchOptions.rootNoiseWeight, 0),
  };
}

function policyEntropy(policy) {
  const shares = (policy || [])
    .map((entry) => Math.max(0, Number(entry.visitShare || 0)))
    .filter((value) => value > 0);
  const total = shares.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return 0;
  const entropy = shares.reduce((sum, value) => {
    const p = value / total;
    return sum - p * Math.log2(p);
  }, 0);
  return round(entropy / Math.log2(Math.max(2, shares.length)), 6);
}

function cloneJsonLike(value) {
  return value ? JSON.parse(JSON.stringify(value)) : value;
}

function finiteOrNull(value) {
  if (value === null || value === undefined) return null;
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function finiteOrDefault(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}
