import { buildAlphaEvaluationSuite } from "./ai-alpha-evaluation.js";
import {
  samplesFromAlphaReplayBuffer,
  selectAlphaReplaySamples,
} from "./ai-alpha-replay-buffer.js";
import { flattenAlphaSelfPlaySamples } from "./ai-alpha-training.js";

export const ALPHA_CHALLENGE_SUITE_SELECTION_SCHEMA = "zizi-el-alamein-alpha-challenge-suite-selection-v1";

export function buildAlphaChallengeEvaluationSuite(inputs = [], options = {}) {
  const snapshots = alphaSnapshotSamplesFromInputs(inputs);
  const maxPositions = Math.max(0, Math.floor(Number(options.maxPositions ?? options.maxSamples ?? 16)));
  const selected = maxPositions > 0
    ? selectAlphaReplaySamples(snapshots, {
      maxSamples: maxPositions,
      balanceBy: options.balanceBy || "none",
      priorityBy: options.priorityBy || "surprise",
    })
    : [];
  const seed = Number.isFinite(Number(options.seed)) ? Number(options.seed) : 1942;
  const suite = selected.length
    ? buildAlphaEvaluationSuite({
      seed,
      games: selected.length,
      seeds: selected.map((_, index) => seed + index),
      candidateSide: normalizeSide(options.candidateSide, selected[0]?.side || "axis"),
      candidateSides: selected.map((sample) => normalizeSide(sample.side, options.candidateSide || "axis")),
      labels: selected.map((sample, index) => challengeLabel(sample, index)),
      maxPlies: finiteOrNull(options.maxPlies),
      initialStates: selected.map((sample) => sample.initialState),
    })
    : emptyChallengeSuite(seed, normalizeSide(options.candidateSide, "axis"));

  return {
    ...suite,
    explicit: true,
    entries: suite.entries.map((entry, index) => enrichChallengeEntry(entry, selected[index], options)),
    challengeSelection: {
      schema: ALPHA_CHALLENGE_SUITE_SELECTION_SCHEMA,
      inputCount: Array.isArray(inputs) ? inputs.length : 0,
      eligibleSamples: snapshots.length,
      selectedSamples: selected.length,
      maxPositions,
      balanceBy: options.balanceBy || "none",
      priorityBy: normalizePriorityBy(options.priorityBy || "surprise"),
      sources: sourceList(selected),
    },
  };
}

export function alphaSnapshotSamplesFromInputs(inputs = []) {
  return (inputs || [])
    .flatMap((input) => samplesFromInput(input))
    .filter((sample) => sample?.initialState && typeof sample.initialState === "object" && !Array.isArray(sample.initialState));
}

function samplesFromInput(input) {
  if (!input) return [];
  if (input.schema === "zizi-el-alamein-alpha-replay-buffer-v1") return samplesFromAlphaReplayBuffer(input);
  return flattenAlphaSelfPlaySamples([input]);
}

function enrichChallengeEntry(entry, sample, options) {
  if (!sample) return entry;
  return {
    ...entry,
    sourceStateHash: sample.stateHash || null,
    sourceSide: sample.side || null,
    sourcePhaseId: sample.phaseId || null,
    sourceTurn: sample.turn ?? null,
    sourceOutcome: finiteOrNull(sample.outcome),
    sourceRootValue: finiteOrNull(sample.rootValue),
    sourceOutcomeSource: typeof sample.outcomeSource === "string" ? sample.outcomeSource : null,
    sourceOutcomeWeight: finiteOrNull(sample.outcomeWeight),
    sourcePriority: samplePriority(sample, options.priorityBy || "surprise"),
    sourcePolicyEntropy: finiteOrNull(sample.decision?.policyEntropy),
    sourceSelectionMode: typeof sample.decision?.selectionMode === "string" ? sample.decision.selectionMode : null,
    sourceTemperature: finiteOrNull(sample.decision?.temperature),
    sourceRootNoiseWeight: finiteOrNull(sample.decision?.rootNoiseWeight),
    sourceSearchIterations: finiteOrNull(sample.decision?.searchIterations),
    sourceRootVisits: finiteOrNull(sample.decision?.rootVisits),
    sourceReplay: sample.replay ? {
      key: sample.replay.key || null,
      source: sample.replay.source || null,
      sampleIndex: sample.replay.sampleIndex ?? null,
      resultIndex: sample.replay.resultIndex ?? null,
      mergedSamples: sample.replay.mergedSamples ?? null,
    } : null,
  };
}

function emptyChallengeSuite(seed, candidateSide) {
  return {
    schema: "zizi-el-alamein-alpha-evaluation-suite-v1",
    seed,
    games: 0,
    candidateSide,
    alternateSides: false,
    explicit: true,
    entries: [],
  };
}

function challengeLabel(sample, index) {
  const parts = [
    "challenge",
    String(index + 1).padStart(2, "0"),
    sample.side || "side",
    sample.phaseId || "phase",
    shortHash(sample.stateHash),
  ];
  return parts
    .filter(Boolean)
    .map((part) => String(part).replace(/[^a-zA-Z0-9_-]+/g, "-"))
    .join("-");
}

function samplePriority(sample, priorityBy = "surprise") {
  const mode = normalizePriorityBy(priorityBy);
  if (mode === "surprise") {
    const outcome = Number(sample?.outcome);
    const rootValue = Number(sample?.rootValue);
    if (!Number.isFinite(outcome) || !Number.isFinite(rootValue)) return 0;
    return round(Math.abs(outcome - rootValue) * Math.max(0, finiteOrDefault(sample.outcomeWeight, 1)), 6);
  }
  if (mode === "policyEntropy") {
    const recorded = Number(sample?.decision?.policyEntropy);
    return Number.isFinite(recorded) ? round(recorded, 6) : policyEntropy(sample?.policy || []);
  }
  return 0;
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

function sourceList(samples) {
  return [...new Set((samples || []).map((sample) => sample.replay?.source || sample.__source).filter(Boolean))].sort();
}

function normalizePriorityBy(value) {
  return value === "surprise" || value === "policyEntropy" ? value : "none";
}

function normalizeSide(side, fallback = "axis") {
  if (side === "axis" || side === "allied") return side;
  return fallback === "axis" || fallback === "allied" ? fallback : "axis";
}

function shortHash(value) {
  return value ? String(value).slice(0, 10) : "";
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function finiteOrDefault(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}
