import {
  activeSide,
  createBoard,
  createEnvironment,
  makeInitialEnvironmentState,
  unitById,
} from "../core/index.js";
import {
  DEFAULT_PHASE_SEARCH_WEIGHTS,
  mergePhaseSearchWeights,
  movementActionFeatureVector,
  scoreFeatureVector,
} from "./ai-phase-search.js";
import { flattenTrainingEvents, flattenTrainingLogs } from "./ai-training.js";

const DEFAULT_ITERATIONS = 4;
const DEFAULT_MAX_NEGATIVES = 12;
const DEFAULT_STEP_SIZES = [256, 128, 64, 32, 16, 8, 4, 2, 1];
const TRAINABLE_WEIGHT_KEYS = Object.freeze(Object.keys(DEFAULT_PHASE_SEARCH_WEIGHTS).filter((key) => (
  !key.endsWith("Factor")
)));
const WEIGHT_CONSTRAINTS = Object.freeze({
  axisProgress: { min: 88 },
  axisObjective: { min: 0 },
  axisObjectiveSupport: { min: 0 },
  axisObjectiveCounterThreat: { max: 0 },
  axisObjectiveUnsupportedPenalty: { max: 0 },
  axisAdjacent: { min: 0 },
  axisNear: { min: 0 },
  axisStateObjectiveHeld: { min: 0 },
  axisStateObjectiveDistance: { max: 0 },
  alliedForwardBand: { min: 160 },
  alliedForwardBandPenalty: { max: 0 },
  alliedObjectiveHugPenalty: { max: -80 },
  alliedContact: { min: 80 },
  alliedContactPenalty: { max: 0 },
  alliedLine: { min: 45 },
  alliedDrift: { max: 0 },
  alliedCombat: { min: 0 },
  alliedStateAxisDistance: { min: 0 },
  alliedStateWallLink: { min: 0 },
});

export function extractPreferenceSamplesFromLogs(logs, context = {}) {
  return extractPreferenceSamples({
    entries: flattenTrainingLogs(logs),
    events: flattenTrainingEvents(logs),
    ...context,
  });
}

export function extractPreferenceSamples({ entries = [], events = [], scenario, rules, board = null } = {}) {
  const resolvedBoard = board || (scenario ? createBoard(scenario) : null);
  const context = { scenario, rules, board: resolvedBoard };
  return [
    ...entries.flatMap((entry) => movementEntryPreferenceSample(entry, context) || []),
    ...events.flatMap((event) => movementEventPreferenceSample(event, context) || []),
  ];
}

export function trainPreferenceWeights(samples, options = {}) {
  const normalized = normalizeSamples(samples).filter((sample) => sample.candidates.length >= 2);
  const baseWeights = mergePhaseSearchWeights(options.baseWeights || null);
  let weights = { ...baseWeights };
  const keys = (options.keys || TRAINABLE_WEIGHT_KEYS).filter((key) => normalized.some((sample) => (
    sample.candidates.some((candidate) => Number(candidate.features?.[key] || 0) !== 0)
  )));
  let bestLoss = preferenceLoss(normalized, weights, options);

  for (let iteration = 0; iteration < Number(options.iterations || DEFAULT_ITERATIONS); iteration += 1) {
    for (const step of options.stepSizes || DEFAULT_STEP_SIZES) {
      let improved = true;
      while (improved) {
        improved = false;
        for (const key of keys) {
          const current = Number(weights[key] || 0);
          for (const direction of [1, -1]) {
            const candidateWeights = {
              ...weights,
              [key]: clampWeight(current + direction * step, key),
            };
            const candidateLoss = preferenceLoss(normalized, candidateWeights, options);
            if (candidateLoss + 1e-9 < bestLoss) {
              weights = candidateWeights;
              bestLoss = candidateLoss;
              improved = true;
            }
          }
        }
      }
    }
  }

  return {
    weights,
    metrics: evaluatePreferenceWeights(normalized, weights, options),
    baselineMetrics: evaluatePreferenceWeights(normalized, baseWeights, options),
    trainableKeys: keys,
    sampleCount: normalized.length,
  };
}

export function evaluatePreferenceWeights(samples, weights = null, options = {}) {
  const normalized = normalizeSamples(samples);
  if (!normalized.length) {
    return {
      samples: 0,
      pairCount: 0,
      loss: null,
      top1Rate: null,
      top3Rate: null,
      meanReciprocalRank: null,
    };
  }

  let top1 = 0;
  let top3 = 0;
  let reciprocalRank = 0;
  let pairCount = 0;
  let pairWins = 0;
  for (const sample of normalized) {
    const ranked = rankSampleCandidates(sample, weights);
    const rank = ranked.findIndex((candidate) => candidate.key === sample.positiveKey);
    if (rank === 0) top1 += 1;
    if (rank >= 0 && rank < 3) top3 += 1;
    if (rank >= 0) reciprocalRank += 1 / (rank + 1);
    const positive = ranked.find((candidate) => candidate.key === sample.positiveKey);
    if (!positive) continue;
    for (const candidate of ranked) {
      if (candidate.key === sample.positiveKey) continue;
      pairCount += 1;
      if (positive.score > candidate.score) pairWins += 1;
    }
  }

  return {
    samples: normalized.length,
    pairCount,
    pairAccuracy: pairCount ? round(pairWins / pairCount, 4) : null,
    loss: round(preferenceLoss(normalized, weights, options), 4),
    top1Rate: round(top1 / normalized.length, 4),
    top3Rate: round(top3 / normalized.length, 4),
    meanReciprocalRank: round(reciprocalRank / normalized.length, 4),
  };
}

export function rankSampleCandidates(sample, weights = null) {
  return (sample.candidates || [])
    .map((candidate) => ({
      ...candidate,
      score: scoreFeatureVector(candidate.features, weights),
    }))
    .sort((a, b) => b.score - a.score || String(a.key).localeCompare(String(b.key)));
}

export function makeExpertWeightsArtifact({ weights, metrics, baselineMetrics, sampleCount, trainableKeys, sources = [] } = {}) {
  return {
    schema: "zizi-el-alamein-ai-weights-v1",
    profile: "expert",
    generatedAt: new Date().toISOString(),
    method: "coordinate-descent-logistic-ranking",
    sources,
    sampleCount: sampleCount || 0,
    trainableKeys: trainableKeys || [],
    metrics: metrics || null,
    baselineMetrics: baselineMetrics || null,
    phaseSearch: mergePhaseSearchWeights(weights || null),
  };
}

function movementEntryPreferenceSample(entry, context) {
  if (entry?.action !== "move" || !entry.humanChoice?.hexId) return null;
  const environment = environmentFromEntry(entry, context);
  const unit = unitById(environment?.state?.units || [], entry.unit?.id);
  if (!environment || !unit) return null;

  const candidates = uniqueChoices([
    ...(entry.aiTopChoices || []),
    entry.aiChoice,
    entry.humanChoice,
  ]).map((choice) => movementChoiceCandidate(environment, unit, {
    type: "MOVE_UNIT",
    unitId: unit.id,
    fromHexId: entry.fromHexId || unit.hexId,
    toHexId: choice.hexId,
    route: choice.route || null,
  }, {
    sourceScore: choice.score,
  })).filter(Boolean);

  return buildSample({
    id: entry.id || `entry-${entry.__entryIndex ?? "unknown"}`,
    source: entry.__source || null,
    type: "move",
    side: entry.side || unit.side,
    positiveKey: moveKey(unit.id, entry.humanChoice.hexId),
    candidates,
  });
}

function movementEventPreferenceSample(event, context) {
  if (event?.type !== "UNIT_MOVED" || !isHumanControlledEvent(event) || !event.stateBefore || !event.toHexId) return null;
  const environment = createEnvironment({
    scenario: context.scenario,
    rules: context.rules,
    board: context.board,
    state: event.stateBefore,
  });
  const unit = unitById(environment.state.units, event.unitId);
  if (!unit) return null;

  const candidates = (event.legalActionsBefore || [])
    .filter((action) => action.type === "MOVE_UNIT" && action.unitId === event.unitId)
    .map((action) => movementChoiceCandidate(environment, unit, action))
    .filter(Boolean);

  if (!candidates.some((candidate) => candidate.key === moveKey(event.unitId, event.toHexId))) {
    const actual = movementChoiceCandidate(environment, unit, {
      type: "MOVE_UNIT",
      unitId: event.unitId,
      fromHexId: event.fromHexId || unit.hexId,
      toHexId: event.toHexId,
      route: event.route || null,
    });
    if (actual) candidates.push(actual);
  }

  return buildSample({
    id: event.id || `event-${event.__eventIndex ?? "unknown"}`,
    source: event.__source || null,
    type: "move",
    side: event.side || activeSide(environment),
    positiveKey: moveKey(event.unitId, event.toHexId),
    candidates,
  });
}

function movementChoiceCandidate(environment, unit, action, extras = {}) {
  if (!action?.toHexId) return null;
  const features = movementActionFeatureVector(environment, action, {
    side: unit.side,
    unitBefore: unit,
    unitAfter: { ...unit, hexId: action.toHexId },
    distanceCache: new Map(),
  });
  return {
    key: moveKey(unit.id, action.toHexId),
    action: {
      type: "MOVE_UNIT",
      unitId: unit.id,
      fromHexId: action.fromHexId || unit.hexId,
      toHexId: action.toHexId,
    },
    features,
    sourceScore: Number.isFinite(Number(extras.sourceScore)) ? Number(extras.sourceScore) : null,
  };
}

function environmentFromEntry(entry, context) {
  if (!context.scenario || !context.rules || !context.board) return null;
  if (entry.stateBefore || entry.stateSnapshot) {
    return createEnvironment({
      scenario: context.scenario,
      rules: context.rules,
      board: context.board,
      state: entry.stateBefore || entry.stateSnapshot,
    });
  }
  const state = makeInitialEnvironmentState(context.scenario);
  state.turn = Number(entry.turn || 1);
  state.phaseIndex = Number(entry.phaseIndex || 0);
  const unitId = entry.unit?.id;
  if (!unitId) return null;
  const unit = unitById(state.units, unitId) || entry.unit;
  state.units = state.units.filter((candidate) => candidate.id !== unitId);
  state.units.push({
    id: unitId,
    side: entry.side || entry.unit?.side || unit.side || "axis",
    hexId: entry.fromHexId || entry.unit?.hexId || unit.hexId,
    combat: Number(entry.unit?.combat ?? unit.combat ?? 0),
    movement: Number(entry.unit?.movement ?? unit.movement ?? 0),
    disrupted: Boolean(entry.unit?.disrupted),
    eliminated: false,
  });
  return createEnvironment({
    scenario: context.scenario,
    rules: context.rules,
    board: context.board,
    state,
  });
}

function buildSample(sample) {
  const candidates = uniqueCandidates(sample.candidates || []);
  if (candidates.length < 2 || !candidates.some((candidate) => candidate.key === sample.positiveKey)) return null;
  return {
    ...sample,
    candidates,
  };
}

function normalizeSamples(samples) {
  return (samples || [])
    .filter(Boolean)
    .map((sample) => ({
      ...sample,
      candidates: trimCandidates(uniqueCandidates(sample.candidates || []), sample.positiveKey),
    }))
    .filter((sample) => sample.positiveKey && sample.candidates.some((candidate) => candidate.key === sample.positiveKey));
}

function trimCandidates(candidates, positiveKey) {
  const positive = candidates.find((candidate) => candidate.key === positiveKey);
  const negatives = candidates.filter((candidate) => candidate.key !== positiveKey).slice(0, DEFAULT_MAX_NEGATIVES);
  return positive ? [positive, ...negatives] : negatives;
}

function preferenceLoss(samples, weights, options = {}) {
  if (!samples.length) return 0;
  let total = 0;
  let pairs = 0;
  const margin = Number(options.margin ?? 1);
  for (const sample of samples) {
    const positive = sample.candidates.find((candidate) => candidate.key === sample.positiveKey);
    if (!positive) continue;
    const positiveScore = scoreFeatureVector(positive.features, weights);
    for (const candidate of sample.candidates) {
      if (candidate.key === sample.positiveKey) continue;
      const delta = (positiveScore - scoreFeatureVector(candidate.features, weights)) / 100;
      total += logisticLoss(delta - margin);
      pairs += 1;
    }
  }
  return pairs ? total / pairs : 0;
}

function logisticLoss(value) {
  if (value > 40) return 0;
  if (value < -40) return -value;
  return Math.log1p(Math.exp(-value));
}

function isHumanControlledEvent(event) {
  if (event.mode === "hotseat") return true;
  if (event.mode === "axis-vs-ai") return event.side === "axis";
  if (event.mode === "allied-vs-ai") return event.side === "allied";
  return false;
}

function uniqueChoices(choices) {
  const seen = new Set();
  const unique = [];
  for (const choice of choices) {
    if (!choice) continue;
    const key = choice.hexId || choice.defenderId || choice.id || JSON.stringify(choice);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(choice);
  }
  return unique;
}

function uniqueCandidates(candidates) {
  const seen = new Set();
  const unique = [];
  for (const candidate of candidates) {
    if (!candidate?.key || seen.has(candidate.key)) continue;
    seen.add(candidate.key);
    unique.push(candidate);
  }
  return unique;
}

function moveKey(unitId, toHexId) {
  return `move:${unitId}:${toHexId}`;
}

function clampWeight(value, key) {
  const defaultValue = Number(DEFAULT_PHASE_SEARCH_WEIGHTS[key] || 0);
  const limit = Math.max(500, Math.abs(defaultValue) * 4);
  const constraint = WEIGHT_CONSTRAINTS[key] || {};
  const min = Number.isFinite(constraint.min) ? Math.max(-limit, constraint.min) : -limit;
  const max = Number.isFinite(constraint.max) ? Math.min(limit, constraint.max) : limit;
  return Math.max(min, Math.min(max, value));
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
