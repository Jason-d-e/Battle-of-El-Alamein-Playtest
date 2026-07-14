import {
  buildAlphaEvaluationSuite,
  buildAlphaPairedEvaluationSuite,
  runAlphaModelMatchGame,
} from "./ai-alpha-evaluation.js";
import {
  samplesFromAlphaReplayBuffer,
  selectAlphaReplaySamples,
} from "./ai-alpha-replay-buffer.js";
import { flattenAlphaSelfPlaySamples } from "./ai-alpha-training.js";
import { flattenTrainingEvents } from "./ai-training.js";

export const ALPHA_CHALLENGE_SUITE_SELECTION_SCHEMA = "zizi-el-alamein-alpha-challenge-suite-selection-v1";
export const ALPHA_CONTESTED_SUITE_SELECTION_SCHEMA = "zizi-el-alamein-alpha-contested-suite-selection-v1";

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

export function buildAlphaContestedEvaluationSuite(inputs = [], options = {}) {
  const screened = screenAlphaContestedSamples(inputs, options);
  const selected = screened.samples;
  const seed = Number.isFinite(Number(options.seed)) ? Number(options.seed) : 1942;
  const suite = selected.length
    ? buildAlphaPairedEvaluationSuite({
      seed,
      seeds: selected.map((_, index) => seed + index),
      maxPlies: finiteOrNull(options.maxPlies),
      labels: selected.map((entry, index) => contestedLabel(entry.sample, index)),
      initialStates: selected.map((entry) => entry.sample.initialState),
    })
    : emptyChallengeSuite(seed, "axis");
  return {
    ...suite,
    explicit: true,
    entries: suite.entries.map((entry, index) => {
      const selectedEntry = selected[Math.floor(index / 2)];
      return selectedEntry
        ? enrichContestedEntry(entry, selectedEntry.sample, selectedEntry.calibration)
        : entry;
    }),
    contestedSelection: screened.report,
  };
}

export function screenAlphaContestedSamples(inputs = [], options = {}) {
  const maxPositions = Math.max(0, Math.floor(Number(options.maxPositions ?? options.maxSamples ?? 16)));
  const maxCandidates = Math.max(
    maxPositions,
    Math.floor(Number(options.maxCandidates ?? Math.max(maxPositions * 4, 16))),
  );
  const excludedStateHashes = new Set((options.excludedStateHashes || []).filter(Boolean));
  const uniqueSnapshots = deduplicateSnapshotSamples(alphaSnapshotSamplesFromInputs(inputs))
    .filter((sample) => !sample.stateHash || !excludedStateHashes.has(sample.stateHash));
  const candidates = selectAlphaReplaySamples(uniqueSnapshots, {
    maxSamples: maxCandidates,
    sides: options.sides || options.side,
    balanceBy: options.balanceBy || "sidePhaseAction",
    priorityBy: options.priorityBy || "uncertainty",
  });
  const scenario = options.scenario;
  const rules = options.rules;
  if (!scenario || !rules) throw new Error("Contested Alpha screening requires scenario and rules data");
  const runMatch = options.runMatch || runAlphaModelMatchGame;
  const seed = Number.isFinite(Number(options.seed)) ? Number(options.seed) : 1942;
  const minPlies = Math.max(0, Math.floor(Number(options.minPlies ?? 8)));
  const maxPlies = Math.max(1, Math.floor(Number(options.maxPlies ?? 160)));
  const screened = candidates.map((sample, index) => {
    const positionSeed = seed + index;
    const assignments = ["axis", "allied"].map((candidateSide) => runMatch({
      scenario,
      rules,
      board: options.board || null,
      initialState: sample.initialState,
      candidateModel: options.referenceModel || null,
      baselineModel: options.opponentModel || null,
      candidateSide,
      seed: positionSeed,
      maxPlies,
      guardAdjudication: "none",
      requireRulesWinner: true,
      searchOptions: options.searchOptions || {},
    }));
    const rulesComplete = assignments.every((result) => (
      (result?.winner?.side === "axis" || result?.winner?.side === "allied")
      && !result.guardHit
      && !(result.errors || []).length
    ));
    const winnerSides = assignments.map((result) => result?.winner?.side || null);
    const observedPlies = assignments.map((result) => Math.max(0, Number(result?.plies || 0)));
    const calibration = {
      seed: positionSeed,
      rulesComplete,
      winnerSides,
      winnerChangedWithAssignment: rulesComplete && winnerSides[0] !== winnerSides[1],
      plies: observedPlies,
      minimumPlies: observedPlies.length ? Math.min(...observedPlies) : 0,
      candidateResults: assignments.map((result) => result?.candidateResult || null),
    };
    return {
      sample,
      calibration,
      accepted: calibration.winnerChangedWithAssignment && calibration.minimumPlies >= minPlies,
    };
  });
  const accepted = screened.filter((entry) => entry.accepted).slice(0, maxPositions);
  return {
    samples: accepted,
    report: {
      schema: ALPHA_CONTESTED_SUITE_SELECTION_SCHEMA,
      inputCount: Array.isArray(inputs) ? inputs.length : 0,
      eligibleSamples: uniqueSnapshots.length,
      excludedStateHashes: excludedStateHashes.size,
      screenedSamples: screened.length,
      rulesCompleteSamples: screened.filter((entry) => entry.calibration.rulesComplete).length,
      assignmentSensitiveSamples: screened.filter((entry) => entry.calibration.winnerChangedWithAssignment).length,
      rejectedShortHorizonSamples: screened.filter((entry) => (
        entry.calibration.winnerChangedWithAssignment && entry.calibration.minimumPlies < minPlies
      )).length,
      selectedSamples: accepted.length,
      maxPositions,
      maxCandidates,
      minPlies,
      balanceBy: options.balanceBy || "sidePhaseAction",
      priorityBy: options.priorityBy || "uncertainty",
      selected: accepted.map((entry) => ({
        stateHash: entry.sample.stateHash || null,
        side: entry.sample.side || null,
        phaseId: entry.sample.phaseId || null,
        turn: entry.sample.turn ?? null,
        ...entry.calibration,
      })),
    },
  };
}

export function alphaSnapshotSamplesFromInputs(inputs = []) {
  return (inputs || [])
    .flatMap((input) => samplesFromInput(input))
    .filter((sample) => sample?.initialState && typeof sample.initialState === "object" && !Array.isArray(sample.initialState));
}

function deduplicateSnapshotSamples(samples) {
  const seen = new Set();
  return (samples || []).filter((sample) => {
    const hash = sample?.stateHash;
    if (!hash) return true;
    if (seen.has(hash)) return false;
    seen.add(hash);
    return true;
  });
}

function enrichContestedEntry(entry, sample, calibration) {
  return {
    ...enrichChallengeEntry(entry, sample, {}),
    contestedCalibration: cloneJsonLike(calibration),
  };
}

function contestedLabel(sample, index) {
  return [
    "contested",
    index + 1,
    sample?.side || "side",
    sample?.phaseId || "phase",
  ].join("-");
}

export function alphaRecommendationSamplesFromTrainingLogs(logs = []) {
  const normalizedLogs = (Array.isArray(logs) ? logs : [logs])
    .map((log, index) => (Array.isArray(log)
      ? { source: `input-${index + 1}`, events: log }
      : log));
  return flattenTrainingEvents(normalizedLogs)
    .map(alphaRecommendationSampleFromEvent)
    .filter(Boolean);
}

function samplesFromInput(input) {
  if (!input) return [];
  if (input.schema === "zizi-el-alamein-alpha-replay-buffer-v1") return samplesFromAlphaReplayBuffer(input);
  if (Array.isArray(input?.events) || isAlphaRecommendationEventArray(input)) {
    return alphaRecommendationSamplesFromTrainingLogs([input]);
  }
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
    sourceRecommendationConfidence: finiteOrNull(sample.decision?.recommendationConfidence),
    sourceRecommendationVisitMargin: finiteOrNull(sample.decision?.recommendationVisitMargin),
    sourceRecommendationQMargin: finiteOrNull(sample.decision?.recommendationQMargin),
    sourceRecommendationLabel: typeof sample.decision?.recommendationLabel === "string"
      ? sample.decision.recommendationLabel
      : null,
    sourceSelectionMode: typeof sample.decision?.selectionMode === "string" ? sample.decision.selectionMode : null,
    sourceTemperature: finiteOrNull(sample.decision?.temperature),
    sourceRootNoiseWeight: finiteOrNull(sample.decision?.rootNoiseWeight),
    sourceSearchIterations: finiteOrNull(sample.decision?.searchIterations),
    sourceRootVisits: finiteOrNull(sample.decision?.rootVisits),
    sourceAlphaRecommendation: sample.alphaRecommendation ? {
      ok: Boolean(sample.alphaRecommendation.ok),
      reason: sample.alphaRecommendation.reason || null,
      selectedSource: sample.alphaRecommendation.selectedSource || null,
      illegalCandidateCount: finiteOrNull(sample.alphaRecommendation.illegalCandidateCount),
      legalActionCount: finiteOrNull(sample.alphaRecommendation.legalActionCount),
      runtimeActionMode: sample.alphaRecommendation.runtimeActionMode || null,
      runtimeAnalysisMode: sample.alphaRecommendation.runtimeAnalysisMode || null,
    } : null,
    sourceReplay: sample.replay ? {
      key: sample.replay.key || null,
      source: sample.replay.source || null,
      sampleIndex: sample.replay.sampleIndex ?? null,
      resultIndex: sample.replay.resultIndex ?? null,
      mergedSamples: sample.replay.mergedSamples ?? null,
    } : null,
  };
}

function isAlphaRecommendationEventArray(input) {
  return Array.isArray(input) && input.some((entry) => entry?.type === "ALPHA_ACTION_RECOMMENDED");
}

function alphaRecommendationSampleFromEvent(event) {
  if (!event || event.type !== "ALPHA_ACTION_RECOMMENDED") return null;
  const alpha = alphaRecommendationPayload(event);
  const initialState = event.stateBefore || null;
  if (!initialState || typeof initialState !== "object" || Array.isArray(initialState)) return null;
  const snapshot = alpha.snapshot || {};
  const analysis = snapshot.analysis || {};
  const recommendation = analysis.recommendation || {};
  const legalSelection = alpha.legalSelection || event.legalSelection || {};
  const runtime = snapshot.runtime || {};
  const selectedSource = legalSelection.selectedSource || snapshot.action?.selectedSource || null;
  const confidence = finiteOrNull(snapshot.action?.confidence ?? recommendation.confidence);
  return {
    schema: "zizi-el-alamein-alpha-training-sample-v1",
    stateHash: event.stateHashBefore || snapshot.stateHash || null,
    side: event.side || alpha.context?.side || analysis.side || null,
    turn: event.stateBefore?.turn ?? event.turn ?? alpha.context?.turn ?? analysis.turn ?? null,
    phaseId: event.stateBefore?.phaseId || event.phaseId || alpha.context?.phaseId || analysis.phaseId || null,
    initialState: cloneJsonLike(initialState),
    rootValue: finiteOrNull(analysis.rootValue),
    outcomeSource: "alpha_recommendation_event",
    decision: {
      schema: "zizi-el-alamein-alpha-recommendation-decision-v1",
      selectionMode: recommendationSelectionMode(selectedSource, Boolean(alpha.ok)),
      recommendationConfidence: confidence,
      recommendationVisitMargin: finiteOrNull(recommendation.visitMargin),
      recommendationQMargin: finiteOrNull(recommendation.qMargin),
      recommendationLabel: recommendation.label || null,
      policyEntropy: finiteOrNull(recommendation.entropy),
      selectedVisitShare: finiteOrNull(recommendation.bestVisitShare),
      selectedPolicyRank: finiteOrNull(legalSelection.selectedCandidateIndex) === null
        ? null
        : finiteOrNull(legalSelection.selectedCandidateIndex) + 1,
      selectedIsBest: selectedSource === "direct" || finiteOrNull(legalSelection.selectedCandidateIndex) === 0,
      searchIterations: finiteOrNull(analysis.search?.iterations),
      rootVisits: finiteOrNull(analysis.search?.rootVisits),
      runtimeActionMode: runtime.actionMode || null,
      runtimeAnalysisMode: runtime.analysisMode || null,
      illegalCandidateCount: finiteOrNull(legalSelection.illegalCandidateCount),
      legalActionCount: finiteOrNull(legalSelection.legalActionCount),
    },
    policy: alphaRecommendationPolicyEntries(alpha, recommendation, confidence),
    alphaRecommendation: {
      ok: Boolean(alpha.ok),
      reason: alpha.reason || event.recommendationReason || null,
      selectedSource,
      illegalCandidateCount: finiteOrNull(legalSelection.illegalCandidateCount),
      legalActionCount: finiteOrNull(legalSelection.legalActionCount),
      runtimeActionMode: runtime.actionMode || null,
      runtimeAnalysisMode: runtime.analysisMode || null,
    },
    replay: {
      key: event.id || `${event.__source || "training-events"}#${event.__eventIndex ?? 0}`,
      source: event.__source || null,
      sampleIndex: event.__eventIndex ?? null,
      eventId: event.id || null,
      eventType: event.type,
      stateHashBefore: event.stateHashBefore || null,
      stateHashAfter: event.stateHashAfter || null,
    },
    __source: event.__source || null,
    __sampleIndex: event.__eventIndex ?? null,
  };
}

function alphaRecommendationPayload(event) {
  if (event?.alpha && typeof event.alpha === "object" && !Array.isArray(event.alpha)) return event.alpha;
  return event && typeof event === "object" && !Array.isArray(event) ? event : {};
}

function recommendationSelectionMode(selectedSource, ok) {
  if (!ok) return "rejected";
  if (!selectedSource) return "unknown";
  return selectedSource === "direct" ? "runtime_direct" : `runtime_${selectedSource}`;
}

function alphaRecommendationPolicyEntries(alpha, recommendation, confidence) {
  const actions = [
    alpha?.action,
    recommendation?.action,
    alpha?.legalSelection?.action,
  ].filter(Boolean);
  const unique = [];
  const seen = new Set();
  for (const action of actions) {
    const key = JSON.stringify(action);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(action);
  }
  return unique.map((action, index) => ({
    action: cloneJsonLike(action),
    visitShare: index === 0 && confidence !== null ? clamp01(confidence) : 0,
  }));
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
  if (mode === "uncertainty") {
    return recommendationUncertainty(sample);
  }
  if (mode === "runtimeRisk") {
    return runtimeRecommendationRisk(sample);
  }
  return 0;
}

function recommendationUncertainty(sample) {
  const confidence = Number(sample?.decision?.recommendationConfidence);
  if (Number.isFinite(confidence)) return round(1 - clamp01(confidence), 6);
  const visitMargin = Number(sample?.decision?.recommendationVisitMargin);
  const marginUncertainty = Number.isFinite(visitMargin) ? 1 - clamp01(Math.abs(visitMargin)) : null;
  const entropy = Number(sample?.decision?.policyEntropy);
  const entropyScore = Number.isFinite(entropy) ? clamp01(entropy) : policyEntropy(sample?.policy || []);
  if (marginUncertainty !== null) return round((marginUncertainty + entropyScore) / 2, 6);
  return round(entropyScore, 6);
}

function runtimeRecommendationRisk(sample) {
  const recommendation = sample?.alphaRecommendation || {};
  const decision = sample?.decision || {};
  if (recommendation.ok === false || decision.selectionMode === "rejected") return 1;
  let risk = 0;
  const selectedSource = recommendation.selectedSource || decision.selectedSource || null;
  if (selectedSource && selectedSource !== "direct") risk += 0.35;
  const illegalCandidateCount = finiteOrDefault(recommendation.illegalCandidateCount ?? decision.illegalCandidateCount, 0);
  risk += Math.min(0.3, Math.max(0, illegalCandidateCount) * 0.1);
  const confidence = Number(decision.recommendationConfidence);
  if (Number.isFinite(confidence)) risk += 0.25 * (1 - clamp01(confidence));
  if (usesDirectRuntime(recommendation) || usesDirectRuntime(decision)) risk += 0.1;
  return round(clamp01(risk), 6);
}

function usesDirectRuntime(value) {
  return value?.runtimeActionMode === "direct" || value?.runtimeAnalysisMode === "direct";
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
  return value === "surprise" || value === "policyEntropy" || value === "uncertainty" || value === "runtimeRisk"
    ? value
    : "none";
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function normalizeSide(side, fallback = "axis") {
  if (side === "axis" || side === "allied") return side;
  return fallback === "axis" || fallback === "allied" ? fallback : "axis";
}

function cloneJsonLike(value) {
  if (value === null || value === undefined) return value;
  return JSON.parse(JSON.stringify(value));
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
