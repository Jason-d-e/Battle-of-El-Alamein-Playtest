import {
  alphaTrainingValueTarget,
  alphaTrainingFeatureContract,
  alphaTrainingSampleFeatureContract,
  alphaTrainingSampleFeatureContractFingerprint,
  flattenAlphaSelfPlaySamples,
} from "./ai-alpha-training.js";
import {
  buildTrajectoryLineageComponents,
  normalizeTrajectoryIds,
  unionTrajectoryIds,
} from "../../../shared/wargame-alpha/trajectory-lineage.js";

export const ALPHA_REPLAY_BUFFER_SCHEMA = "zizi-el-alamein-alpha-replay-buffer-v1";
export const ALPHA_REPLAY_BUFFER_QUALITY_SCHEMA = "zizi-el-alamein-alpha-replay-quality-v1";

const DEFAULT_MAX_SAMPLES = 4096;
const REPLAY_DECISION_AGGREGATE_KEYS = Object.freeze([
  "selectedVisitShare",
  "selectedVisits",
  "selectedQ",
  "selectedPrior",
  "selectedPolicyRank",
  "temperature",
  "rootNoiseWeight",
  "policyEntropy",
  "recommendationConfidence",
  "recommendationVisitMargin",
  "recommendationQMargin",
  "illegalCandidateCount",
  "legalActionCount",
  "policySize",
  "searchIterations",
  "rootVisits",
]);

export function buildAlphaReplayBuffer(inputs = [], options = {}) {
  const maxSamples = Math.max(1, Math.floor(Number(options.maxSamples || DEFAULT_MAX_SAMPLES)));
  const generatedAt = typeof options.generatedAt === "string" ? options.generatedAt : new Date().toISOString();
  const duplicateMode = normalizeDuplicateMode(options.duplicateMode);
  const existing = [];
  const logs = [];
  for (const input of inputs || []) {
    if (input?.schema === ALPHA_REPLAY_BUFFER_SCHEMA) existing.push(input);
    else logs.push(input);
  }
  const samples = [
    ...existing.flatMap((buffer) => samplesFromAlphaReplayBuffer(buffer)),
    ...flattenAlphaSelfPlaySamples(logs),
  ];
  const valueNormalizedSamples = samples.map(normalizeReplayValueSample);
  const deduped = duplicateMode === "merge"
    ? mergeDuplicateSamples(valueNormalizedSamples)
    : latestDuplicateSamples(valueNormalizedSamples);
  const lineagePropagated = propagateReplayTrajectoryComponents(deduped);
  const windowed = lineagePropagated.slice(-maxSamples).map((sample, index) => cleanReplaySample(sample, index));
  return {
    schema: ALPHA_REPLAY_BUFFER_SCHEMA,
    generatedAt,
    maxSamples,
    duplicateMode,
    sampleCount: windowed.length,
    sourceCount: sourceList(windowed).length,
    sources: sourceList(windowed),
    quality: analyzeAlphaReplayBufferQuality(windowed, {
      minSamples: 0,
      minSides: 0,
      requireFeatureContract: options.requireFeatureContract,
    }),
    samples: windowed,
  };
}

function propagateReplayTrajectoryComponents(samples) {
  const lineaged = [];
  const sourceIndexes = [];
  for (const [index, sample] of (samples || []).entries()) {
    const trajectoryIds = normalizeTrajectoryIds(sample?.trajectoryIds);
    if (!trajectoryIds.length) continue;
    lineaged.push({ ...sample, trajectoryIds });
    sourceIndexes.push(index);
  }
  if (!lineaged.length) return samples;
  const propagated = new Map();
  for (const component of buildTrajectoryLineageComponents(lineaged)) {
    for (const componentIndex of component.sampleIndexes) {
      propagated.set(sourceIndexes[componentIndex], component.trajectoryIds);
    }
  }
  return samples.map((sample, index) => {
    const trajectoryIds = propagated.get(index);
    return trajectoryIds ? { ...sample, trajectoryIds: [...trajectoryIds] } : sample;
  });
}

export function normalizeAlphaReplayBuffer(value, options = {}) {
  if (!value || typeof value !== "object" || value.schema !== ALPHA_REPLAY_BUFFER_SCHEMA) return null;
  return buildAlphaReplayBuffer([{
    schema: ALPHA_REPLAY_BUFFER_SCHEMA,
    samples: Array.isArray(value.samples) ? value.samples : [],
  }], {
    maxSamples: options.maxSamples || value.maxSamples || DEFAULT_MAX_SAMPLES,
    generatedAt: typeof value.generatedAt === "string" ? value.generatedAt : undefined,
    duplicateMode: options.duplicateMode || value.duplicateMode,
  });
}

export function samplesFromAlphaReplayBuffer(buffer) {
  if (!buffer || typeof buffer !== "object") return [];
  const samples = Array.isArray(buffer.samples) ? buffer.samples : [];
  return samples
    .filter((sample) => sample?.schema === "zizi-el-alamein-alpha-training-sample-v1")
    .map((sample, index) => ({
      ...sample,
      __source: sample.replay?.source || sample.__source || buffer.source || "replay-buffer",
      __sources: sample.replay?.sources || sample.__sources || null,
      __sampleIndex: sample.replay?.sampleIndex ?? sample.__sampleIndex ?? index,
      __resultIndex: sample.replay?.resultIndex ?? sample.__resultIndex ?? null,
    }));
}

export function selectAlphaReplaySamples(input, options = {}) {
  const samples = filterReplayTrainingSamples(
    Array.isArray(input) ? input.slice() : samplesFromAlphaReplayBuffer(input),
    options,
  );
  const maxSamples = Math.floor(Number(options.maxSamples || 0));
  if (!(maxSamples > 0) || samples.length <= maxSamples) return samples;
  const balanceBy = options.balanceBy || "none";
  const priorityBy = normalizePriorityBy(options.priorityBy);
  if (balanceBy === "none") {
    return priorityBy === "none"
      ? samples.slice(-maxSamples)
      : prioritizedSamples(samples, priorityBy).slice(0, maxSamples).map((entry) => entry.sample);
  }

  const groups = new Map();
  for (const [index, sample] of samples.entries()) {
    const key = sampleSelectionKey(sample, balanceBy);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ sample, index, priority: samplePriority(sample, priorityBy) });
  }
  const keys = [...groups.keys()].sort();
  if (priorityBy !== "none") {
    for (const group of groups.values()) group.sort(comparePriorityEntriesAscending);
  }
  const selected = [];
  while (selected.length < maxSamples && keys.some((key) => groups.get(key).length > 0)) {
    for (const key of keys) {
      const group = groups.get(key);
      if (!group.length) continue;
      selected.push(group.pop().sample);
      if (selected.length >= maxSamples) break;
    }
  }
  return selected.reverse();
}

export function summarizeAlphaReplayBuffer(buffer) {
  const samples = samplesFromAlphaReplayBuffer(buffer);
  return {
    schema: "zizi-el-alamein-alpha-replay-buffer-summary-v1",
    sampleCount: samples.length,
    duplicateMode: normalizeDuplicateMode(buffer?.duplicateMode),
    mergedSamples: samples.reduce((sum, sample) => sum + Math.max(0, Number(sample.replay?.mergedSamples || 1) - 1), 0),
    sources: sourceList(samples),
    sides: countBy(samples, (sample) => sample.side || "unknown"),
    phases: countBy(samples, (sample) => sample.phaseId || "unknown"),
    outcomes: countBy(samples, outcomeBucket),
    outcomeSources: countBy(samples, (sample) => sample.outcomeSource || "unknown"),
    featureContracts: countBy(samples, (sample) => alphaTrainingSampleFeatureContractFingerprint(sample) || "missing"),
    averageOutcomeWeight: averageOutcomeWeight(samples),
  };
}

export function summarizeAlphaReplaySampleSelection(samples, options = {}) {
  const selectedSides = normalizeSideFilter(options.sides || options.side);
  return {
    schema: "zizi-el-alamein-alpha-replay-selection-summary-v1",
    sampleCount: (samples || []).length,
    maxSamples: Number(options.maxSamples || 0) || null,
    balanceBy: options.balanceBy || "none",
    priorityBy: normalizePriorityBy(options.priorityBy),
    minSelectedVisitShare: optionalFiniteOrNull(options.minSelectedVisitShare),
    selectedSides,
    averagePriority: averagePriority(samples, options.priorityBy),
    sides: countBy(samples, (sample) => sample.side || "unknown"),
    phases: countBy(samples, (sample) => sample.phaseId || "unknown"),
    outcomes: countBy(samples, outcomeBucket),
    outcomeSources: countBy(samples, (sample) => sample.outcomeSource || "unknown"),
    featureContracts: countBy(samples, (sample) => alphaTrainingSampleFeatureContractFingerprint(sample) || "missing"),
    averageOutcomeWeight: averageOutcomeWeight(samples),
  };
}

export function analyzeAlphaReplayBufferQuality(input, options = {}) {
  const samples = Array.isArray(input) ? input.slice() : samplesFromAlphaReplayBuffer(input);
  const policyTolerance = Math.max(0, Number(options.policyTolerance ?? 0.02));
  const expectedFeatureContract = options.expectedFeatureContract === false
    ? null
    : alphaTrainingSampleFeatureContract(options.expectedFeatureContract || alphaTrainingFeatureContract());
  const minSamples = Math.max(0, Math.floor(Number(options.minSamples ?? 1)));
  const minSides = Math.max(0, Math.floor(Number(options.minSides ?? 1)));
  const maxInvalidSamples = Math.max(0, Math.floor(Number(options.maxInvalidSamples ?? 0)));
  const issues = samples.flatMap((sample, index) => sampleQualityIssues(sample, index, {
    policyTolerance,
    expectedFeatureContract,
    requireFeatureContract: options.requireFeatureContract,
  }));
  const invalid = issues.filter((entry) => entry.severity === "error");
  const warnings = issues.filter((entry) => entry.severity === "warning");
  const invalidSampleCount = new Set(invalid.map((entry) => entry.sampleIndex)).size;
  const sides = countBy(samples, (sample) => sample.side || "unknown");
  const phases = countBy(samples, (sample) => sample.phaseId || "unknown");
  const reasons = [];
  if (samples.length < minSamples) reasons.push("sample_count_too_low");
  if (Object.keys(sides).filter((side) => side !== "unknown").length < minSides) reasons.push("side_coverage_too_narrow");
  if (invalidSampleCount > maxInvalidSamples) reasons.push("invalid_samples");
  const issueCounts = countBy(issues, (issue) => issue.reason);
  return {
    schema: ALPHA_REPLAY_BUFFER_QUALITY_SCHEMA,
    ok: reasons.length === 0,
    reason: reasons[0] || null,
    reasons,
    sampleCount: samples.length,
    validSampleCount: Math.max(0, samples.length - invalidSampleCount),
    invalidSampleCount,
    warningCount: warnings.length,
    minSamples,
    minSides,
    maxInvalidSamples,
    policyTolerance,
    sides,
    phases,
    outcomes: countBy(samples, outcomeBucket),
    outcomeSources: countBy(samples, (sample) => sample.outcomeSource || "unknown"),
    featureContracts: countBy(samples, (sample) => alphaTrainingSampleFeatureContractFingerprint(sample) || "missing"),
    averageOutcomeWeight: averageOutcomeWeight(samples),
    issueCounts,
    issues: issues.slice(0, Math.max(0, Math.floor(Number(options.issueLimit ?? 24)))),
  };
}

export function validateAlphaReplayBufferQuality(input, options = {}) {
  return analyzeAlphaReplayBufferQuality(input, options);
}

function latestDuplicateSamples(samples) {
  const byKey = new Map();
  for (const sample of samples || []) {
    if (sample?.schema !== "zizi-el-alamein-alpha-training-sample-v1") continue;
    const key = replaySampleKey(sample);
    const previous = byKey.get(key);
    const trajectoryIds = unionTrajectoryIds(previous?.trajectoryIds, sample.trajectoryIds);
    if (byKey.has(key)) byKey.delete(key);
    byKey.set(key, {
      ...sample,
      ...(trajectoryIds.length ? { trajectoryIds } : {}),
    });
  }
  return [...byKey.values()];
}

function mergeDuplicateSamples(samples) {
  const byKey = new Map();
  for (const sample of samples || []) {
    if (sample?.schema !== "zizi-el-alamein-alpha-training-sample-v1") continue;
    const key = replaySampleKey(sample);
    const group = byKey.get(key) || [];
    group.push(sample);
    if (byKey.has(key)) byKey.delete(key);
    byKey.set(key, group);
  }
  return [...byKey.values()].map(mergeSampleGroup);
}

function mergeSampleGroup(samples) {
  if (!samples.length) return null;
  if (samples.length === 1) return samples[0];
  const latest = samples[samples.length - 1];
  const sources = mergedSources(samples);
  const replayAggregate = mergeReplaySampleAggregates(samples);
  const trajectoryIds = unionTrajectoryIds(...samples.map((sample) => sample.trajectoryIds));
  return {
    ...latest,
    features: replayAggregateFeatures(replayAggregate),
    featureContract: mergeFeatureContracts(samples),
    rootValue: replayAggregateAverage(replayAggregate.rootValue),
    decision: mergeDecisionMetadata(samples, replayAggregate),
    outcome: replayAggregateAverage(replayAggregate.outcome),
    outcomeSource: mergedOutcomeSource(samples),
    outcomeWeight: replayAggregateAverage(replayAggregate.outcomeWeight) ?? 0,
    policy: mergePolicyTargets(samples),
    ...(trajectoryIds.length ? { trajectoryIds } : {}),
    __source: latest.__source || latest.replay?.source || sources[sources.length - 1] || "unknown",
    __sources: sources,
    __sampleIndex: latest.__sampleIndex ?? latest.replay?.sampleIndex ?? null,
    __resultIndex: latest.__resultIndex ?? latest.replay?.resultIndex ?? null,
    __mergedSamples: replayAggregate.sampleCount,
    __replayAggregate: replayAggregate,
  };
}

function sampleQualityIssues(sample, index, options) {
  const issues = [];
  const prefix = { sampleIndex: index, stateHash: sample?.stateHash || null };
  if (!sample || typeof sample !== "object") {
    return [{ ...prefix, severity: "error", reason: "invalid_sample" }];
  }
  if (!sample.stateHash) issues.push({ ...prefix, severity: "error", reason: "missing_state_hash" });
  if (!sample.side) issues.push({ ...prefix, severity: "error", reason: "missing_side" });
  if (!sample.phaseId) issues.push({ ...prefix, severity: "error", reason: "missing_phase_id" });
  const valueTarget = alphaTrainingValueTarget(sample);
  if (valueTarget.reason === "invalid_terminal_outcome") {
    issues.push({ ...prefix, severity: "error", reason: "invalid_terminal_outcome" });
  } else if (valueTarget.reason === "missing_outcome") {
    issues.push({ ...prefix, severity: "error", reason: "missing_outcome" });
  }
  if (valueTarget.reason === "policy_only") {
    if (sample.outcome !== null && sample.outcome !== undefined) {
      issues.push({ ...prefix, severity: "error", reason: "policy_only_outcome_present" });
    }
    if (Number(sample.outcomeWeight ?? 0) !== 0) {
      issues.push({ ...prefix, severity: "error", reason: "policy_only_outcome_weight" });
    }
  }
  if (sample.reanalysis?.stateHashMatches === false) {
    issues.push({ ...prefix, severity: "error", reason: "reanalysis_state_hash_mismatch" });
  }
  const sampleFeatureContract = alphaTrainingSampleFeatureContractFingerprint(sample);
  if (!sampleFeatureContract) {
    issues.push({ ...prefix, severity: options.requireFeatureContract ? "error" : "warning", reason: "missing_sample_feature_contract" });
  } else if (
    options.expectedFeatureContract?.fingerprint
    && sampleFeatureContract !== options.expectedFeatureContract.fingerprint
  ) {
    issues.push({
      ...prefix,
      severity: "error",
      reason: "sample_feature_contract_mismatch",
      featureContractFingerprint: sampleFeatureContract,
      expectedFeatureContractFingerprint: options.expectedFeatureContract.fingerprint,
    });
  }
  const policy = Array.isArray(sample.policy) ? sample.policy : [];
  if (!policy.length) {
    issues.push({ ...prefix, severity: "error", reason: "empty_policy" });
  } else {
    const shareTotal = policy.reduce((sum, entry) => sum + Math.max(0, Number(entry.visitShare || 0)), 0);
    const shareError = Math.abs(1 - shareTotal);
    if (shareError > options.policyTolerance) {
      issues.push({
        ...prefix,
        severity: "error",
        reason: "policy_share_not_normalized",
        policyShareTotal: round(shareTotal, 6),
      });
    }
    if (policy.some((entry) => !entry?.action)) {
      issues.push({ ...prefix, severity: "error", reason: "missing_policy_action" });
    }
  }
  if (valueTarget.reason !== "policy_only" && Number(sample.outcomeWeight ?? 1) <= 0) {
    issues.push({ ...prefix, severity: "warning", reason: "zero_outcome_weight" });
  }
  if (sample.decision && Number(sample.decision.rootVisits || 0) <= 0) {
    issues.push({ ...prefix, severity: "warning", reason: "missing_decision_root_visits" });
  }
  return issues;
}

function cleanReplaySample(sample, index) {
  const source = sample.__source || sample.replay?.source || "unknown";
  const sources = replaySources(sample, source);
  const mergedSamples = sampleMultiplicity(sample);
  const valueTarget = alphaTrainingValueTarget(sample);
  const policyOnly = valueTarget.reason === "policy_only";
  const invalidTerminal = valueTarget.reason === "invalid_terminal_outcome";
  const trajectoryIds = normalizeTrajectoryIds(sample.trajectoryIds);
  const replayAggregate = cleanReplaySampleAggregate(sample.__replayAggregate || sample.replay?.aggregate);
  return {
    schema: "zizi-el-alamein-alpha-training-sample-v1",
    stateHash: sample.stateHash || null,
    side: sample.side || null,
    turn: sample.turn ?? null,
    phaseId: sample.phaseId || null,
    ...(trajectoryIds.length ? { trajectoryIds } : {}),
    ...(sample.initialState ? { initialState: cloneJsonLike(sample.initialState) } : {}),
    features: { ...(sample.features || {}) },
    ...(cleanFeatureContract(sample.featureContract) ? { featureContract: cleanFeatureContract(sample.featureContract) } : {}),
    rootValue: finiteOrNull(sample.rootValue),
    decision: cleanDecisionMetadata(sample.decision),
    ...(cleanAlphaRecommendationMetadata(sample.alphaRecommendation)
      ? { alphaRecommendation: cleanAlphaRecommendationMetadata(sample.alphaRecommendation) }
      : {}),
    reanalysis: cleanReanalysisMetadata(sample.reanalysis),
    outcome: policyOnly || invalidTerminal ? null : finiteOrNull(sample.outcome),
    outcomeSource: typeof sample.outcomeSource === "string" ? sample.outcomeSource : null,
    outcomeWeight: policyOnly || invalidTerminal ? 0 : finiteOrDefault(sample.outcomeWeight, 1),
    policy: Array.isArray(sample.policy) ? sample.policy.map(cleanPolicyEntry).filter(Boolean) : [],
    replay: {
      key: replaySampleKey(sample),
      source,
      sources,
      sampleIndex: sample.__sampleIndex ?? sample.replay?.sampleIndex ?? index,
      resultIndex: sample.__resultIndex ?? sample.replay?.resultIndex ?? null,
      mergedSamples,
      ...(replayAggregate ? { aggregate: replayAggregate } : {}),
    },
  };
}

function cleanDecisionMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-self-play-decision-v1",
    selectedAction: value.selectedAction && typeof value.selectedAction === "object" ? value.selectedAction : null,
    selectedVisitShare: finiteOrNull(value.selectedVisitShare),
    selectedVisits: finiteOrNull(value.selectedVisits),
    selectedQ: finiteOrNull(value.selectedQ),
    selectedPrior: finiteOrNull(value.selectedPrior),
    selectedPolicyRank: finiteOrNull(value.selectedPolicyRank),
    selectedIsBest: booleanOrNull(value.selectedIsBest),
    selectedSource: typeof value.selectedSource === "string" ? value.selectedSource : null,
    selectionMode: typeof value.selectionMode === "string" ? value.selectionMode : "unknown",
    temperature: finiteOrDefault(value.temperature, 0),
    rootNoiseWeight: finiteOrDefault(value.rootNoiseWeight, 0),
    policyEntropy: finiteOrDefault(value.policyEntropy, 0),
    recommendationConfidence: finiteOrNull(value.recommendationConfidence),
    recommendationVisitMargin: finiteOrNull(value.recommendationVisitMargin),
    recommendationQMargin: finiteOrNull(value.recommendationQMargin),
    recommendationLabel: typeof value.recommendationLabel === "string" ? value.recommendationLabel : null,
    illegalCandidateCount: finiteOrNull(value.illegalCandidateCount),
    legalActionCount: finiteOrNull(value.legalActionCount),
    runtimeActionMode: typeof value.runtimeActionMode === "string" ? value.runtimeActionMode : null,
    runtimeAnalysisMode: typeof value.runtimeAnalysisMode === "string" ? value.runtimeAnalysisMode : null,
    policySize: finiteOrDefault(value.policySize, 0),
    searchIterations: finiteOrDefault(value.searchIterations, 0),
    rootVisits: finiteOrDefault(value.rootVisits, 0),
  };
}

function cleanAlphaRecommendationMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    ok: Boolean(value.ok),
    reason: typeof value.reason === "string" ? value.reason : null,
    selectedSource: typeof value.selectedSource === "string" ? value.selectedSource : null,
    illegalCandidateCount: finiteOrNull(value.illegalCandidateCount),
    legalActionCount: finiteOrNull(value.legalActionCount),
    runtimeActionMode: typeof value.runtimeActionMode === "string" ? value.runtimeActionMode : null,
    runtimeAnalysisMode: typeof value.runtimeAnalysisMode === "string" ? value.runtimeAnalysisMode : null,
  };
}

function cleanReanalysisMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-reanalysis-sample-v1",
    sourceStateHash: typeof value.sourceStateHash === "string" ? value.sourceStateHash : null,
    reanalyzedStateHash: typeof value.reanalyzedStateHash === "string" ? value.reanalyzedStateHash : null,
    stateHashMatches: booleanOrNull(value.stateHashMatches),
    sourceSide: typeof value.sourceSide === "string" ? value.sourceSide : null,
    sourceTurn: value.sourceTurn ?? null,
    sourcePhaseId: typeof value.sourcePhaseId === "string" ? value.sourcePhaseId : null,
    sourceRootValue: finiteOrNull(value.sourceRootValue),
    reanalyzedRootValue: finiteOrNull(value.reanalyzedRootValue),
    rootValueDelta: finiteOrNull(value.rootValueDelta),
    sourceSelectedAction: value.sourceSelectedAction && typeof value.sourceSelectedAction === "object"
      ? cloneJsonLike(value.sourceSelectedAction)
      : null,
    reanalyzedSelectedAction: value.reanalyzedSelectedAction && typeof value.reanalyzedSelectedAction === "object"
      ? cloneJsonLike(value.reanalyzedSelectedAction)
      : null,
    selectedActionChanged: booleanOrNull(value.selectedActionChanged),
    sourcePolicyEntropy: finiteOrNull(value.sourcePolicyEntropy),
    sourceSearchIterations: finiteOrNull(value.sourceSearchIterations),
    sourceRootVisits: finiteOrNull(value.sourceRootVisits),
    sourceOutcome: finiteOrNull(value.sourceOutcome),
    sourceOutcomeSource: typeof value.sourceOutcomeSource === "string" ? value.sourceOutcomeSource : null,
    sourceOutcomeWeight: finiteOrNull(value.sourceOutcomeWeight),
    sourceReplay: value.sourceReplay && typeof value.sourceReplay === "object"
      ? cloneJsonLike(value.sourceReplay)
      : null,
  };
}

function mergeDecisionMetadata(samples, replayAggregate = mergeReplaySampleAggregates(samples)) {
  const entries = (samples || [])
    .map((sample) => ({ decision: sample.decision }))
    .filter((entry) => entry.decision && typeof entry.decision === "object");
  if (!entries.length) return null;
  const decisions = entries.map((entry) => entry.decision);
  const latest = decisions[decisions.length - 1];
  const average = (key) => replayAggregateAverage(replayAggregate.decision[key]);
  return {
    schema: "zizi-el-alamein-alpha-self-play-decision-v1",
    selectedAction: latest.selectedAction && typeof latest.selectedAction === "object" ? latest.selectedAction : null,
    selectedVisitShare: average("selectedVisitShare"),
    selectedVisits: average("selectedVisits"),
    selectedQ: average("selectedQ"),
    selectedPrior: average("selectedPrior"),
    selectedPolicyRank: average("selectedPolicyRank"),
    selectedIsBest: booleanOrNull(latest.selectedIsBest),
    selectedSource: typeof latest.selectedSource === "string" ? latest.selectedSource : null,
    selectionMode: typeof latest.selectionMode === "string" ? latest.selectionMode : "unknown",
    temperature: average("temperature") ?? 0,
    rootNoiseWeight: average("rootNoiseWeight") ?? 0,
    policyEntropy: average("policyEntropy") ?? 0,
    recommendationConfidence: average("recommendationConfidence"),
    recommendationVisitMargin: average("recommendationVisitMargin"),
    recommendationQMargin: average("recommendationQMargin"),
    recommendationLabel: mergedRecommendationLabel(decisions),
    illegalCandidateCount: average("illegalCandidateCount"),
    legalActionCount: average("legalActionCount"),
    runtimeActionMode: typeof latest.runtimeActionMode === "string" ? latest.runtimeActionMode : null,
    runtimeAnalysisMode: typeof latest.runtimeAnalysisMode === "string" ? latest.runtimeAnalysisMode : null,
    policySize: average("policySize") ?? 0,
    searchIterations: average("searchIterations") ?? 0,
    rootVisits: average("rootVisits") ?? 0,
  };
}

function cleanPolicyEntry(entry) {
  if (!entry?.action) return null;
  const replayAggregate = cleanReplayPolicyAggregate(entry.replayAggregate);
  return {
    action: entry.action,
    visitShare: Number(entry.visitShare || 0),
    visits: Number(entry.visits || 0),
    q: entry.q !== null && entry.q !== undefined && entry.q !== "" && Number.isFinite(Number(entry.q))
      ? Number(entry.q)
      : null,
    prior: entry.prior !== null && entry.prior !== undefined && entry.prior !== "" && Number.isFinite(Number(entry.prior))
      ? Number(entry.prior)
      : null,
    ...(replayAggregate ? { replayAggregate } : {}),
  };
}

function cleanReplayPolicyAggregate(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== "zizi-el-alamein-alpha-replay-policy-aggregate-v1") return null;
  const targetWeight = nonnegativeFiniteOrNull(value.targetWeight);
  const qWeight = nonnegativeFiniteOrNull(value.qWeight);
  const priorWeight = nonnegativeFiniteOrNull(value.priorWeight);
  const qWeightedSum = finiteOrNull(value.qWeightedSum);
  const priorWeightedSum = finiteOrNull(value.priorWeightedSum);
  if (
    targetWeight === null
    || qWeight === null
    || priorWeight === null
    || qWeightedSum === null
    || priorWeightedSum === null
  ) return null;
  return {
    schema: "zizi-el-alamein-alpha-replay-policy-aggregate-v1",
    targetWeight,
    qWeightedSum,
    qWeight,
    priorWeightedSum,
    priorWeight,
  };
}

function replaySampleKey(sample) {
  return [
    sample.stateHash || "no-state",
    sample.side || "no-side",
    sample.phaseId || "no-phase",
    sample.turn ?? "no-turn",
  ].join("|");
}

function mergeFeatureContracts(samples) {
  const contracts = (samples || [])
    .map((sample) => cleanFeatureContract(sample.featureContract))
    .filter(Boolean);
  if (!contracts.length) return null;
  const fingerprints = [...new Set(contracts.map((contract) => contract.fingerprint))];
  if (fingerprints.length === 1) return contracts[contracts.length - 1];
  return {
    schema: contracts[contracts.length - 1].schema,
    fingerprint: "mixed",
    sources: fingerprints.sort(),
  };
}

function cleanFeatureContract(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (typeof value.schema !== "string" || typeof value.fingerprint !== "string") return null;
  return {
    schema: value.schema,
    fingerprint: value.fingerprint,
  };
}

function mergePolicyTargets(samples) {
  const byAction = new Map();
  for (const sample of samples) {
    const sampleWeight = policySampleWeight(sample);
    for (const entry of sample.policy || []) {
      if (!entry?.action) continue;
      const key = stableActionKey(entry.action);
      const stats = replayPolicyEntryAggregate(entry, sampleWeight);
      const aggregate = byAction.get(key) || {
        action: entry.action,
        targetWeight: 0,
        visits: 0,
        qWeightedSum: 0,
        qWeight: 0,
        priorWeightedSum: 0,
        priorWeight: 0,
      };
      aggregate.targetWeight += stats.targetWeight;
      aggregate.visits += Math.max(0, Number(entry.visits || 0));
      aggregate.qWeightedSum += stats.qWeightedSum;
      aggregate.qWeight += stats.qWeight;
      aggregate.priorWeightedSum += stats.priorWeightedSum;
      aggregate.priorWeight += stats.priorWeight;
      byAction.set(key, aggregate);
    }
  }
  const total = [...byAction.values()].reduce((sum, entry) => sum + entry.targetWeight, 0);
  return [...byAction.entries()]
    .map(([key, entry]) => ({
      action: entry.action,
      visitShare: total > 0 ? round(entry.targetWeight / total, 6) : 0,
      visits: entry.visits,
      q: entry.qWeight > 0 ? round(entry.qWeightedSum / entry.qWeight, 6) : null,
      prior: entry.priorWeight > 0 ? round(entry.priorWeightedSum / entry.priorWeight, 6) : null,
      replayAggregate: {
        schema: "zizi-el-alamein-alpha-replay-policy-aggregate-v1",
        targetWeight: entry.targetWeight,
        qWeightedSum: entry.qWeightedSum,
        qWeight: entry.qWeight,
        priorWeightedSum: entry.priorWeightedSum,
        priorWeight: entry.priorWeight,
      },
      key,
    }))
    .sort((a, b) => b.visitShare - a.visitShare || codeUnitCompare(a.key, b.key))
    .map(({ key, ...entry }) => entry);
}

function policySampleWeight(sample) {
  const visits = (sample.policy || []).reduce((sum, entry) => sum + Math.max(0, Number(entry.visits || 0)), 0);
  if (visits > 0) return visits;
  return Math.max(0, finiteOrDefault(sample.policyWeight, sampleMultiplicity(sample)));
}

function policyEntryWeight(entry, sampleWeight) {
  const visits = Math.max(0, Number(entry.visits || 0));
  if (visits > 0) return visits;
  return Math.max(0, Number(entry.visitShare || 0)) * sampleWeight;
}

function replayPolicyEntryAggregate(entry, sampleWeight) {
  const stored = cleanReplayPolicyAggregate(entry.replayAggregate);
  if (stored) return stored;
  const targetWeight = policyEntryWeight(entry, sampleWeight);
  const q = entry.q !== null && entry.q !== undefined && entry.q !== "" && Number.isFinite(Number(entry.q))
    ? Number(entry.q)
    : null;
  const prior = entry.prior !== null && entry.prior !== undefined && entry.prior !== "" && Number.isFinite(Number(entry.prior))
    ? Number(entry.prior)
    : null;
  return {
    schema: "zizi-el-alamein-alpha-replay-policy-aggregate-v1",
    targetWeight,
    qWeightedSum: q === null ? 0 : q * targetWeight,
    qWeight: q === null ? 0 : targetWeight,
    priorWeightedSum: prior === null ? 0 : prior * targetWeight,
    priorWeight: prior === null ? 0 : targetWeight,
  };
}

function mergedOutcomeSource(samples) {
  const contributors = valueContributors(samples);
  if (!contributors.length) return "policy_only_merged";
  const sources = [...new Set(contributors.map((entry) => entry.target.source).filter(Boolean))];
  const terminalSources = new Set(["terminal_result", "merged_terminal_expectation"]);
  if (contributors.length > 1 && sources.every((source) => terminalSources.has(source))) {
    return "merged_terminal_expectation";
  }
  if (sources.length === 1) return sources[0];
  return sources.length ? "merged" : null;
}

function valueContributors(samples) {
  return (samples || []).map((sample) => ({ sample, target: alphaTrainingValueTarget(sample) }))
    .filter((entry) => entry.target.usable);
}

function normalizeReplayValueSample(sample) {
  const target = alphaTrainingValueTarget(sample);
  if (target.reason !== "policy_only" && target.reason !== "invalid_terminal_outcome") return sample;
  return {
    ...sample,
    outcome: null,
    outcomeWeight: 0,
  };
}

function mergedSources(samples) {
  return [...new Set(samples.flatMap((sample) => replaySources(sample, sample.__source || sample.replay?.source || null)))].filter(Boolean).sort();
}

function replaySources(sample, fallback = null) {
  const sources = sample.__sources || sample.replay?.sources;
  if (Array.isArray(sources)) return [...new Set(sources.map(String).filter(Boolean))].sort();
  const source = fallback || sample.__source || sample.replay?.source;
  return source ? [String(source)] : [];
}

function sampleMultiplicity(sample) {
  const value = Number(sample?.__mergedSamples ?? sample?.replay?.mergedSamples ?? 1);
  return Number.isInteger(value) && value > 0 ? value : 1;
}

function mergeReplaySampleAggregates(samples) {
  const output = emptyReplaySampleAggregate();
  for (const sample of samples || []) {
    const stored = cleanReplaySampleAggregate(sample?.__replayAggregate || sample?.replay?.aggregate);
    if (stored && replayAggregateMatchesMultiplicity(sample, stored)) {
      addReplaySampleAggregate(output, stored);
      continue;
    }
    addSampleToReplayAggregate(output, sample);
  }
  return output;
}

function replayAggregateMatchesMultiplicity(sample, aggregate) {
  const declared = Number(sample?.__mergedSamples ?? sample?.replay?.mergedSamples);
  return Number.isInteger(declared) && declared > 0 && declared === aggregate.sampleCount;
}

function emptyReplaySampleAggregate() {
  return {
    schema: "zizi-el-alamein-alpha-replay-sample-aggregate-v1",
    sampleCount: 0,
    features: {},
    rootValue: emptyReplayNumericAggregate(),
    decision: Object.fromEntries(REPLAY_DECISION_AGGREGATE_KEYS.map((key) => [key, emptyReplayNumericAggregate()])),
    outcome: emptyReplayNumericAggregate(),
    outcomeWeight: emptyReplayNumericAggregate(),
  };
}

function emptyReplayNumericAggregate() {
  return { sum: 0, weight: 0 };
}

function addReplaySampleAggregate(target, source) {
  target.sampleCount += source.sampleCount;
  for (const [key, stats] of Object.entries(source.features)) {
    if (!target.features[key]) target.features[key] = emptyReplayNumericAggregate();
    addReplayNumericAggregate(target.features[key], stats);
  }
  addReplayNumericAggregate(target.rootValue, source.rootValue);
  for (const key of REPLAY_DECISION_AGGREGATE_KEYS) {
    addReplayNumericAggregate(target.decision[key], source.decision[key]);
  }
  addReplayNumericAggregate(target.outcome, source.outcome);
  addReplayNumericAggregate(target.outcomeWeight, source.outcomeWeight);
}

function addSampleToReplayAggregate(target, sample) {
  const count = sampleMultiplicity(sample);
  target.sampleCount += count;
  for (const [key, rawValue] of Object.entries(sample?.features || {})) {
    const value = finitePresentOrNull(rawValue);
    if (value === null) continue;
    if (!target.features[key]) target.features[key] = emptyReplayNumericAggregate();
    target.features[key].sum += value * count;
    target.features[key].weight += count;
  }
  addReplayValue(target.rootValue, sample?.rootValue, count);
  for (const key of REPLAY_DECISION_AGGREGATE_KEYS) {
    addReplayValue(target.decision[key], sample?.decision?.[key], count);
  }
  const valueTarget = alphaTrainingValueTarget(sample);
  if (valueTarget.usable) {
    const outcomeWeight = Math.max(0, finiteOrDefault(valueTarget.outcomeWeight, 1));
    target.outcome.sum += valueTarget.outcome * outcomeWeight * count;
    target.outcome.weight += outcomeWeight * count;
    target.outcomeWeight.sum += outcomeWeight * count;
    target.outcomeWeight.weight += count;
  }
}

function addReplayValue(target, rawValue, weight) {
  const value = finitePresentOrNull(rawValue);
  if (value === null || !(weight > 0)) return;
  target.sum += value * weight;
  target.weight += weight;
}

function addReplayNumericAggregate(target, source) {
  target.sum += source.sum;
  target.weight += source.weight;
}

function replayAggregateFeatures(aggregate) {
  return Object.fromEntries(Object.entries(aggregate.features)
    .sort(([left], [right]) => codeUnitCompare(left, right))
    .map(([key, stats]) => [key, replayAggregateAverage(stats)])
    .filter(([, value]) => value !== null));
}

function replayAggregateAverage(value) {
  return value?.weight > 0 ? round(value.sum / value.weight, 6) : null;
}

function cleanReplaySampleAggregate(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== "zizi-el-alamein-alpha-replay-sample-aggregate-v1") return null;
  const sampleCount = Number(value.sampleCount);
  if (!Number.isInteger(sampleCount) || sampleCount < 1) return null;
  const rootValue = cleanReplayNumericAggregate(value.rootValue);
  const outcome = cleanReplayNumericAggregate(value.outcome);
  const outcomeWeight = cleanReplayNumericAggregate(value.outcomeWeight);
  if (!rootValue || !outcome || !outcomeWeight) return null;
  const features = {};
  for (const [key, stats] of Object.entries(value.features || {}).sort(([left], [right]) => codeUnitCompare(left, right))) {
    const clean = cleanReplayNumericAggregate(stats);
    if (!clean) return null;
    features[key] = clean;
  }
  const decision = {};
  for (const key of REPLAY_DECISION_AGGREGATE_KEYS) {
    const clean = cleanReplayNumericAggregate(value.decision?.[key]);
    if (!clean) return null;
    decision[key] = clean;
  }
  return {
    schema: "zizi-el-alamein-alpha-replay-sample-aggregate-v1",
    sampleCount,
    features,
    rootValue,
    decision,
    outcome,
    outcomeWeight,
  };
}

function cleanReplayNumericAggregate(value) {
  const sum = finitePresentOrNull(value?.sum);
  const weight = nonnegativeFiniteOrNull(value?.weight);
  return sum === null || weight === null ? null : { sum, weight };
}

function sampleSelectionKey(sample, balanceBy) {
  const actionType = selectedSampleActionType(sample);
  if (balanceBy === "side") return sample.side || "unknown-side";
  if (balanceBy === "phase") return sample.phaseId || "unknown-phase";
  if (balanceBy === "actionType") return actionType;
  if (balanceBy === "outcome") return outcomeBucket(sample);
  if (balanceBy === "sideOutcome") return `${sample.side || "unknown-side"}|${outcomeBucket(sample)}`;
  if (balanceBy === "sidePhase") return `${sample.side || "unknown-side"}|${sample.phaseId || "unknown-phase"}`;
  if (balanceBy === "sidePhaseAction") return `${sample.side || "unknown-side"}|${sample.phaseId || "unknown-phase"}|${actionType}`;
  if (balanceBy === "sidePhaseOutcome") return `${sample.side || "unknown-side"}|${sample.phaseId || "unknown-phase"}|${outcomeBucket(sample)}`;
  return "all";
}

function selectedSampleActionType(sample) {
  return sample?.decision?.selectedAction?.type
    || sample?.policy?.[0]?.action?.type
    || "unknown-action";
}

function filterReplayTrainingSamples(samples, options = {}) {
  const minSelectedVisitShare = optionalFiniteOrNull(options.minSelectedVisitShare);
  const selectedSides = normalizeSideFilter(options.sides || options.side);
  return (samples || []).filter((sample) => {
    if (selectedSides.length && !selectedSides.includes(sample?.side)) return false;
    if (minSelectedVisitShare === null) return true;
    const selectedShare = Number(sample?.decision?.selectedVisitShare);
    return Number.isFinite(selectedShare) && selectedShare >= minSelectedVisitShare;
  });
}

function normalizeSideFilter(value) {
  const values = Array.isArray(value) ? value : String(value || "").split(",");
  return [...new Set(values.map((side) => String(side).trim().toLowerCase()).filter((side) => (
    side === "axis" || side === "allied"
  )))].sort();
}

function prioritizedSamples(samples, priorityBy) {
  return (samples || [])
    .map((sample, index) => ({ sample, index, priority: samplePriority(sample, priorityBy) }))
    .sort(comparePriorityEntriesDescending);
}

function comparePriorityEntriesDescending(left, right) {
  return right.priority - left.priority || right.index - left.index;
}

function comparePriorityEntriesAscending(left, right) {
  return left.priority - right.priority || left.index - right.index;
}

function samplePriority(sample, priorityBy = "none") {
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

function averagePriority(samples, priorityBy) {
  const mode = normalizePriorityBy(priorityBy);
  if (mode === "none") return null;
  const priorities = (samples || []).map((sample) => samplePriority(sample, mode));
  if (!priorities.length) return 0;
  return round(priorities.reduce((sum, value) => sum + value, 0) / priorities.length, 6);
}

function outcomeBucket(sample) {
  const target = alphaTrainingValueTarget(sample);
  if (target.reason === "policy_only") return "policy_only";
  const outcome = target.outcome;
  if (!Number.isFinite(outcome)) return "unknown";
  if (outcome > 0.25) return "win";
  if (outcome < -0.25) return "loss";
  return "neutral_value";
}

function sourceList(samples) {
  return [...new Set((samples || []).flatMap((sample) => replaySources(sample, sample.replay?.source || sample.__source)))].sort();
}

function countBy(items, selector) {
  return (items || []).reduce((counts, item) => {
    const key = selector(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function averageOutcomeWeight(samples) {
  const weights = (samples || []).map((sample) => Number(sample.outcomeWeight ?? 1)).filter(Number.isFinite);
  if (!weights.length) return 0;
  return round(weights.reduce((sum, value) => sum + value, 0) / weights.length, 4);
}

function finiteOrNull(value) {
  if (value === null || value === undefined) return null;
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function finitePresentOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function optionalFiniteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function nonnegativeFiniteOrNull(value) {
  const next = finiteOrNull(value);
  return next !== null && next >= 0 ? next : null;
}

function finiteOrDefault(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function booleanOrNull(value) {
  if (value === true || value === false) return value;
  return null;
}

function mergedRecommendationLabel(decisions) {
  const labels = (decisions || [])
    .map((decision) => decision?.recommendationLabel)
    .filter((label) => typeof label === "string" && label);
  if (!labels.length) return null;
  return labels[labels.length - 1];
}

function stableActionKey(action) {
  if (!action) return "";
  const keys = Object.keys(action).sort();
  return keys.map((key) => `${key}:${stableValue(action[key])}`).join("|");
}

function stableValue(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableValue).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableValue(value[key])}`).join(",")}}`;
}

function codeUnitCompare(left, right) {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function cloneJsonLike(value) {
  return value ? JSON.parse(JSON.stringify(value)) : value;
}

function normalizeDuplicateMode(value) {
  return value === "merge" ? "merge" : "latest";
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

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}
