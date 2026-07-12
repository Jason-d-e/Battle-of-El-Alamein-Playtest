import { flattenAlphaSelfPlaySamples } from "./ai-alpha-training.js";

export const ALPHA_REPLAY_BUFFER_SCHEMA = "zizi-el-alamein-alpha-replay-buffer-v1";
export const ALPHA_REPLAY_BUFFER_QUALITY_SCHEMA = "zizi-el-alamein-alpha-replay-quality-v1";

const DEFAULT_MAX_SAMPLES = 4096;

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
  const deduped = duplicateMode === "merge"
    ? mergeDuplicateSamples(samples)
    : latestDuplicateSamples(samples);
  const windowed = deduped.slice(-maxSamples).map((sample, index) => cleanReplaySample(sample, index));
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
    }),
    samples: windowed,
  };
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
  const samples = Array.isArray(input) ? input.slice() : samplesFromAlphaReplayBuffer(input);
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
    averageOutcomeWeight: averageOutcomeWeight(samples),
  };
}

export function summarizeAlphaReplaySampleSelection(samples, options = {}) {
  return {
    schema: "zizi-el-alamein-alpha-replay-selection-summary-v1",
    sampleCount: (samples || []).length,
    maxSamples: Number(options.maxSamples || 0) || null,
    balanceBy: options.balanceBy || "none",
    priorityBy: normalizePriorityBy(options.priorityBy),
    averagePriority: averagePriority(samples, options.priorityBy),
    sides: countBy(samples, (sample) => sample.side || "unknown"),
    phases: countBy(samples, (sample) => sample.phaseId || "unknown"),
    outcomes: countBy(samples, outcomeBucket),
    outcomeSources: countBy(samples, (sample) => sample.outcomeSource || "unknown"),
    averageOutcomeWeight: averageOutcomeWeight(samples),
  };
}

export function analyzeAlphaReplayBufferQuality(input, options = {}) {
  const samples = Array.isArray(input) ? input.slice() : samplesFromAlphaReplayBuffer(input);
  const policyTolerance = Math.max(0, Number(options.policyTolerance ?? 0.02));
  const minSamples = Math.max(0, Math.floor(Number(options.minSamples ?? 1)));
  const minSides = Math.max(0, Math.floor(Number(options.minSides ?? 1)));
  const maxInvalidSamples = Math.max(0, Math.floor(Number(options.maxInvalidSamples ?? 0)));
  const issues = samples.flatMap((sample, index) => sampleQualityIssues(sample, index, { policyTolerance }));
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
    if (byKey.has(key)) byKey.delete(key);
    byKey.set(key, sample);
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
  const outcome = weightedAverage(
    samples.map((sample) => sample.outcome),
    samples.map((sample) => finiteOrDefault(sample.outcomeWeight, 1)),
  );
  return {
    ...latest,
    features: mergeFeatureMaps(samples),
    rootValue: averageFinite(samples.map((sample) => sample.rootValue)),
    decision: mergeDecisionMetadata(samples),
    outcome,
    outcomeSource: mergedOutcomeSource(samples),
    outcomeWeight: averageFinite(samples.map((sample) => finiteOrDefault(sample.outcomeWeight, 1))) ?? 1,
    policy: mergePolicyTargets(samples),
    __source: latest.__source || latest.replay?.source || sources[sources.length - 1] || "unknown",
    __sources: sources,
    __sampleIndex: latest.__sampleIndex ?? latest.replay?.sampleIndex ?? null,
    __resultIndex: latest.__resultIndex ?? latest.replay?.resultIndex ?? null,
    __mergedSamples: samples.length,
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
  if (!Number.isFinite(Number(sample.outcome))) issues.push({ ...prefix, severity: "error", reason: "missing_outcome" });
  if (sample.reanalysis?.stateHashMatches === false) {
    issues.push({ ...prefix, severity: "error", reason: "reanalysis_state_hash_mismatch" });
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
  if (Number(sample.outcomeWeight ?? 1) <= 0) {
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
  const mergedSamples = Math.max(1, Number(sample.__mergedSamples || sample.replay?.mergedSamples || 1));
  return {
    schema: "zizi-el-alamein-alpha-training-sample-v1",
    stateHash: sample.stateHash || null,
    side: sample.side || null,
    turn: sample.turn ?? null,
    phaseId: sample.phaseId || null,
    ...(sample.initialState ? { initialState: cloneJsonLike(sample.initialState) } : {}),
    features: { ...(sample.features || {}) },
    rootValue: finiteOrNull(sample.rootValue),
    decision: cleanDecisionMetadata(sample.decision),
    reanalysis: cleanReanalysisMetadata(sample.reanalysis),
    outcome: finiteOrNull(sample.outcome),
    outcomeSource: typeof sample.outcomeSource === "string" ? sample.outcomeSource : null,
    outcomeWeight: finiteOrDefault(sample.outcomeWeight, 1),
    policy: Array.isArray(sample.policy) ? sample.policy.map(cleanPolicyEntry).filter(Boolean) : [],
    replay: {
      key: replaySampleKey(sample),
      source,
      sources,
      sampleIndex: sample.__sampleIndex ?? sample.replay?.sampleIndex ?? index,
      resultIndex: sample.__resultIndex ?? sample.replay?.resultIndex ?? null,
      mergedSamples,
    },
  };
}

function cleanDecisionMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-self-play-decision-v1",
    selectedAction: value.selectedAction && typeof value.selectedAction === "object" ? value.selectedAction : null,
    selectionMode: typeof value.selectionMode === "string" ? value.selectionMode : "unknown",
    temperature: finiteOrDefault(value.temperature, 0),
    rootNoiseWeight: finiteOrDefault(value.rootNoiseWeight, 0),
    policyEntropy: finiteOrDefault(value.policyEntropy, 0),
    policySize: finiteOrDefault(value.policySize, 0),
    searchIterations: finiteOrDefault(value.searchIterations, 0),
    rootVisits: finiteOrDefault(value.rootVisits, 0),
  };
}

function cleanReanalysisMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-reanalysis-sample-v1",
    sourceStateHash: typeof value.sourceStateHash === "string" ? value.sourceStateHash : null,
    reanalyzedStateHash: typeof value.reanalyzedStateHash === "string" ? value.reanalyzedStateHash : null,
    stateHashMatches: value.stateHashMatches !== false,
    sourceSide: typeof value.sourceSide === "string" ? value.sourceSide : null,
    sourceTurn: value.sourceTurn ?? null,
    sourcePhaseId: typeof value.sourcePhaseId === "string" ? value.sourcePhaseId : null,
    sourceRootValue: finiteOrNull(value.sourceRootValue),
    sourceOutcome: finiteOrNull(value.sourceOutcome),
    sourceOutcomeSource: typeof value.sourceOutcomeSource === "string" ? value.sourceOutcomeSource : null,
    sourceOutcomeWeight: finiteOrDefault(value.sourceOutcomeWeight, 1),
    sourceReplay: value.sourceReplay && typeof value.sourceReplay === "object"
      ? cloneJsonLike(value.sourceReplay)
      : null,
  };
}

function mergeDecisionMetadata(samples) {
  const decisions = (samples || []).map((sample) => sample.decision).filter((decision) => decision && typeof decision === "object");
  if (!decisions.length) return null;
  const latest = decisions[decisions.length - 1];
  return {
    schema: "zizi-el-alamein-alpha-self-play-decision-v1",
    selectedAction: latest.selectedAction && typeof latest.selectedAction === "object" ? latest.selectedAction : null,
    selectionMode: typeof latest.selectionMode === "string" ? latest.selectionMode : "unknown",
    temperature: averageFinite(decisions.map((decision) => decision.temperature)) ?? 0,
    rootNoiseWeight: averageFinite(decisions.map((decision) => decision.rootNoiseWeight)) ?? 0,
    policyEntropy: averageFinite(decisions.map((decision) => decision.policyEntropy)) ?? 0,
    policySize: averageFinite(decisions.map((decision) => decision.policySize)) ?? 0,
    searchIterations: averageFinite(decisions.map((decision) => decision.searchIterations)) ?? 0,
    rootVisits: averageFinite(decisions.map((decision) => decision.rootVisits)) ?? 0,
  };
}

function cleanPolicyEntry(entry) {
  if (!entry?.action) return null;
  return {
    action: entry.action,
    visitShare: Number(entry.visitShare || 0),
    visits: Number(entry.visits || 0),
    q: Number.isFinite(Number(entry.q)) ? Number(entry.q) : null,
    prior: Number.isFinite(Number(entry.prior)) ? Number(entry.prior) : null,
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

function mergeFeatureMaps(samples) {
  const keys = new Set();
  for (const sample of samples) {
    for (const key of Object.keys(sample.features || {})) keys.add(key);
  }
  const features = {};
  for (const key of keys) {
    const value = averageFinite(samples.map((sample) => sample.features?.[key]));
    if (value !== null) features[key] = value;
  }
  return features;
}

function mergePolicyTargets(samples) {
  const byAction = new Map();
  for (const sample of samples) {
    const sampleWeight = policySampleWeight(sample);
    for (const entry of sample.policy || []) {
      if (!entry?.action) continue;
      const key = stableActionKey(entry.action);
      const weight = policyEntryWeight(entry, sampleWeight);
      const aggregate = byAction.get(key) || {
        action: entry.action,
        targetWeight: 0,
        visits: 0,
        qValues: [],
        qWeights: [],
        priorValues: [],
        priorWeights: [],
      };
      aggregate.targetWeight += weight;
      aggregate.visits += Math.max(0, Number(entry.visits || 0));
      if (Number.isFinite(Number(entry.q))) {
        aggregate.qValues.push(Number(entry.q));
        aggregate.qWeights.push(weight);
      }
      if (Number.isFinite(Number(entry.prior))) {
        aggregate.priorValues.push(Number(entry.prior));
        aggregate.priorWeights.push(weight);
      }
      byAction.set(key, aggregate);
    }
  }
  const total = [...byAction.values()].reduce((sum, entry) => sum + entry.targetWeight, 0);
  return [...byAction.entries()]
    .map(([key, entry]) => ({
      action: entry.action,
      visitShare: total > 0 ? round(entry.targetWeight / total, 6) : 0,
      visits: entry.visits,
      q: weightedAverage(entry.qValues, entry.qWeights),
      prior: weightedAverage(entry.priorValues, entry.priorWeights),
      key,
    }))
    .sort((a, b) => b.visitShare - a.visitShare || a.key.localeCompare(b.key))
    .map(({ key, ...entry }) => entry);
}

function policySampleWeight(sample) {
  const visits = (sample.policy || []).reduce((sum, entry) => sum + Math.max(0, Number(entry.visits || 0)), 0);
  if (visits > 0) return visits;
  return Math.max(0, finiteOrDefault(sample.outcomeWeight, 1));
}

function policyEntryWeight(entry, sampleWeight) {
  const visits = Math.max(0, Number(entry.visits || 0));
  if (visits > 0) return visits;
  return Math.max(0, Number(entry.visitShare || 0)) * sampleWeight;
}

function mergedOutcomeSource(samples) {
  const sources = [...new Set(samples.map((sample) => sample.outcomeSource).filter(Boolean))];
  if (sources.length === 1) return sources[0];
  return sources.length ? "merged" : null;
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

function sampleSelectionKey(sample, balanceBy) {
  if (balanceBy === "side") return sample.side || "unknown-side";
  if (balanceBy === "phase") return sample.phaseId || "unknown-phase";
  if (balanceBy === "outcome") return outcomeBucket(sample);
  if (balanceBy === "sideOutcome") return `${sample.side || "unknown-side"}|${outcomeBucket(sample)}`;
  if (balanceBy === "sidePhase") return `${sample.side || "unknown-side"}|${sample.phaseId || "unknown-phase"}`;
  if (balanceBy === "sidePhaseOutcome") return `${sample.side || "unknown-side"}|${sample.phaseId || "unknown-phase"}|${outcomeBucket(sample)}`;
  return "all";
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

function averagePriority(samples, priorityBy) {
  const mode = normalizePriorityBy(priorityBy);
  if (mode === "none") return null;
  const priorities = (samples || []).map((sample) => samplePriority(sample, mode));
  if (!priorities.length) return 0;
  return round(priorities.reduce((sum, value) => sum + value, 0) / priorities.length, 6);
}

function outcomeBucket(sample) {
  const outcome = Number(sample?.outcome);
  if (!Number.isFinite(outcome)) return "unknown";
  if (outcome > 0.25) return "win";
  if (outcome < -0.25) return "loss";
  return "draw";
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

function finiteOrDefault(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function averageFinite(values) {
  const clean = (values || []).map(Number).filter(Number.isFinite);
  if (!clean.length) return null;
  return round(clean.reduce((sum, value) => sum + value, 0) / clean.length, 6);
}

function weightedAverage(values, weights) {
  let total = 0;
  let weighted = 0;
  for (let index = 0; index < (values || []).length; index += 1) {
    const value = Number(values[index]);
    if (!Number.isFinite(value)) continue;
    const weight = Math.max(0, Number(weights?.[index] ?? 1));
    if (!(weight > 0)) continue;
    total += weight;
    weighted += value * weight;
  }
  return total > 0 ? round(weighted / total, 6) : null;
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

function cloneJsonLike(value) {
  return value ? JSON.parse(JSON.stringify(value)) : value;
}

function normalizeDuplicateMode(value) {
  return value === "merge" ? "merge" : "latest";
}

function normalizePriorityBy(value) {
  return value === "surprise" || value === "policyEntropy" ? value : "none";
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}
