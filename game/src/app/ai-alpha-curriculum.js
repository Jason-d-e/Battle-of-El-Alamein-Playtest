import { alphaSnapshotSamplesFromInputs } from "./ai-alpha-challenge-suite.js";
import { selectAlphaReplaySamples } from "./ai-alpha-replay-buffer.js";

export const ALPHA_SELF_PLAY_CURRICULUM_SCHEMA = "zizi-el-alamein-alpha-self-play-curriculum-v1";

export function buildAlphaSelfPlayCurriculum(inputs = [], options = {}) {
  const sourceEligible = alphaSnapshotSamplesFromInputs(inputs);
  const filteredEligible = filterCurriculumEligibleSamples(sourceEligible, options);
  const maxPositions = positiveInteger(options.maxPositions, 16);
  const balanceBy = options.balanceBy || "sidePhaseAction";
  const priorityBy = options.priorityBy || "policyEntropy";
  const eligible = deduplicateCurriculumSamples(filteredEligible, priorityBy);
  const selected = balanceBy === "sidePhaseAction"
    ? selectHierarchicalCurriculumSamples(eligible, { maxPositions, priorityBy })
    : selectAlphaReplaySamples(eligible, {
      maxSamples: maxPositions,
      balanceBy,
      priorityBy,
    });
  const positions = selected.map((sample, index) => curriculumPosition(sample, index));
  return {
    schema: ALPHA_SELF_PLAY_CURRICULUM_SCHEMA,
    generatedAt: options.generatedAt || new Date().toISOString(),
    sourceEligiblePositions: sourceEligible.length,
    filteredEligiblePositions: filteredEligible.length,
    eligiblePositions: eligible.length,
    duplicateStatePositions: filteredEligible.length - eligible.length,
    selectedPositions: positions.length,
    maxPositions,
    balanceBy,
    priorityBy,
    filters: curriculumFilters(options),
    coverage: curriculumCoverage(positions),
    positions,
  };
}

function deduplicateCurriculumSamples(samples, priorityBy) {
  const unique = [];
  const indexByStateHash = new Map();
  for (const sample of samples || []) {
    const stateHash = typeof sample?.stateHash === "string" && sample.stateHash ? sample.stateHash : null;
    if (!stateHash) {
      unique.push(sample);
      continue;
    }
    const existingIndex = indexByStateHash.get(stateHash);
    if (existingIndex === undefined) {
      indexByStateHash.set(stateHash, unique.length);
      unique.push(sample);
      continue;
    }
    const existing = unique[existingIndex];
    if (curriculumSamplePriority(sample, priorityBy) > curriculumSamplePriority(existing, priorityBy)) {
      unique[existingIndex] = sample;
    }
  }
  return unique;
}

function filterCurriculumEligibleSamples(samples, options) {
  const sides = normalizeSides(options.sides ?? options.side);
  const outcomeSources = normalizeStrings(options.outcomeSources ?? options.outcomeSource);
  const minOutcome = optionalFinite(options.minOutcome);
  const maxOutcome = optionalFinite(options.maxOutcome);
  return (samples || []).filter((sample) => (
    (!sides.length || sides.includes(sample?.side))
    && (!outcomeSources.length || outcomeSources.includes(sample?.outcomeSource))
    && (minOutcome === null || finiteNumber(sample?.outcome) >= minOutcome)
    && (maxOutcome === null || finiteNumber(sample?.outcome) <= maxOutcome)
  ));
}

function curriculumFilters(options) {
  return {
    sides: normalizeSides(options.sides ?? options.side),
    outcomeSources: normalizeStrings(options.outcomeSources ?? options.outcomeSource),
    minOutcome: optionalFinite(options.minOutcome),
    maxOutcome: optionalFinite(options.maxOutcome),
  };
}

export function normalizeAlphaSelfPlayCurriculum(value) {
  if (!value || value.schema !== ALPHA_SELF_PLAY_CURRICULUM_SCHEMA || !Array.isArray(value.positions)) return null;
  const positions = value.positions
    .filter((position) => position?.initialState && typeof position.initialState === "object" && !Array.isArray(position.initialState))
    .map((position, index) => ({
      index,
      stateHash: position.stateHash || null,
      side: position.side || null,
      phaseId: position.phaseId || null,
      turn: finiteOrNull(position.turn),
      selectedActionType: position.selectedActionType || "unknown-action",
      source: position.source || null,
      outcome: finiteOrNull(position.outcome),
      outcomeSource: typeof position.outcomeSource === "string" ? position.outcomeSource : null,
      outcomeWeight: finiteOrNull(position.outcomeWeight),
      initialState: cloneJsonLike(position.initialState),
    }));
  if (!positions.length || positions.length !== value.positions.length) return null;
  return {
    schema: ALPHA_SELF_PLAY_CURRICULUM_SCHEMA,
    generatedAt: typeof value.generatedAt === "string" ? value.generatedAt : new Date().toISOString(),
    eligiblePositions: Math.max(positions.length, nonNegativeInteger(value.eligiblePositions, positions.length)),
    selectedPositions: positions.length,
    maxPositions: Math.max(positions.length, positiveInteger(value.maxPositions, positions.length)),
    balanceBy: value.balanceBy || "sidePhaseAction",
    priorityBy: value.priorityBy || "policyEntropy",
    coverage: curriculumCoverage(positions),
    positions,
  };
}

export function filterAlphaSelfPlayCurriculum(value, options = {}) {
  const normalized = normalizeAlphaSelfPlayCurriculum(value);
  if (!normalized) return null;
  const hasSideFilter = options.sides !== undefined || options.side !== undefined;
  const hasPhaseFilter = options.phases !== undefined || options.phase !== undefined;
  const sides = normalizeSides(options.sides || options.side);
  const phases = normalizeStrings(options.phases || options.phase);
  if ((hasSideFilter && !sides.length) || (hasPhaseFilter && !phases.length)) return null;
  const filtered = normalized.positions.filter((position) => (
    (!sides.length || sides.includes(position.side))
    && (!phases.length || phases.includes(position.phaseId))
  ));
  const maxPositions = Math.max(1, Math.floor(Number(options.maxPositions || filtered.length || 1)));
  const positions = filtered.slice(0, maxPositions);
  if (!positions.length) return null;
  return normalizeAlphaSelfPlayCurriculum({
    ...normalized,
    eligiblePositions: filtered.length,
    maxPositions,
    positions,
  });
}

export function alphaSelfPlayCurriculumInitialStates(curriculum) {
  const normalized = normalizeAlphaSelfPlayCurriculum(curriculum);
  return normalized ? normalized.positions.map((position) => cloneJsonLike(position.initialState)) : [];
}

export function summarizeAlphaSelfPlayCurriculum(curriculum) {
  const normalized = normalizeAlphaSelfPlayCurriculum(curriculum);
  if (!normalized) return null;
  return {
    schema: ALPHA_SELF_PLAY_CURRICULUM_SCHEMA,
    eligiblePositions: normalized.eligiblePositions,
    selectedPositions: normalized.selectedPositions,
    balanceBy: normalized.balanceBy,
    priorityBy: normalized.priorityBy,
    coverage: normalized.coverage,
  };
}

export function partitionAlphaSelfPlayCurriculum(curriculum, options = {}) {
  const normalized = normalizeAlphaSelfPlayCurriculum(curriculum);
  if (!normalized) return null;
  const requestedEvaluationPositions = nonNegativeInteger(options.evaluationPositions, 0);
  const evaluationCount = Math.min(
    requestedEvaluationPositions,
    Math.max(0, normalized.positions.length - 1),
  );
  const splitIndex = normalized.positions.length - evaluationCount;
  return {
    schema: "zizi-el-alamein-alpha-self-play-curriculum-partition-v1",
    training: curriculumFromPositions(normalized, normalized.positions.slice(0, splitIndex), "training"),
    evaluation: evaluationCount
      ? curriculumFromPositions(normalized, normalized.positions.slice(splitIndex), "evaluation")
      : null,
    overlapStateHashes: curriculumOverlapCount(
      normalized.positions.slice(0, splitIndex),
      normalized.positions.slice(splitIndex),
    ),
  };
}

function curriculumPosition(sample, index) {
  return {
    index,
    stateHash: sample.stateHash || null,
    side: sample.side || null,
    phaseId: sample.phaseId || null,
    turn: finiteOrNull(sample.turn),
    selectedActionType: sample?.decision?.selectedAction?.type || sample?.policy?.[0]?.action?.type || "unknown-action",
    source: sample?.replay?.source || sample.__source || null,
    outcome: finiteOrNull(sample.outcome),
    outcomeSource: typeof sample.outcomeSource === "string" ? sample.outcomeSource : null,
    outcomeWeight: finiteOrNull(sample.outcomeWeight),
    initialState: cloneJsonLike(sample.initialState),
  };
}

function curriculumFromPositions(source, positions, partition) {
  const cloned = positions.map((position, index) => ({
    ...cloneJsonLike(position),
    index,
  }));
  return {
    schema: ALPHA_SELF_PLAY_CURRICULUM_SCHEMA,
    generatedAt: source.generatedAt,
    eligiblePositions: source.eligiblePositions,
    selectedPositions: cloned.length,
    maxPositions: cloned.length,
    balanceBy: source.balanceBy,
    priorityBy: source.priorityBy,
    partition,
    coverage: curriculumCoverage(cloned),
    positions: cloned,
  };
}

function curriculumOverlapCount(trainingPositions, evaluationPositions) {
  const trainingHashes = new Set(trainingPositions.map((position) => position.stateHash).filter(Boolean));
  return new Set(
    evaluationPositions
      .map((position) => position.stateHash)
      .filter((hash) => hash && trainingHashes.has(hash)),
  ).size;
}

function selectHierarchicalCurriculumSamples(samples, options) {
  const phaseBuckets = new Map();
  for (const [index, sample] of (samples || []).entries()) {
    const phaseKey = `${sample.side || "unknown-side"}|${sample.phaseId || "unknown-phase"}`;
    if (!phaseBuckets.has(phaseKey)) {
      phaseBuckets.set(phaseKey, {
        side: sample.side || "unknown-side",
        phaseId: sample.phaseId || "unknown-phase",
        actionGroups: new Map(),
        actionCursor: 0,
      });
    }
    const bucket = phaseBuckets.get(phaseKey);
    const actionType = selectedActionType(sample);
    if (!bucket.actionGroups.has(actionType)) bucket.actionGroups.set(actionType, []);
    bucket.actionGroups.get(actionType).push({
      sample,
      index,
      priority: curriculumSamplePriority(sample, options.priorityBy),
    });
  }
  const buckets = [...phaseBuckets.values()].sort(comparePhaseBuckets);
  for (const bucket of buckets) {
    bucket.actionKeys = [...bucket.actionGroups.keys()].sort();
    for (const group of bucket.actionGroups.values()) {
      group.sort((left, right) => (
        right.priority - left.priority
        || (options.priorityBy === "phaseStart" ? left.index - right.index : right.index - left.index)
      ));
    }
  }

  const selected = [];
  while (selected.length < options.maxPositions) {
    let progressed = false;
    for (const bucket of buckets) {
      const entry = takeCurriculumEntry(bucket);
      if (!entry) continue;
      selected.push(entry.sample);
      progressed = true;
      if (selected.length >= options.maxPositions) break;
    }
    if (!progressed) break;
  }
  return selected;
}

function takeCurriculumEntry(bucket) {
  const keys = bucket.actionKeys || [];
  for (let offset = 0; offset < keys.length; offset += 1) {
    const actionIndex = (bucket.actionCursor + offset) % keys.length;
    const group = bucket.actionGroups.get(keys[actionIndex]);
    if (!group?.length) continue;
    bucket.actionCursor = (actionIndex + 1) % keys.length;
    return group.shift();
  }
  return null;
}

function comparePhaseBuckets(left, right) {
  return sideOrder(left.side) - sideOrder(right.side)
    || phaseOrder(left.phaseId) - phaseOrder(right.phaseId)
    || left.phaseId.localeCompare(right.phaseId);
}

function sideOrder(side) {
  if (side === "axis") return 0;
  if (side === "allied") return 1;
  return 2;
}

function normalizeSides(value) {
  return normalizeStrings(value).filter((side) => side === "axis" || side === "allied");
}

function normalizeStrings(value) {
  const values = Array.isArray(value) ? value : String(value || "").split(",");
  return [...new Set(values.map((entry) => String(entry).trim()).filter(Boolean))].sort();
}

function phaseOrder(phaseId) {
  if (String(phaseId).endsWith("-move")) return 0;
  if (String(phaseId).endsWith("-combat")) return 1;
  return 2;
}

function selectedActionType(sample) {
  return sample?.decision?.selectedAction?.type || sample?.policy?.[0]?.action?.type || "unknown-action";
}

function curriculumSamplePriority(sample, mode) {
  if (mode === "policyEntropy") return finiteNumber(sample?.decision?.policyEntropy);
  if (mode === "surprise") {
    return Math.abs(finiteNumber(sample?.outcome) - finiteNumber(sample?.rootValue))
      * Math.max(0, finiteNumber(sample?.outcomeWeight, 1));
  }
  if (mode === "uncertainty") {
    const confidence = Number(sample?.decision?.recommendationConfidence);
    return Number.isFinite(confidence) ? 1 - Math.min(1, Math.max(0, confidence)) : 0;
  }
  if (mode === "phaseStart") return phaseStartPriority(sample);
  return 0;
}

function phaseStartPriority(sample) {
  const state = sample?.initialState || {};
  const side = sample?.side;
  const units = (state.units || []).filter((unit) => !unit?.eliminated && unit?.side === side);
  const capacity = Math.max(1, units.length);
  const phaseId = String(sample?.phaseId || "");
  let consumed = 0;
  if (phaseId.endsWith("-move")) {
    consumed = new Set(state.movedUnits || []).size;
  } else if (phaseId.endsWith("-combat")) {
    consumed = new Set([
      ...(state.usedAttackers || []),
      ...(state.usedDefenders || []),
    ]).size + (state.declaredCombats || []).length;
  }
  if (state.retreatTask || state.advanceTask) consumed += capacity;
  return Math.max(0, 1 - consumed / capacity);
}

function curriculumCoverage(positions) {
  return {
    sides: countBy(positions, (position) => position.side || "unknown"),
    phases: countBy(positions, (position) => position.phaseId || "unknown"),
    actionTypes: countBy(positions, (position) => position.selectedActionType || "unknown-action"),
    outcomeSources: countBy(positions, (position) => position.outcomeSource || "unknown"),
    turns: countBy(positions, (position) => String(position.turn ?? "unknown")),
  };
}

function countBy(items, selector) {
  return (items || []).reduce((counts, item) => {
    const key = selector(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function cloneJsonLike(value) {
  return value === null || value === undefined ? value : JSON.parse(JSON.stringify(value));
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function optionalFinite(value) {
  if (value === null || value === undefined || value === "") return null;
  return finiteOrNull(value);
}

function finiteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function positiveInteger(value, fallback) {
  const next = Math.floor(Number(value));
  return Number.isFinite(next) && next > 0 ? next : fallback;
}

function nonNegativeInteger(value, fallback) {
  const next = Math.floor(Number(value));
  return Number.isFinite(next) && next >= 0 ? next : fallback;
}
