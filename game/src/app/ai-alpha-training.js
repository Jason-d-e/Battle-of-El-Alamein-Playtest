import { DEFAULT_SITUATION_WEIGHTS } from "./ai-situation.js";
import {
  ALPHA_MODEL_SCHEMA,
  ALPHA_TRAINING_DATA_SUMMARY_SCHEMA,
  ALPHA_TRAINING_METADATA_SCHEMA,
  alphaModelFeatureContract,
  alphaModelEnvironmentFingerprint,
  alphaModelMetadata,
} from "./ai-alpha-model.js";

export const VALUE_FEATURE_SCALES = Object.freeze({
  materialBalance: 80,
  unitBalance: 30,
  axisObjectiveHeld: 1,
  axisObjectiveProgress: 18,
  axisObjectivePressure: 8,
  axisObjectiveLocalAdvantage: 50,
  alliedExitPressure: 8,
  alliedExitLocalAdvantage: 50,
  friendlyCohesion: 100,
  enemyCohesion: 100,
  friendlyThreats: 80,
  enemyThreats: 80,
});

export const POLICY_FEATURE_KEYS = Object.freeze([
  "bias",
  "axisSide",
  "alliedSide",
  "turnProgress",
  "movementPhase",
  "combatPhase",
  "retreatPhase",
  "advancePhase",
  "stateMaterialBalance",
  "stateUnitBalance",
  "stateAxisObjectiveHeld",
  "stateAxisObjectiveProgress",
  "stateAxisObjectivePressure",
  "stateAxisObjectiveLocalAdvantage",
  "stateAlliedExitPressure",
  "stateAlliedExitLocalAdvantage",
  "stateFriendlyCohesion",
  "stateEnemyCohesion",
  "stateFriendlyThreats",
  "stateEnemyThreats",
  "move",
  "declareCombat",
  "finishDeclarations",
  "endPhase",
  "retreat",
  "advance",
  "skipAdvance",
  "routeRemaining",
  "routeSpent",
  "attackerCount",
  "axisObjectiveDestination",
  "alliedExitDestination",
  "moveWithAxisObjectivePressure",
  "moveWithAxisObjectiveProgress",
  "moveWithAxisObjectiveLocalAdvantage",
  "combatWithFriendlyThreats",
  "combatWithEnemyThreats",
  "combatWithAxisObjectiveLocalAdvantage",
  "retreatWithEnemyThreats",
  "advanceWithAxisObjectivePressure",
  "advanceWithAxisObjectiveLocalAdvantage",
  "skipAdvanceWithEnemyThreats",
  "endPhaseWithAxisObjectivePressure",
  "moveWithAlliedExitLocalAdvantage",
  "endPhaseWithAlliedExitLocalAdvantage",
  "endPhaseWithEnemyThreats",
]);

const VALUE_FEATURE_KEYS = Object.freeze(Object.keys(VALUE_FEATURE_SCALES));
const ALPHA_TRAINING_VALIDATION_SCHEMA = "zizi-el-alamein-alpha-training-validation-v1";
const DEFAULT_ITERATIONS = 80;
const DEFAULT_VALUE_LEARNING_RATE = 0.035;
const DEFAULT_POLICY_LEARNING_RATE = 0.08;
const DEFAULT_POLICY_REFERENCE_VISITS = 64;
const MAX_WEIGHT = 8;

export function flattenAlphaSelfPlaySamples(logs) {
  return (logs || []).flatMap((log, fileIndex) => {
    const source = log?.source || log?.path || `input-${fileIndex + 1}`;
    if (Array.isArray(log)) {
      return log.map((sample, sampleIndex) => annotateSample(sample, source, sampleIndex));
    }
    if (Array.isArray(log?.results)) {
      return log.results.flatMap((result, resultIndex) => (
        (result.samples || []).map((sample, sampleIndex) => annotateSample(sample, source, sampleIndex, resultIndex))
      ));
    }
    if (Array.isArray(log?.samples)) {
      return log.samples.map((sample, sampleIndex) => annotateSample(sample, source, sampleIndex));
    }
    if (log?.schema === "zizi-el-alamein-alpha-training-sample-v1") return [annotateSample(log, source, 0)];
    return [];
  });
}

export function trainAlphaModelFromSelfPlay(logs, options = {}) {
  const samples = flattenAlphaSelfPlaySamples(logs);
  return trainAlphaModelFromSamples(samples, options);
}

export function trainAlphaModelFromSamples(samples, options = {}) {
  const inputSamples = Array.isArray(samples) ? samples : [];
  const split = splitAlphaTrainingSamples(inputSamples, options.validation || options);
  const trainingSamples = split.trainingSamples;
  const validationSamples = split.validationSamples;
  const value = trainAlphaValueModel(trainingSamples, options.value || options);
  const policy = trainAlphaPolicyModel(trainingSamples, {
    scenario: options.scenario,
    ...(options.policy || options),
  });
  const validation = validationSamples.length
    ? evaluateAlphaTrainingValidation(validationSamples, {
      value,
      policy,
      scenario: options.scenario,
      split,
    })
    : null;
  return makeAlphaModelArtifact({
    value,
    policy,
    sampleCount: inputSamples.length,
    sources: [...new Set(inputSamples.map((sample) => sample.__source).filter(Boolean))],
    environment: options.environment || alphaModelEnvironmentFingerprint({
      scenario: options.scenario,
      rules: options.rules,
    }),
    featureContract: options.featureContract || alphaTrainingFeatureContract(),
    training: makeAlphaTrainingMetadata(options.parentModel, trainingSamples, validation),
  });
}

export function alphaTrainingFeatureContract() {
  return alphaModelFeatureContract({
    valueFeatureScales: VALUE_FEATURE_SCALES,
    policyFeatureKeys: POLICY_FEATURE_KEYS,
  });
}

export function trainAlphaValueModel(samples, options = {}) {
  const usable = (samples || []).filter((sample) => sample?.features && Number.isFinite(Number(sample.outcome)));
  const weights = initializeValueWeights(options.baseWeights);
  const learningRate = Number(options.learningRate || DEFAULT_VALUE_LEARNING_RATE);
  const iterations = Math.max(1, Number(options.iterations || DEFAULT_ITERATIONS));

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (const sample of usable) {
      const target = clamp(Number(sample.outcome), -1, 1);
      const prediction = scoreAlphaValueSample(sample, { weights });
      const error = prediction - target;
      const slope = 1 - prediction * prediction;
      const sampleWeight = valueSampleWeight(sample);
      const signed = signedValueFeatures(sample);
      for (const key of VALUE_FEATURE_KEYS) {
        const featureValue = normalizeValueFeature(key, signed[key]);
        weights[key] = clampWeight(weights[key] - learningRate * sampleWeight * error * slope * featureValue);
      }
    }
  }

  const model = {
    schema: "zizi-el-alamein-alpha-value-model-v1",
    featureKeys: VALUE_FEATURE_KEYS.slice(),
    featureScales: { ...VALUE_FEATURE_SCALES },
    weights,
  };
  return {
    ...model,
    metrics: evaluateAlphaValueModel(usable, model),
  };
}

export function evaluateAlphaValueModel(samples, model = {}) {
  const usable = (samples || []).filter((sample) => sample?.features && Number.isFinite(Number(sample.outcome)));
  if (!usable.length) {
    return { samples: 0, mse: null, mae: null, signAccuracy: null };
  }
  let squared = 0;
  let absolute = 0;
  let weightedSquared = 0;
  let weightedAbsolute = 0;
  let weightSum = 0;
  let signMatches = 0;
  for (const sample of usable) {
    const target = clamp(Number(sample.outcome), -1, 1);
    const prediction = scoreAlphaValueSample(sample, model);
    const error = prediction - target;
    const sampleWeight = valueSampleWeight(sample);
    squared += error * error;
    absolute += Math.abs(error);
    weightedSquared += sampleWeight * error * error;
    weightedAbsolute += sampleWeight * Math.abs(error);
    weightSum += sampleWeight;
    if (Math.sign(prediction) === Math.sign(target) || target === 0) signMatches += 1;
  }
  return {
    samples: usable.length,
    mse: round(squared / usable.length, 6),
    mae: round(absolute / usable.length, 6),
    weightedMse: weightSum > 0 ? round(weightedSquared / weightSum, 6) : null,
    weightedMae: weightSum > 0 ? round(weightedAbsolute / weightSum, 6) : null,
    averageWeight: round(weightSum / usable.length, 4),
    signAccuracy: round(signMatches / usable.length, 4),
  };
}

export function scoreAlphaValueSample(sample, model = {}) {
  const weights = model.weights || initializeValueWeights(model.baseWeights);
  const featureScales = {
    ...VALUE_FEATURE_SCALES,
    ...(model.featureScales || {}),
  };
  const featureKeys = Array.isArray(model.featureKeys) && model.featureKeys.length
    ? model.featureKeys.map(String)
    : VALUE_FEATURE_KEYS;
  const signed = signedValueFeatures(sample);
  let score = 0;
  for (const key of featureKeys) {
    score += Number(weights[key] || 0) * normalizeValueFeature(key, signed[key], featureScales);
  }
  return Math.tanh(score);
}

export function trainAlphaPolicyModel(samples, options = {}) {
  const rows = policyRowsFromSamples(samples, options);
  const groups = groupPolicyRows(rows);
  const weights = initializePolicyWeights(options.baseWeights);
  const learningRate = Number(options.learningRate || DEFAULT_POLICY_LEARNING_RATE);
  const iterations = Math.max(1, Number(options.iterations || DEFAULT_ITERATIONS));

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (const group of groups) {
      const groupWeight = policyGroupWeight(group);
      const probabilities = softmax(group.map((row) => scoreAlphaPolicyLogit(row.features, { weights })));
      const targets = normalizePolicyTargets(group);
      for (let rowIndex = 0; rowIndex < group.length; rowIndex += 1) {
        const row = group[rowIndex];
        const error = (probabilities[rowIndex] - targets[rowIndex]) * groupWeight;
        for (const key of POLICY_FEATURE_KEYS) {
          weights[key] = clampWeight(weights[key] - learningRate * error * Number(row.features[key] || 0));
        }
      }
    }
  }

  const model = {
    schema: "zizi-el-alamein-alpha-policy-model-v1",
    featureKeys: POLICY_FEATURE_KEYS.slice(),
    weights,
  };
  return {
    ...model,
    metrics: evaluateAlphaPolicyModel(rows, model),
  };
}

export function evaluateAlphaPolicyModel(rowsOrSamples, model = {}) {
  const rows = rowsOrSamples?.[0]?.features && rowsOrSamples?.[0]?.target !== undefined
    ? rowsOrSamples
    : policyRowsFromSamples(rowsOrSamples || [], model);
  if (!rows.length) {
    return {
      rows: 0,
      mse: null,
      crossEntropy: null,
      weightedMse: null,
      weightedCrossEntropy: null,
      averageWeight: 0,
      topChoiceAccuracy: null,
    };
  }
  let squared = 0;
  let crossEntropy = 0;
  let weightedSquared = 0;
  let weightedCrossEntropy = 0;
  let weightSum = 0;
  let topMatches = 0;
  let groups = 0;
  for (const group of groupPolicyRows(rows)) {
    groups += 1;
    const groupWeight = policyGroupWeight(group);
    const probabilities = softmax(group.map((row) => scoreAlphaPolicyLogit(row.features, model)));
    const targets = normalizePolicyTargets(group);
    for (let index = 0; index < group.length; index += 1) {
      const error = probabilities[index] - targets[index];
      const rowCrossEntropy = -targets[index] * Math.log(Math.max(1e-12, probabilities[index]));
      squared += error * error;
      crossEntropy += rowCrossEntropy;
      weightedSquared += groupWeight * error * error;
      weightedCrossEntropy += groupWeight * rowCrossEntropy;
    }
    weightSum += groupWeight * group.length;
    const predicted = group
      .map((row, index) => ({ ...row, prediction: probabilities[index] }))
      .sort((a, b) => b.prediction - a.prediction || a.actionKey.localeCompare(b.actionKey))[0];
    const target = group
      .slice()
      .sort((a, b) => b.target - a.target || a.actionKey.localeCompare(b.actionKey))[0];
    if (predicted?.actionKey === target?.actionKey) topMatches += 1;
  }

  return {
    rows: rows.length,
    mse: round(squared / rows.length, 6),
    crossEntropy: round(crossEntropy / Math.max(1, groups), 6),
    weightedMse: weightSum > 0 ? round(weightedSquared / weightSum, 6) : null,
    weightedCrossEntropy: weightSum > 0 ? round(weightedCrossEntropy / Math.max(1, groups), 6) : null,
    averageWeight: round(weightSum / rows.length, 4),
    topChoiceAccuracy: groups ? round(topMatches / groups, 4) : null,
  };
}

export function scoreAlphaPolicyFeatures(features, model = {}) {
  return sigmoid(scoreAlphaPolicyLogit(features, model));
}

export function scoreAlphaPolicyLogit(features, model = {}) {
  const weights = model.weights || initializePolicyWeights(model.baseWeights);
  const featureKeys = Array.isArray(model.featureKeys) && model.featureKeys.length
    ? model.featureKeys.map(String)
    : POLICY_FEATURE_KEYS;
  let score = 0;
  for (const key of featureKeys) score += Number(weights[key] || 0) * Number(features[key] || 0);
  return score;
}

export function policyRowsFromSamples(samples, options = {}) {
  return (samples || []).flatMap((sample, sampleIndex) => (
    (sample.policy || []).map((entry) => ({
      stateHash: sample.stateHash || null,
      groupKey: sample.stateHash || `sample-${sampleIndex}`,
      side: sample.side || null,
      action: entry.action,
      actionKey: stableActionKey(entry.action),
      target: clamp(Number(entry.visitShare || 0), 0, 1),
      sampleWeight: policySampleWeight(sample, options),
      features: actionPolicyFeatures(entry.action, sample, options.scenario),
    }))
  ));
}

export function splitAlphaTrainingSamples(samples = [], options = {}) {
  const rows = Array.isArray(samples) ? samples.slice() : [];
  const validationFraction = clamp(Number(options.validationFraction ?? options.holdoutFraction ?? 0), 0, 0.9);
  const minValidationSamples = Math.max(0, Math.floor(Number(options.minValidationSamples ?? 1)));
  const seed = String(options.validationSeed ?? options.seed ?? "alpha-training-validation-v1");
  const validationGroupBy = normalizeValidationGroupBy(options.validationGroupBy ?? options.holdoutGroupBy);
  if (!rows.length || !(validationFraction > 0)) {
    return makeTrainingSplit({ seed, validationFraction, validationGroupBy, trainingSamples: rows, validationSamples: [] });
  }
  const validationCount = Math.min(
    Math.max(1, minValidationSamples, Math.round(rows.length * validationFraction)),
    Math.max(0, rows.length - 1),
  );
  if (!(validationCount > 0)) {
    return makeTrainingSplit({ seed, validationFraction, validationGroupBy, trainingSamples: rows, validationSamples: [] });
  }
  if (validationGroupBy !== "sample") {
    return splitAlphaTrainingSamplesByGroup(rows, {
      seed,
      validationFraction,
      validationCount,
      validationGroupBy,
    });
  }
  const ranked = rows
    .map((sample, index) => ({
      sample,
      index,
      key: stableTrainingSplitKey(sample, index, seed),
    }))
    .sort((left, right) => left.key.localeCompare(right.key) || left.index - right.index);
  const validationIndexes = new Set(ranked.slice(0, validationCount).map((entry) => entry.index));
  return makeTrainingSplit({
    seed,
    validationFraction,
    validationGroupBy,
    trainingSamples: rows.filter((sample, index) => !validationIndexes.has(index)),
    validationSamples: rows.filter((sample, index) => validationIndexes.has(index)),
  });
}

export function evaluateAlphaTrainingValidation(samples = [], options = {}) {
  const rows = Array.isArray(samples) ? samples : [];
  return {
    schema: ALPHA_TRAINING_VALIDATION_SCHEMA,
    seed: options.split?.seed || null,
    validationFraction: Number(options.split?.validationFraction || 0),
    validationGroupBy: options.split?.validationGroupBy || "sample",
    trainingGroups: Number(options.split?.trainingGroupCount || 0),
    validationGroups: Number(options.split?.validationGroupCount || 0),
    trainingSamples: Number(options.split?.trainingCount || 0),
    sampleCount: rows.length,
    sides: countByString(rows, (sample) => sample?.side),
    phases: countByString(rows, (sample) => sample?.phaseId),
    outcomeSources: countByString(rows, (sample) => sample?.outcomeSource),
    value: options.value ? evaluateAlphaValueModel(rows, options.value) : null,
    policy: options.policy ? evaluateAlphaPolicyModel(rows, {
      ...options.policy,
      scenario: options.scenario,
    }) : null,
  };
}

export function actionPolicyFeatures(action, sample = {}, scenario = null) {
  const objectives = scenario?.objectives || {};
  const axisTargets = new Set([
    ...(objectives.alamHalfaRidge || []),
    ...(objectives.coastalRoadEast || []),
  ]);
  const alliedExit = new Set(objectives.alliedWestExitEdge || []);
  const stateFeatures = sample.features || {};
  const phaseId = sample.phaseId || "";
  const isMove = action?.type === "MOVE_UNIT";
  const isCombat = action?.type === "DECLARE_COMBAT";
  const isRetreat = action?.type === "RETREAT_UNIT";
  const isAdvance = action?.type === "ADVANCE_UNIT";
  const isSkipAdvance = action?.type === "SKIP_ADVANCE";
  const isEndPhase = action?.type === "END_PHASE";
  const axisObjectiveProgress = normalizePolicyStateFeature("axisObjectiveProgress", stateFeatures.axisObjectiveProgress);
  const axisObjectivePressure = normalizePolicyStateFeature("axisObjectivePressure", stateFeatures.axisObjectivePressure);
  const axisObjectiveLocalAdvantage = normalizePolicyStateFeature(
    "axisObjectiveLocalAdvantage",
    stateFeatures.axisObjectiveLocalAdvantage,
  );
  const alliedExitLocalAdvantage = normalizePolicyStateFeature(
    "alliedExitLocalAdvantage",
    stateFeatures.alliedExitLocalAdvantage,
  );
  const friendlyThreats = normalizePolicyStateFeature("friendlyThreats", stateFeatures.friendlyThreats);
  const enemyThreats = normalizePolicyStateFeature("enemyThreats", stateFeatures.enemyThreats);
  return {
    bias: 1,
    axisSide: sample.side === "axis" ? 1 : 0,
    alliedSide: sample.side === "allied" ? 1 : 0,
    turnProgress: clamp(Number(sample.turn || 0) / 12, 0, 1),
    movementPhase: phaseId.includes("movement") ? 1 : 0,
    combatPhase: phaseId.includes("combat") ? 1 : 0,
    retreatPhase: phaseId.includes("retreat") ? 1 : 0,
    advancePhase: phaseId.includes("advance") ? 1 : 0,
    stateMaterialBalance: normalizePolicyStateFeature("materialBalance", stateFeatures.materialBalance),
    stateUnitBalance: normalizePolicyStateFeature("unitBalance", stateFeatures.unitBalance),
    stateAxisObjectiveHeld: clamp(Number(stateFeatures.axisObjectiveHeld || 0), 0, 1),
    stateAxisObjectiveProgress: axisObjectiveProgress,
    stateAxisObjectivePressure: axisObjectivePressure,
    stateAxisObjectiveLocalAdvantage: axisObjectiveLocalAdvantage,
    stateAlliedExitPressure: normalizePolicyStateFeature("alliedExitPressure", stateFeatures.alliedExitPressure),
    stateAlliedExitLocalAdvantage: alliedExitLocalAdvantage,
    stateFriendlyCohesion: normalizePolicyStateFeature("friendlyCohesion", stateFeatures.friendlyCohesion),
    stateEnemyCohesion: normalizePolicyStateFeature("enemyCohesion", stateFeatures.enemyCohesion),
    stateFriendlyThreats: friendlyThreats,
    stateEnemyThreats: enemyThreats,
    move: isMove ? 1 : 0,
    declareCombat: isCombat ? 1 : 0,
    finishDeclarations: action?.type === "FINISH_DECLARATIONS" ? 1 : 0,
    endPhase: isEndPhase ? 1 : 0,
    retreat: isRetreat ? 1 : 0,
    advance: isAdvance ? 1 : 0,
    skipAdvance: isSkipAdvance ? 1 : 0,
    routeRemaining: clamp(Number(action?.route?.remaining || 0) / 10, 0, 1),
    routeSpent: clamp(Number(action?.route?.spent || 0) / 10, 0, 1),
    attackerCount: clamp(Number(action?.attackerIds?.length || 0) / 6, 0, 1),
    axisObjectiveDestination: axisTargets.has(action?.toHexId || action?.targetHexId) ? 1 : 0,
    alliedExitDestination: alliedExit.has(action?.toHexId || action?.targetHexId) ? 1 : 0,
    moveWithAxisObjectivePressure: isMove ? axisObjectivePressure : 0,
    moveWithAxisObjectiveProgress: isMove ? axisObjectiveProgress : 0,
    moveWithAxisObjectiveLocalAdvantage: isMove ? axisObjectiveLocalAdvantage : 0,
    combatWithFriendlyThreats: isCombat ? friendlyThreats : 0,
    combatWithEnemyThreats: isCombat ? enemyThreats : 0,
    combatWithAxisObjectiveLocalAdvantage: isCombat ? axisObjectiveLocalAdvantage : 0,
    retreatWithEnemyThreats: isRetreat ? enemyThreats : 0,
    advanceWithAxisObjectivePressure: isAdvance ? axisObjectivePressure : 0,
    advanceWithAxisObjectiveLocalAdvantage: isAdvance ? axisObjectiveLocalAdvantage : 0,
    skipAdvanceWithEnemyThreats: isSkipAdvance ? enemyThreats : 0,
    endPhaseWithAxisObjectivePressure: isEndPhase ? axisObjectivePressure : 0,
    moveWithAlliedExitLocalAdvantage: isMove ? alliedExitLocalAdvantage : 0,
    endPhaseWithAlliedExitLocalAdvantage: isEndPhase ? alliedExitLocalAdvantage : 0,
    endPhaseWithEnemyThreats: isEndPhase ? enemyThreats : 0,
  };
}

function normalizePolicyStateFeature(key, value) {
  return clamp(Number(value || 0) / Number(VALUE_FEATURE_SCALES[key] || 1), -1, 1);
}

export function makeAlphaModelArtifact({
  value,
  policy,
  sampleCount = 0,
  sources = [],
  environment = null,
  featureContract = null,
  training = null,
} = {}) {
  return {
    schema: ALPHA_MODEL_SCHEMA,
    generatedAt: new Date().toISOString(),
    method: "linear-policy-value-self-play",
    sampleCount,
    sources,
    environment,
    featureContract,
    training: training || makeAlphaTrainingMetadata(null),
    value,
    policy,
  };
}

export function makeAlphaTrainingMetadata(parentModel = null, samples = [], validation = null) {
  const parent = compactParentModelMetadata(parentModel);
  return {
    schema: ALPHA_TRAINING_METADATA_SCHEMA,
    warmStarted: Boolean(parent),
    parent,
    data: summarizeAlphaTrainingData(samples),
    validation,
  };
}

export function summarizeAlphaTrainingData(samples = []) {
  const rows = Array.isArray(samples) ? samples : [];
  return {
    schema: ALPHA_TRAINING_DATA_SUMMARY_SCHEMA,
    sampleCount: rows.length,
    sources: uniqueStrings(rows.map(trainingSampleSource)),
    sides: countByString(rows, (sample) => sample?.side),
    phases: countByString(rows, (sample) => sample?.phaseId),
    outcomeSources: countByString(rows, (sample) => sample?.outcomeSource),
    selectionModes: countByString(rows, (sample) => sample?.decision?.selectionMode, { missing: null }),
    samplesWithPolicy: rows.filter((sample) => Array.isArray(sample?.policy) && sample.policy.length > 0).length,
    samplesWithDecision: rows.filter((sample) => sample?.decision && typeof sample.decision === "object").length,
    samplesWithStateSnapshot: rows.filter((sample) => Boolean(sample?.initialState || sample?.stateSnapshot || sample?.stateBefore)).length,
    reanalysisSamples: rows.filter((sample) => Boolean(sample?.reanalysis) || sample?.decision?.selectionMode === "reanalyzed").length,
    averageOutcomeWeight: averageFinite(rows.map((sample) => sample?.outcomeWeight ?? 1), 4),
    averageRootVisits: averageFinite(rows.map((sample) => sample?.decision?.rootVisits), 4),
    averagePolicyEntropy: averageFinite(rows.map((sample) => sample?.decision?.policyEntropy), 6),
  };
}

function annotateSample(sample, source, sampleIndex, resultIndex = null) {
  return {
    ...sample,
    __source: source,
    __sampleIndex: sampleIndex,
    __resultIndex: resultIndex,
  };
}

function compactParentModelMetadata(parentModel) {
  const metadata = alphaModelMetadata(parentModel);
  if (!metadata) return null;
  return {
    schema: metadata.schema,
    generatedAt: metadata.generatedAt,
    method: metadata.method,
    sampleCount: metadata.sampleCount,
    sources: metadata.sources || [],
    hasValue: metadata.hasValue,
    hasPolicy: metadata.hasPolicy,
    valueWeights: metadata.valueWeights,
    policyWeights: metadata.policyWeights,
    trainingData: metadata.training?.data || null,
    trainingValidation: metadata.training?.validation || null,
    release: metadata.release ? {
      schema: metadata.release.schema,
      sourceSchema: metadata.release.sourceSchema,
      sourceArtifact: metadata.release.sourceArtifact,
      sourceHash: metadata.release.sourceHash,
      activeGeneration: metadata.release.activeGeneration ?? null,
    } : null,
  };
}

function trainingSampleSource(sample) {
  return sample?.__source || sample?.source || sample?.replay?.source || null;
}

function uniqueStrings(values) {
  return [...new Set((values || []).filter((value) => typeof value === "string" && value.length > 0))];
}

function countByString(items, select, { missing = "unknown" } = {}) {
  const counts = {};
  for (const item of items || []) {
    const raw = select(item);
    const key = typeof raw === "string" && raw ? raw : missing;
    if (!key) continue;
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

function averageFinite(values, digits = 4) {
  const finite = (values || []).map(Number).filter(Number.isFinite);
  if (!finite.length) return null;
  return round(finite.reduce((sum, value) => sum + value, 0) / finite.length, digits);
}

function signedValueFeatures(sample) {
  const features = sample.features || {};
  const axisSign = sample.side === "axis" ? 1 : -1;
  return {
    materialBalance: Number(features.materialBalance || 0),
    unitBalance: Number(features.unitBalance || 0),
    axisObjectiveHeld: axisSign * Number(features.axisObjectiveHeld || 0),
    axisObjectiveProgress: axisSign * Number(features.axisObjectiveProgress || 0),
    axisObjectivePressure: axisSign * Number(features.axisObjectivePressure || 0),
    axisObjectiveLocalAdvantage: axisSign * Number(features.axisObjectiveLocalAdvantage || 0),
    alliedExitPressure: -axisSign * Number(features.alliedExitPressure || 0),
    alliedExitLocalAdvantage: -axisSign * Number(features.alliedExitLocalAdvantage || 0),
    friendlyCohesion: Number(features.friendlyCohesion || 0),
    enemyCohesion: Number(features.enemyCohesion || 0),
    friendlyThreats: Number(features.friendlyThreats || 0),
    enemyThreats: Number(features.enemyThreats || 0),
  };
}

function valueSampleWeight(sample) {
  const next = Number(sample?.outcomeWeight ?? 1);
  if (!Number.isFinite(next)) return 1;
  return clamp(next, 0, 1);
}

function policySampleWeight(sample, options = {}) {
  const outcomeWeight = valueSampleWeight(sample);
  const decision = sample?.decision || {};
  if (!decision || typeof decision !== "object" || !Object.keys(decision).length) return outcomeWeight;
  const visits = Math.max(
    0,
    Number(decision.rootVisits || 0),
    Number(decision.searchIterations || 0),
    policyVisitCount(sample.policy),
  );
  const referenceVisits = Math.max(1, Number(options.policyReferenceVisits || DEFAULT_POLICY_REFERENCE_VISITS));
  const visitWeight = visits > 0
    ? clamp(Math.log2(visits + 1) / Math.log2(referenceVisits + 1), 0.1, 1)
    : 0.5;
  const rootNoiseWeight = clamp(Number(decision.rootNoiseWeight || 0), 0, 1);
  const temperature = Math.max(0, Number(decision.temperature || 0));
  const explorationPenalty = clamp(1 - (rootNoiseWeight * 0.35) - (Math.min(1, temperature) * 0.15), 0.35, 1);
  return clamp(outcomeWeight * visitWeight * explorationPenalty, 0, 1);
}

function policyVisitCount(policy) {
  return (policy || []).reduce((sum, entry) => sum + Math.max(0, Number(entry.visits || 0)), 0);
}

function normalizeValueFeature(key, value, featureScales = VALUE_FEATURE_SCALES) {
  return clamp(Number(value || 0) / Number(featureScales[key] || VALUE_FEATURE_SCALES[key] || 1), -1, 1);
}

function initializeValueWeights(baseWeights = null) {
  const weights = {};
  for (const key of VALUE_FEATURE_KEYS) {
    weights[key] = Number(baseWeights?.[key] ?? DEFAULT_SITUATION_WEIGHTS[key] ?? 0);
  }
  return weights;
}

function initializePolicyWeights(baseWeights = null) {
  return POLICY_FEATURE_KEYS.reduce((weights, key) => {
    weights[key] = Number(baseWeights?.[key] || 0);
    return weights;
  }, {});
}

function groupPolicyRows(rows) {
  const grouped = new Map();
  for (const row of rows || []) {
    const key = row.groupKey || row.stateHash || row.actionKey || "__unknown__";
    const group = grouped.get(key) || [];
    group.push(row);
    grouped.set(key, group);
  }
  return [...grouped.values()].filter((group) => group.length);
}

function policyGroupWeight(group) {
  if (!group?.length) return 1;
  const total = group.reduce((sum, row) => sum + clamp(Number(row.sampleWeight ?? 1), 0, 1), 0);
  return total / group.length;
}

function normalizePolicyTargets(group) {
  const raw = group.map((row) => clamp(Number(row.target || 0), 0, 1));
  const total = raw.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return raw.map(() => 1 / raw.length);
  return raw.map((value) => value / total);
}

function softmax(logits) {
  if (!logits.length) return [];
  const max = Math.max(...logits.map((value) => Number(value || 0)));
  const exp = logits.map((value) => Math.exp(Math.max(-40, Math.min(40, Number(value || 0) - max))));
  const total = exp.reduce((sum, value) => sum + value, 0) || 1;
  return exp.map((value) => value / total);
}

function splitAlphaTrainingSamplesByGroup(rows, options) {
  const groups = groupedTrainingSamplesForValidation(rows, options.validationGroupBy);
  if (groups.length < 2) {
    return makeTrainingSplit({
      seed: options.seed,
      validationFraction: options.validationFraction,
      validationGroupBy: options.validationGroupBy,
      trainingSamples: rows,
      validationSamples: [],
    });
  }
  const ranked = groups
    .map((group, index) => ({
      ...group,
      index,
      key: stableHash(`${options.seed}|${options.validationGroupBy}|${group.key}|${index}`),
    }))
    .sort((left, right) => left.key.localeCompare(right.key) || left.index - right.index);
  const validationIndexes = new Set();
  for (const group of ranked) {
    if (validationIndexes.size >= options.validationCount) break;
    if (validationIndexes.size + group.indexes.length >= rows.length) continue;
    for (const index of group.indexes) validationIndexes.add(index);
  }
  return makeTrainingSplit({
    seed: options.seed,
    validationFraction: options.validationFraction,
    validationGroupBy: options.validationGroupBy,
    trainingSamples: rows.filter((sample, index) => !validationIndexes.has(index)),
    validationSamples: rows.filter((sample, index) => validationIndexes.has(index)),
  });
}

function groupedTrainingSamplesForValidation(rows, validationGroupBy) {
  const groups = new Map();
  for (const [index, sample] of rows.entries()) {
    const key = trainingValidationGroupKey(sample, index, validationGroupBy);
    const group = groups.get(key) || { key, indexes: [] };
    group.indexes.push(index);
    groups.set(key, group);
  }
  return [...groups.values()];
}

function trainingValidationGroupKey(sample, index, validationGroupBy) {
  if (validationGroupBy === "stateHash") return sample?.stateHash || `sample-${index}`;
  if (validationGroupBy === "side") return sample?.side || `sample-${index}`;
  if (validationGroupBy === "phase") return sample?.phaseId || `sample-${index}`;
  return `sample-${index}`;
}

function normalizeValidationGroupBy(value) {
  const next = String(value || "sample");
  return ["sample", "stateHash", "side", "phase"].includes(next) ? next : "sample";
}

function makeTrainingSplit({ seed, validationFraction, validationGroupBy = "sample", trainingSamples, validationSamples }) {
  const trainingGroups = groupedTrainingSamplesForValidation(trainingSamples || [], validationGroupBy);
  const validationGroups = groupedTrainingSamplesForValidation(validationSamples || [], validationGroupBy);
  return {
    schema: "zizi-el-alamein-alpha-training-split-v1",
    seed,
    validationFraction,
    validationGroupBy,
    trainingSamples,
    validationSamples,
    trainingCount: trainingSamples.length,
    validationCount: validationSamples.length,
    trainingGroupCount: trainingGroups.length,
    validationGroupCount: validationGroups.length,
  };
}

function stableTrainingSplitKey(sample, index, seed) {
  return stableHash(`${seed}|${sample?.stateHash || ""}|${sample?.side || ""}|${sample?.phaseId || ""}|${sample?.turn ?? ""}|${sample?.__source || ""}|${index}`);
}

function stableHash(text) {
  let hash = 2166136261;
  for (let index = 0; index < String(text).length; index += 1) {
    hash ^= String(text).charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
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

function sigmoid(value) {
  if (value > 40) return 1;
  if (value < -40) return 0;
  return 1 / (1 + Math.exp(-value));
}

function clampWeight(value) {
  return clamp(Number(value || 0), -MAX_WEIGHT, MAX_WEIGHT);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
