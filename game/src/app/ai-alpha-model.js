export const ALPHA_MODEL_SCHEMA = "zizi-el-alamein-alpha-model-v1";
export const ALPHA_VALUE_MODEL_SCHEMA = "zizi-el-alamein-alpha-value-model-v1";
export const ALPHA_POLICY_MODEL_SCHEMA = "zizi-el-alamein-alpha-policy-model-v1";
export const ALPHA_RELEASE_METADATA_SCHEMA = "zizi-el-alamein-alpha-release-metadata-v1";
export const ALPHA_TRAINING_METADATA_SCHEMA = "zizi-el-alamein-alpha-training-metadata-v1";
export const ALPHA_TRAINING_DATA_SUMMARY_SCHEMA = "zizi-el-alamein-alpha-training-data-summary-v1";
export const ALPHA_TRAINING_VALIDATION_SCHEMA = "zizi-el-alamein-alpha-training-validation-v1";
export const ALPHA_ENVIRONMENT_SCHEMA = "zizi-el-alamein-alpha-environment-v1";
export const ALPHA_FEATURE_CONTRACT_SCHEMA = "zizi-el-alamein-alpha-feature-contract-v1";

export function extractAlphaModelArtifact(value) {
  if (!value || typeof value !== "object") return null;
  const direct = normalizeAlphaModelArtifact(value);
  if (direct) return direct;
  for (const key of ["activeModel", "candidateModel", "model", "baseModel"]) {
    const nested = normalizeAlphaModelArtifact(value[key]);
    if (nested) return nested;
  }
  return null;
}

export function extractReleasedAlphaModelArtifact(value, options = {}) {
  const validation = validateReleasedAlphaModelArtifact(value, options);
  return validation.ok ? validation.model : null;
}

export function validateAlphaModelArtifact(value) {
  const model = normalizeAlphaModelArtifact(value);
  if (!model) {
    return {
      ok: false,
      reason: "invalid_alpha_model",
      model: null,
    };
  }
  return {
    ok: true,
    reason: null,
    model,
  };
}

export function validateReleasedAlphaModelArtifact(value, options = {}) {
  const validation = validateAlphaModelArtifact(value);
  if (!validation.ok) return validation;
  const minSuiteGames = Math.max(0, Number(options.minSuiteGames ?? 2));
  const minSuiteSides = Math.max(0, Number(options.minSuiteSides ?? 2));
  const minFixedPositions = Math.max(0, Number(options.minFixedPositions ?? 0));
  const minChallengePositions = Math.max(0, Number(options.minChallengePositions ?? 0));
  const maxErrors = Math.max(0, Number(options.maxErrors ?? 0));
  const requireExplicitSuite = options.requireExplicitSuite !== false;
  const requireSourceFingerprint = options.requireSourceFingerprint !== false;
  const release = validation.model.release;
  const minSideScore = optionalFiniteOrNull(options.minSideScore ?? release?.minSideScore);
  const minScoreLowerBound = optionalFiniteOrNull(options.minScoreLowerBound ?? release?.minScoreLowerBound);
  const minAnalyzedActions = optionalFiniteOrNull(options.minAnalyzedActions ?? release?.minAnalyzedActions);
  const minAverageRootVisits = optionalFiniteOrNull(options.minAverageRootVisits ?? release?.minAverageRootVisits);
  const trainingGate = trainingDataGate(validation.model.training?.data, {
    minTrainingSamples: options.minTrainingSamples ?? release?.minTrainingSamples,
    minTrainingSides: options.minTrainingSides ?? release?.minTrainingSides,
    minTrainingSources: options.minTrainingSources ?? release?.minTrainingSources,
    minTrainingReanalysisSamples: options.minTrainingReanalysisSamples ?? release?.minTrainingReanalysisSamples,
    minTrainingStateSnapshots: options.minTrainingStateSnapshots ?? release?.minTrainingStateSnapshots,
    minTrainingAverageRootVisits: options.minTrainingAverageRootVisits ?? release?.minTrainingAverageRootVisits,
  });
  const validationGate = trainingValidationGate(validation.model.training?.validation, {
    minTrainingValidationSamples: options.minTrainingValidationSamples ?? release?.minTrainingValidationSamples,
    maxTrainingValidationValueMse: options.maxTrainingValidationValueMse ?? release?.maxTrainingValidationValueMse,
    maxTrainingValidationPolicyCrossEntropy: options.maxTrainingValidationPolicyCrossEntropy ?? release?.maxTrainingValidationPolicyCrossEntropy,
  });
  const expectedEnvironment = normalizeAlphaModelEnvironment(options.expectedEnvironment)
    || alphaModelEnvironmentFingerprint({ scenario: options.scenario, rules: options.rules });
  const environmentGate = validateAlphaModelEnvironment(validation.model, expectedEnvironment, {
    requireEnvironmentFingerprint: options.requireEnvironmentFingerprint,
  });
  const expectedFeatureContract = normalizeAlphaModelFeatureContract(options.expectedFeatureContract);
  const featureContractGate = validateAlphaModelFeatureContract(validation.model, expectedFeatureContract, {
    requireFeatureContract: options.requireFeatureContract,
  });
  if (!release) return releaseValidationError("missing_release_metadata", validation.model);
  if (!release.promoted) return releaseValidationError("model_not_promoted", validation.model);
  if (requireSourceFingerprint && !release.sourceArtifact) {
    return releaseValidationError("missing_release_source_artifact", validation.model);
  }
  if (requireSourceFingerprint && !release.sourceHash) {
    return releaseValidationError("missing_release_source_hash", validation.model);
  }
  if (Number(release.errors || 0) > maxErrors) return releaseValidationError("release_errors_exceed_limit", validation.model);
  if (requireExplicitSuite && !release.evaluationSuite?.explicit) {
    return releaseValidationError("evaluation_suite_not_explicit", validation.model);
  }
  if (Number(release.evaluationSuite?.games || 0) < minSuiteGames) {
    return releaseValidationError("evaluation_suite_too_small", validation.model);
  }
  if (Number(release.evaluationSuite?.sides || 0) < minSuiteSides) {
    return releaseValidationError("evaluation_suite_side_coverage_too_narrow", validation.model);
  }
  if (Number(release.evaluationSuite?.fixedPositions || 0) < minFixedPositions) {
    return releaseValidationError("evaluation_suite_fixed_positions_too_few", validation.model);
  }
  if (Number(release.evaluationSuite?.challengePositions || 0) < minChallengePositions) {
    return releaseValidationError("evaluation_suite_challenge_positions_too_few", validation.model);
  }
  if (minSideScore !== null) {
    if (!release.sideScores?.length) return releaseValidationError("missing_side_score_evidence", validation.model);
    const weakSide = release.sideScores.find((score) => (
      Number(score.scoredGames || 0) < 1 || Number(score.candidateScore || 0) < minSideScore
    ));
    if (weakSide) return releaseValidationError("candidate_side_score_below_threshold", validation.model);
  }
  if (minScoreLowerBound !== null) {
    if (!release.arena?.scoreInterval95) return releaseValidationError("missing_arena_evidence", validation.model);
    if (
      release.scoreLowerBoundPass === false
      || Number(release.arena.scoreInterval95.low || 0) < minScoreLowerBound
    ) {
      return releaseValidationError("candidate_score_lower_bound_below_threshold", validation.model);
    }
  }
  if (minAnalyzedActions !== null || minAverageRootVisits !== null) {
    if (!release.decisionEvidence) return releaseValidationError("missing_decision_evidence", validation.model);
    if (minAnalyzedActions !== null && Number(release.decisionEvidence.analyzedActions || 0) < minAnalyzedActions) {
      return releaseValidationError("decision_evidence_too_few_analyzed_actions", validation.model);
    }
    if (minAverageRootVisits !== null && Number(release.decisionEvidence.averageRootVisits || 0) < minAverageRootVisits) {
      return releaseValidationError("decision_evidence_root_visits_too_low", validation.model);
    }
  }
  if (!trainingGate.ok) return releaseValidationError(trainingGate.reason, validation.model);
  if (!validationGate.ok) return releaseValidationError(validationGate.reason, validation.model);
  if (!environmentGate.ok) return releaseValidationError(environmentGate.reason, validation.model, environmentGate);
  if (!featureContractGate.ok) {
    return releaseValidationError(featureContractGate.reason, validation.model, environmentGate, featureContractGate);
  }
  return {
    ...validation,
    environment: environmentGate,
    featureContract: featureContractGate,
  };
}

export function normalizeAlphaModelArtifact(value) {
  if (!value || typeof value !== "object") return null;
  if (value.schema !== ALPHA_MODEL_SCHEMA) return null;
  const valueModel = normalizeSubModel(value.value, ALPHA_VALUE_MODEL_SCHEMA);
  const policyModel = normalizeSubModel(value.policy, ALPHA_POLICY_MODEL_SCHEMA);
  if (!valueModel && !policyModel) return null;
  return {
    schema: ALPHA_MODEL_SCHEMA,
    generatedAt: typeof value.generatedAt === "string" ? value.generatedAt : null,
    method: typeof value.method === "string" ? value.method : "unknown",
    sampleCount: finiteNumber(value.sampleCount, 0),
    sources: Array.isArray(value.sources) ? value.sources.map(String) : [],
    environment: normalizeAlphaModelEnvironment(value.environment),
    featureContract: normalizeAlphaModelFeatureContract(value.featureContract),
    training: normalizeAlphaTrainingMetadata(value.training),
    value: valueModel,
    policy: policyModel,
    release: normalizeAlphaReleaseMetadata(value.release),
  };
}

export function alphaModelMetadata(value) {
  const model = normalizeAlphaModelArtifact(value);
  if (!model) return null;
  return {
    schema: model.schema,
    generatedAt: model.generatedAt,
    method: model.method,
    sampleCount: model.sampleCount,
    sources: model.sources.slice(),
    environment: model.environment,
    featureContract: model.featureContract,
    hasValue: Boolean(model.value),
    hasPolicy: Boolean(model.policy),
    valueWeights: model.value ? Object.keys(model.value.weights).length : 0,
    policyWeights: model.policy ? Object.keys(model.policy.weights).length : 0,
    training: model.training ? {
      schema: model.training.schema,
      warmStarted: model.training.warmStarted,
      parent: model.training.parent,
      data: model.training.data,
      validation: model.training.validation,
    } : null,
    release: model.release ? {
      schema: model.release.schema,
      releasedAt: model.release.releasedAt,
      sourceSchema: model.release.sourceSchema,
      sourceArtifact: model.release.sourceArtifact,
      sourceHash: model.release.sourceHash,
      sourceSizeBytes: model.release.sourceSizeBytes,
      candidateScore: model.release.candidateScore,
      minSideScore: model.release.minSideScore,
      minScoreLowerBound: model.release.minScoreLowerBound,
      minAnalyzedActions: model.release.minAnalyzedActions,
      minAverageRootVisits: model.release.minAverageRootVisits,
      minTrainingSamples: model.release.minTrainingSamples,
      minTrainingSides: model.release.minTrainingSides,
      minTrainingSources: model.release.minTrainingSources,
      minTrainingReanalysisSamples: model.release.minTrainingReanalysisSamples,
      minTrainingStateSnapshots: model.release.minTrainingStateSnapshots,
      minTrainingAverageRootVisits: model.release.minTrainingAverageRootVisits,
      minTrainingValidationSamples: model.release.minTrainingValidationSamples,
      maxTrainingValidationValueMse: model.release.maxTrainingValidationValueMse,
      maxTrainingValidationPolicyCrossEntropy: model.release.maxTrainingValidationPolicyCrossEntropy,
      scoreLowerBoundPass: model.release.scoreLowerBoundPass,
      arena: model.release.arena,
      decisionEvidence: model.release.decisionEvidence,
      activeGeneration: model.release.activeGeneration,
      evaluationSuite: model.release.evaluationSuite,
      sideScores: model.release.sideScores,
      runtime: model.release.runtime,
    } : null,
  };
}

export function alphaModelEnvironmentFingerprint({ scenario = null, rules = null } = {}) {
  const scenarioFingerprint = alphaScenarioEnvironmentFingerprint(scenario);
  const rulesFingerprint = alphaRulesEnvironmentFingerprint(rules);
  if (!scenarioFingerprint && !rulesFingerprint) return null;
  return normalizeAlphaModelEnvironment({
    schema: ALPHA_ENVIRONMENT_SCHEMA,
    scenario: scenarioFingerprint,
    rules: rulesFingerprint,
  });
}

export function normalizeAlphaModelEnvironment(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== ALPHA_ENVIRONMENT_SCHEMA) return null;
  const scenario = normalizeAlphaEnvironmentScenario(value.scenario);
  const rules = normalizeAlphaEnvironmentRules(value.rules);
  if (!scenario && !rules) return null;
  const expectedFingerprint = alphaEnvironmentHash({ scenario, rules });
  const fingerprint = typeof value.fingerprint === "string" && value.fingerprint
    ? value.fingerprint
    : expectedFingerprint;
  return {
    schema: ALPHA_ENVIRONMENT_SCHEMA,
    fingerprint,
    scenario,
    rules,
  };
}

export function validateAlphaModelEnvironment(modelOrEnvironment, expectedEnvironment = null, options = {}) {
  const environment = normalizeAlphaModelEnvironment(modelOrEnvironment?.environment ?? modelOrEnvironment);
  const expected = normalizeAlphaModelEnvironment(expectedEnvironment);
  const requireEnvironmentFingerprint = Boolean(options.requireEnvironmentFingerprint);
  if (!expected) {
    if (requireEnvironmentFingerprint && !environment) {
      return {
        ok: false,
        reason: "missing_model_environment",
        match: null,
        environment: null,
        expected: null,
      };
    }
    return {
      ok: true,
      reason: null,
      match: null,
      environment,
      expected: null,
    };
  }
  if (!environment) {
    return {
      ok: !requireEnvironmentFingerprint,
      reason: requireEnvironmentFingerprint ? "missing_model_environment" : null,
      match: null,
      environment: null,
      expected,
    };
  }
  const match = environment.fingerprint === expected.fingerprint;
  return {
    ok: match,
    reason: match ? null : "model_environment_mismatch",
    match,
    environment,
    expected,
  };
}

export function alphaModelFeatureContract({
  valueFeatureScales = null,
  policyFeatureKeys = null,
} = {}) {
  const valueScales = normalizeNumberMap(valueFeatureScales);
  const policyKeys = Array.isArray(policyFeatureKeys) ? policyFeatureKeys.map(String) : [];
  if (!Object.keys(valueScales).length && !policyKeys.length) return null;
  return normalizeAlphaModelFeatureContract({
    schema: ALPHA_FEATURE_CONTRACT_SCHEMA,
    value: Object.keys(valueScales).length
      ? {
        featureKeys: Object.keys(valueScales).sort(),
        featureScales: valueScales,
      }
      : null,
    policy: policyKeys.length
      ? {
        featureKeys: policyKeys,
      }
      : null,
  });
}

export function normalizeAlphaModelFeatureContract(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== ALPHA_FEATURE_CONTRACT_SCHEMA) return null;
  const valueContract = normalizeAlphaValueFeatureContract(value.value);
  const policyContract = normalizeAlphaPolicyFeatureContract(value.policy);
  if (!valueContract && !policyContract) return null;
  const expectedFingerprint = alphaEnvironmentHash({
    value: valueContract,
    policy: policyContract,
  });
  const fingerprint = typeof value.fingerprint === "string" && value.fingerprint
    ? value.fingerprint
    : expectedFingerprint;
  return {
    schema: ALPHA_FEATURE_CONTRACT_SCHEMA,
    fingerprint,
    value: valueContract,
    policy: policyContract,
  };
}

export function validateAlphaModelFeatureContract(modelOrContract, expectedContract = null, options = {}) {
  const model = normalizeAlphaModelArtifact(modelOrContract);
  const featureContract = normalizeAlphaModelFeatureContract(model?.featureContract ?? modelOrContract?.featureContract ?? modelOrContract);
  const expected = normalizeAlphaModelFeatureContract(expectedContract);
  const requireFeatureContract = Boolean(options.requireFeatureContract);
  const integrity = featureContract ? validateFeatureContractIntegrity(featureContract) : null;
  const submodels = model && featureContract ? validateSubmodelFeatureContract(model, featureContract) : null;
  if (!expected) {
    if (requireFeatureContract && !featureContract) {
      return {
        ok: false,
        reason: "missing_model_feature_contract",
        match: null,
        featureContract: null,
        expected: null,
        integrity,
        submodels,
      };
    }
    if (integrity && !integrity.ok) {
      return {
        ok: false,
        reason: "model_feature_contract_mismatch",
        match: false,
        featureContract,
        expected: null,
        integrity,
        submodels,
      };
    }
    if (submodels && !submodels.ok) {
      return {
        ok: false,
        reason: "model_feature_contract_mismatch",
        match: false,
        featureContract,
        expected: null,
        integrity,
        submodels,
      };
    }
    return {
      ok: true,
      reason: null,
      match: null,
      featureContract,
      expected: null,
      integrity,
      submodels,
    };
  }
  if (!featureContract) {
    return {
      ok: !requireFeatureContract,
      reason: requireFeatureContract ? "missing_model_feature_contract" : null,
      match: null,
      featureContract: null,
      expected,
      integrity,
      submodels,
    };
  }
  if (!integrity.ok || (submodels && !submodels.ok)) {
    return {
      ok: false,
      reason: "model_feature_contract_mismatch",
      match: false,
      featureContract,
      expected,
      integrity,
      submodels,
    };
  }
  const match = featureContract.fingerprint === expected.fingerprint;
  return {
    ok: match,
    reason: match ? null : "model_feature_contract_mismatch",
    match,
    featureContract,
    expected,
    integrity,
    submodels,
  };
}

export function normalizeAlphaReleaseMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== ALPHA_RELEASE_METADATA_SCHEMA) return null;
  return {
    schema: ALPHA_RELEASE_METADATA_SCHEMA,
    releasedAt: typeof value.releasedAt === "string" ? value.releasedAt : null,
    sourceSchema: typeof value.sourceSchema === "string" ? value.sourceSchema : null,
    sourceArtifact: typeof value.sourceArtifact === "string" ? value.sourceArtifact : null,
    sourceHash: normalizeSourceHash(value.sourceHash),
    sourceSizeBytes: finiteOrNull(value.sourceSizeBytes),
    promoted: Boolean(value.promoted),
    candidateScore: finiteOrNull(value.candidateScore),
    promotionThreshold: finiteOrNull(value.promotionThreshold),
    minSideScore: optionalFiniteOrNull(value.minSideScore),
    minScoreLowerBound: optionalFiniteOrNull(value.minScoreLowerBound),
    minAnalyzedActions: optionalFiniteOrNull(value.minAnalyzedActions),
    minAverageRootVisits: optionalFiniteOrNull(value.minAverageRootVisits),
    minTrainingSamples: optionalFiniteOrNull(value.minTrainingSamples),
    minTrainingSides: optionalFiniteOrNull(value.minTrainingSides),
    minTrainingSources: optionalFiniteOrNull(value.minTrainingSources),
    minTrainingReanalysisSamples: optionalFiniteOrNull(value.minTrainingReanalysisSamples),
    minTrainingStateSnapshots: optionalFiniteOrNull(value.minTrainingStateSnapshots),
    minTrainingAverageRootVisits: optionalFiniteOrNull(value.minTrainingAverageRootVisits),
    minTrainingValidationSamples: optionalFiniteOrNull(value.minTrainingValidationSamples),
    maxTrainingValidationValueMse: optionalFiniteOrNull(value.maxTrainingValidationValueMse),
    maxTrainingValidationPolicyCrossEntropy: optionalFiniteOrNull(value.maxTrainingValidationPolicyCrossEntropy),
    scoreLowerBoundPass: value.scoreLowerBoundPass !== false,
    arena: normalizeReleaseArena(value.arena),
    decisionEvidence: normalizeDecisionEvidence(value.decisionEvidence),
    errors: finiteNumber(value.errors, 0),
    activeGeneration: finiteOrNull(value.activeGeneration),
    evaluationSuite: normalizeReleaseSuiteEvidence(value.evaluationSuite),
    sideScores: normalizeReleaseSideScores(value.sideScores),
    runtime: normalizeRuntimeInstallEvidence(value.runtime),
  };
}

function normalizeRuntimeInstallEvidence(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: value.schema === "zizi-el-alamein-alpha-runtime-install-v1"
      ? value.schema
      : "zizi-el-alamein-alpha-runtime-install-v1",
    target: typeof value.target === "string" ? value.target : null,
    modelFile: typeof value.modelFile === "string" ? value.modelFile : null,
    installApproved: Boolean(value.installApproved),
  };
}

export function normalizeAlphaTrainingMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== ALPHA_TRAINING_METADATA_SCHEMA) return null;
  return {
    schema: ALPHA_TRAINING_METADATA_SCHEMA,
    warmStarted: Boolean(value.warmStarted),
    parent: normalizeTrainingParentMetadata(value.parent),
    data: normalizeTrainingDataSummary(value.data),
    validation: normalizeTrainingValidation(value.validation),
  };
}

function normalizeReleaseSuiteEvidence(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {
      schema: "zizi-el-alamein-alpha-release-suite-evidence-v1",
      explicit: false,
      games: 0,
      sides: 0,
      seeds: 0,
      fixedPositions: 0,
      challengePositions: 0,
      challengeQuality: null,
      challengeSelection: null,
    };
  }
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-release-suite-evidence-v1",
    explicit: Boolean(value.explicit),
    games: finiteNumber(value.games, 0),
    sides: finiteNumber(value.sides, 0),
    seeds: finiteNumber(value.seeds, 0),
    fixedPositions: finiteNumber(value.fixedPositions, 0),
    challengePositions: finiteNumber(value.challengePositions, 0),
    challengeQuality: normalizeChallengeQuality(value.challengeQuality),
    challengeSelection: normalizeChallengeSelection(value.challengeSelection),
  };
}

function normalizeChallengeQuality(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-challenge-quality-v1",
    samples: finiteNumber(value.samples, 0),
    averagePriority: finiteOrNull(value.averagePriority),
    averagePolicyEntropy: finiteOrNull(value.averagePolicyEntropy),
    averageTemperature: finiteOrNull(value.averageTemperature),
    averageRootNoiseWeight: finiteOrNull(value.averageRootNoiseWeight),
    averageSearchIterations: finiteOrNull(value.averageSearchIterations),
    averageRootVisits: finiteOrNull(value.averageRootVisits),
    selectionModes: normalizeCountMap(value.selectionModes),
  };
}

function normalizeChallengeSelection(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-challenge-suite-selection-v1",
    inputCount: finiteNumber(value.inputCount, 0),
    eligibleSamples: finiteNumber(value.eligibleSamples, 0),
    selectedSamples: finiteNumber(value.selectedSamples, 0),
    maxPositions: finiteNumber(value.maxPositions, 0),
    balanceBy: typeof value.balanceBy === "string" ? value.balanceBy : "none",
    priorityBy: typeof value.priorityBy === "string" ? value.priorityBy : "none",
    sources: Array.isArray(value.sources) ? value.sources.map(String) : [],
  };
}

function normalizeReleaseSideScores(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => ({
      candidateSide: entry?.candidateSide === "axis" || entry?.candidateSide === "allied"
        ? entry.candidateSide
        : null,
      games: finiteNumber(entry?.games, 0),
      candidateWins: finiteNumber(entry?.candidateWins, 0),
      baselineWins: finiteNumber(entry?.baselineWins, 0),
      draws: finiteNumber(entry?.draws, 0),
      errors: finiteNumber(entry?.errors, 0),
      scoredGames: finiteNumber(entry?.scoredGames, 0),
      candidateScore: finiteNumber(entry?.candidateScore, 0),
      arena: normalizeReleaseArena(entry?.arena),
    }))
    .filter((entry) => entry.candidateSide);
}

function normalizeDecisionEvidence(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-decision-evidence-v1",
    games: finiteNumber(value.games, 0),
    actionCount: finiteNumber(value.actionCount, 0),
    decisionCount: finiteNumber(value.decisionCount, 0),
    analyzedActions: finiteNumber(value.analyzedActions, 0),
    candidateActions: finiteNumber(value.candidateActions, 0),
    baselineActions: finiteNumber(value.baselineActions, 0),
    chanceActions: finiteNumber(value.chanceActions, 0),
    fallbackActions: finiteNumber(value.fallbackActions, 0),
    averagePolicyEntropy: finiteOrNull(value.averagePolicyEntropy),
    averageSearchIterations: finiteOrNull(value.averageSearchIterations),
    averageRootVisits: finiteOrNull(value.averageRootVisits),
    selectedActionShare: finiteOrNull(value.selectedActionShare),
    actionTypes: normalizeCountMap(value.actionTypes),
    selectionModes: normalizeCountMap(value.selectionModes),
    roles: {
      candidate: normalizeRoleDecisionEvidence(value.roles?.candidate),
      baseline: normalizeRoleDecisionEvidence(value.roles?.baseline),
    },
  };
}

function normalizeRoleDecisionEvidence(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {
      actionCount: 0,
      decisionCount: 0,
      analyzedActions: 0,
      averagePolicyEntropy: null,
      averageSearchIterations: null,
      averageRootVisits: null,
      selectedActionShare: null,
      actionTypes: {},
    };
  }
  return {
    actionCount: finiteNumber(value.actionCount, 0),
    decisionCount: finiteNumber(value.decisionCount, 0),
    analyzedActions: finiteNumber(value.analyzedActions, 0),
    averagePolicyEntropy: finiteOrNull(value.averagePolicyEntropy),
    averageSearchIterations: finiteOrNull(value.averageSearchIterations),
    averageRootVisits: finiteOrNull(value.averageRootVisits),
    selectedActionShare: finiteOrNull(value.selectedActionShare),
    actionTypes: normalizeCountMap(value.actionTypes),
  };
}

function normalizeCountMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .map(([key, raw]) => [String(key), Math.max(0, finiteNumber(raw, 0))])
      .filter(([, count]) => count > 0),
  );
}

function normalizeReleaseArena(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-arena-metrics-v1",
    games: finiteNumber(value.games, 0),
    scoredGames: finiteNumber(value.scoredGames, 0),
    errors: finiteNumber(value.errors, 0),
    score: finiteNumber(value.score, 0),
    scoreInterval95: normalizeScoreInterval(value.scoreInterval95),
    eloDiff: finiteOrNull(value.eloDiff),
    eloDiffInterval95: normalizeEloInterval(value.eloDiffInterval95),
    decisiveRate: finiteNumber(value.decisiveRate, 0),
    drawRate: finiteNumber(value.drawRate, 0),
  };
}

function normalizeScoreInterval(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    low: finiteNumber(value.low, 0),
    high: finiteNumber(value.high, 0),
  };
}

function normalizeEloInterval(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    low: finiteOrNull(value.low),
    high: finiteOrNull(value.high),
  };
}

function normalizeTrainingParentMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : null,
    generatedAt: typeof value.generatedAt === "string" ? value.generatedAt : null,
    method: typeof value.method === "string" ? value.method : null,
    sampleCount: finiteNumber(value.sampleCount, 0),
    sources: Array.isArray(value.sources) ? value.sources.map(String) : [],
    hasValue: Boolean(value.hasValue),
    hasPolicy: Boolean(value.hasPolicy),
    valueWeights: finiteNumber(value.valueWeights, 0),
    policyWeights: finiteNumber(value.policyWeights, 0),
    release: normalizeTrainingParentRelease(value.release),
    trainingData: normalizeTrainingDataSummary(value.trainingData),
    trainingValidation: normalizeTrainingValidation(value.trainingValidation),
  };
}

function normalizeTrainingParentRelease(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    schema: typeof value.schema === "string" ? value.schema : null,
    sourceSchema: typeof value.sourceSchema === "string" ? value.sourceSchema : null,
    sourceArtifact: typeof value.sourceArtifact === "string" ? value.sourceArtifact : null,
    sourceHash: normalizeSourceHash(value.sourceHash),
    activeGeneration: finiteOrNull(value.activeGeneration),
  };
}

function normalizeTrainingDataSummary(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== ALPHA_TRAINING_DATA_SUMMARY_SCHEMA) return null;
  return {
    schema: ALPHA_TRAINING_DATA_SUMMARY_SCHEMA,
    sampleCount: finiteCount(value.sampleCount),
    sources: Array.isArray(value.sources) ? value.sources.map(String) : [],
    sides: normalizeTrainingCountMap(value.sides),
    phases: normalizeTrainingCountMap(value.phases),
    outcomeSources: normalizeTrainingCountMap(value.outcomeSources),
    selectionModes: normalizeTrainingCountMap(value.selectionModes),
    samplesWithPolicy: finiteCount(value.samplesWithPolicy),
    samplesWithDecision: finiteCount(value.samplesWithDecision),
    samplesWithStateSnapshot: finiteCount(value.samplesWithStateSnapshot),
    reanalysisSamples: finiteCount(value.reanalysisSamples),
    averageOutcomeWeight: finiteOrNull(value.averageOutcomeWeight),
    averageRootVisits: finiteOrNull(value.averageRootVisits),
    averagePolicyEntropy: finiteOrNull(value.averagePolicyEntropy),
  };
}

function normalizeTrainingValidation(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (value.schema !== ALPHA_TRAINING_VALIDATION_SCHEMA) return null;
  return {
    schema: ALPHA_TRAINING_VALIDATION_SCHEMA,
    seed: typeof value.seed === "string" ? value.seed : null,
    validationFraction: finiteOrNull(value.validationFraction),
    validationGroupBy: typeof value.validationGroupBy === "string" ? value.validationGroupBy : "sample",
    trainingGroups: finiteCount(value.trainingGroups),
    validationGroups: finiteCount(value.validationGroups),
    trainingSamples: finiteCount(value.trainingSamples),
    sampleCount: finiteCount(value.sampleCount),
    sides: normalizeTrainingCountMap(value.sides),
    phases: normalizeTrainingCountMap(value.phases),
    outcomeSources: normalizeTrainingCountMap(value.outcomeSources),
    value: normalizeMetrics(value.value),
    policy: normalizeMetrics(value.policy),
  };
}

function normalizeTrainingCountMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(Object.entries(value)
    .map(([key, count]) => [String(key), finiteCount(count)])
    .filter(([key, count]) => key && count > 0));
}

function alphaScenarioEnvironmentFingerprint(scenario) {
  if (!scenario || typeof scenario !== "object" || Array.isArray(scenario)) return null;
  const board = scenario.board && typeof scenario.board === "object" ? scenario.board : {};
  const hexes = Array.isArray(board.hexes) ? board.hexes : [];
  const units = Array.isArray(scenario.units) ? scenario.units : [];
  const objectives = scenario.objectives && typeof scenario.objectives === "object" ? scenario.objectives : {};
  return {
    id: stringOrNull(scenario.id || scenario.name || scenario.title),
    format: stringOrNull(scenario.format),
    version: finiteOrNull(scenario.version),
    boardName: stringOrNull(board.name),
    boardImage: stringOrNull(board.image),
    boardSize: {
      width: finiteOrNull(board.width),
      height: finiteOrNull(board.height),
    },
    hexCount: hexes.length,
    unitCount: units.length,
    objectiveKeys: Object.keys(objectives).sort(),
    hexHash: alphaEnvironmentHash(hexes.map(compactScenarioHexForEnvironment)),
    unitHash: alphaEnvironmentHash(units.map(compactScenarioUnitForEnvironment)),
    objectiveHash: alphaEnvironmentHash(objectives),
  };
}

function alphaRulesEnvironmentFingerprint(rules) {
  if (!rules || typeof rules !== "object" || Array.isArray(rules)) return null;
  return {
    format: stringOrNull(rules.format),
    version: finiteOrNull(rules.version),
    turnCount: Array.isArray(rules.turns) ? rules.turns.length : 0,
    phases: Array.isArray(rules.phases)
      ? rules.phases.map((phase) => ({
        id: stringOrNull(phase?.id),
        side: stringOrNull(phase?.side),
        type: stringOrNull(phase?.type),
      }))
      : [],
    crtHash: alphaEnvironmentHash(rules.crt || rules.combatResults || null),
    terrainHash: alphaEnvironmentHash(rules.terrain || null),
    rulesHash: alphaEnvironmentHash(stripDisplayTextForEnvironment(rules)),
  };
}

function normalizeAlphaEnvironmentScenario(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    id: stringOrNull(value.id),
    format: stringOrNull(value.format),
    version: finiteOrNull(value.version),
    boardName: stringOrNull(value.boardName),
    boardImage: stringOrNull(value.boardImage),
    boardSize: {
      width: finiteOrNull(value.boardSize?.width),
      height: finiteOrNull(value.boardSize?.height),
    },
    hexCount: finiteCount(value.hexCount),
    unitCount: finiteCount(value.unitCount),
    objectiveKeys: Array.isArray(value.objectiveKeys) ? value.objectiveKeys.map(String).sort() : [],
    hexHash: stringOrNull(value.hexHash),
    unitHash: stringOrNull(value.unitHash),
    objectiveHash: stringOrNull(value.objectiveHash),
  };
}

function normalizeAlphaEnvironmentRules(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    format: stringOrNull(value.format),
    version: finiteOrNull(value.version),
    turnCount: finiteCount(value.turnCount),
    phases: Array.isArray(value.phases)
      ? value.phases.map((phase) => ({
        id: stringOrNull(phase?.id),
        side: stringOrNull(phase?.side),
        type: stringOrNull(phase?.type),
      }))
      : [],
    crtHash: stringOrNull(value.crtHash),
    terrainHash: stringOrNull(value.terrainHash),
    rulesHash: stringOrNull(value.rulesHash),
  };
}

function normalizeAlphaValueFeatureContract(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const featureScales = normalizeNumberMap(value.featureScales);
  const featureKeys = Array.isArray(value.featureKeys)
    ? value.featureKeys.map(String).sort()
    : Object.keys(featureScales).sort();
  if (!featureKeys.length && !Object.keys(featureScales).length) return null;
  return {
    featureKeys,
    featureScales,
    featureScalesHash: alphaEnvironmentHash(featureScales),
  };
}

function normalizeAlphaPolicyFeatureContract(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const featureKeys = Array.isArray(value.featureKeys) ? value.featureKeys.map(String) : [];
  if (!featureKeys.length) return null;
  return {
    featureKeys,
    featureKeysHash: alphaEnvironmentHash(featureKeys),
  };
}

function validateFeatureContractIntegrity(featureContract) {
  const expectedFingerprint = alphaEnvironmentHash({
    value: featureContract.value,
    policy: featureContract.policy,
  });
  const match = featureContract.fingerprint === expectedFingerprint;
  return {
    ok: match,
    reason: match ? null : "feature_contract_fingerprint_mismatch",
    match,
    fingerprint: featureContract.fingerprint,
    expectedFingerprint,
  };
}

function validateSubmodelFeatureContract(model, featureContract) {
  const value = validateValueSubmodelFeatureContract(model.value, featureContract.value);
  const policy = validatePolicySubmodelFeatureContract(model.policy, featureContract.policy);
  return {
    ok: value.ok && policy.ok,
    value,
    policy,
  };
}

function validateValueSubmodelFeatureContract(valueModel, valueContract) {
  if (!valueModel) return { ok: true, reason: null, match: null };
  if (!valueContract) return { ok: false, reason: "missing_value_feature_contract", match: false };
  const featureScales = normalizeNumberMap(valueModel.featureScales);
  const featureKeys = (valueModel.featureKeys.length
    ? valueModel.featureKeys
    : Object.keys(featureScales)).map(String).sort();
  const featureKeysMatch = sameStringArray(featureKeys, valueContract.featureKeys);
  const featureScalesHash = alphaEnvironmentHash(featureScales);
  const featureScalesMatch = featureScalesHash === valueContract.featureScalesHash;
  return {
    ok: featureKeysMatch && featureScalesMatch,
    reason: featureKeysMatch && featureScalesMatch ? null : "value_feature_contract_mismatch",
    match: featureKeysMatch && featureScalesMatch,
    featureKeysMatch,
    featureScalesMatch,
    featureKeys,
    expectedFeatureKeys: valueContract.featureKeys,
    featureScalesHash,
    expectedFeatureScalesHash: valueContract.featureScalesHash,
  };
}

function validatePolicySubmodelFeatureContract(policyModel, policyContract) {
  if (!policyModel) return { ok: true, reason: null, match: null };
  if (!policyContract) return { ok: false, reason: "missing_policy_feature_contract", match: false };
  const featureKeys = policyModel.featureKeys.map(String);
  const featureKeysMatch = sameStringArray(featureKeys, policyContract.featureKeys);
  return {
    ok: featureKeysMatch,
    reason: featureKeysMatch ? null : "policy_feature_contract_mismatch",
    match: featureKeysMatch,
    featureKeys,
    expectedFeatureKeys: policyContract.featureKeys,
  };
}

function sameStringArray(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
  return left.every((value, index) => String(value) === String(right[index]));
}

function compactScenarioHexForEnvironment(hex) {
  return {
    id: stringOrNull(hex?.id),
    col: finiteOrNull(hex?.col),
    row: finiteOrNull(hex?.row),
    terrain: stringOrNull(hex?.terrain),
    road: Boolean(hex?.road),
    britishPosition: Boolean(hex?.britishPosition),
    objective: Array.isArray(hex?.objective) ? hex.objective.map(String).sort() : [],
  };
}

function compactScenarioUnitForEnvironment(unit) {
  return {
    id: stringOrNull(unit?.id),
    side: stringOrNull(unit?.side),
    type: stringOrNull(unit?.type),
    attack: finiteOrNull(unit?.attack),
    defense: finiteOrNull(unit?.defense),
    movement: finiteOrNull(unit?.movement),
    steps: finiteOrNull(unit?.steps),
    hexId: stringOrNull(unit?.hexId),
    entryTurn: finiteOrNull(unit?.entryTurn),
  };
}

function stripDisplayTextForEnvironment(value, seen = new WeakSet()) {
  if (value === null) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "undefined" || typeof value === "function" || typeof value === "symbol") return undefined;
  if (Array.isArray(value)) {
    return value
      .map((entry) => stripDisplayTextForEnvironment(entry, seen))
      .filter((entry) => entry !== undefined);
  }
  if (typeof value !== "object") return null;
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  const output = {};
  for (const key of Object.keys(value).sort()) {
    if (["label", "labels", "name", "description", "text"].includes(key)) continue;
    const next = stripDisplayTextForEnvironment(value[key], seen);
    if (next !== undefined) output[key] = next;
  }
  seen.delete(value);
  return output;
}

function alphaEnvironmentHash(value) {
  return `fnv1a32:${stableHash(stableValue(value))}`;
}

function trainingDataGate(data, options = {}) {
  const thresholds = {
    minTrainingSamples: optionalFiniteOrNull(options.minTrainingSamples),
    minTrainingSides: optionalFiniteOrNull(options.minTrainingSides),
    minTrainingSources: optionalFiniteOrNull(options.minTrainingSources),
    minTrainingReanalysisSamples: optionalFiniteOrNull(options.minTrainingReanalysisSamples),
    minTrainingStateSnapshots: optionalFiniteOrNull(options.minTrainingStateSnapshots),
    minTrainingAverageRootVisits: optionalFiniteOrNull(options.minTrainingAverageRootVisits),
  };
  if (Object.values(thresholds).every((value) => value === null)) return { ok: true, reason: null };
  if (!data) return { ok: false, reason: "missing_training_data_evidence" };
  if (thresholds.minTrainingSamples !== null && Number(data.sampleCount || 0) < thresholds.minTrainingSamples) {
    return { ok: false, reason: "training_samples_too_few" };
  }
  if (thresholds.minTrainingSides !== null && Object.keys(data.sides || {}).length < thresholds.minTrainingSides) {
    return { ok: false, reason: "training_side_coverage_too_narrow" };
  }
  if (thresholds.minTrainingSources !== null && (data.sources || []).length < thresholds.minTrainingSources) {
    return { ok: false, reason: "training_sources_too_few" };
  }
  if (
    thresholds.minTrainingReanalysisSamples !== null
    && Number(data.reanalysisSamples || 0) < thresholds.minTrainingReanalysisSamples
  ) {
    return { ok: false, reason: "training_reanalysis_samples_too_few" };
  }
  if (
    thresholds.minTrainingStateSnapshots !== null
    && Number(data.samplesWithStateSnapshot || 0) < thresholds.minTrainingStateSnapshots
  ) {
    return { ok: false, reason: "training_state_snapshots_too_few" };
  }
  if (
    thresholds.minTrainingAverageRootVisits !== null
    && (data.averageRootVisits === null || Number(data.averageRootVisits || 0) < thresholds.minTrainingAverageRootVisits)
  ) {
    return { ok: false, reason: "training_root_visits_too_low" };
  }
  return { ok: true, reason: null };
}

function trainingValidationGate(validation, options = {}) {
  const minSamples = optionalFiniteOrNull(options.minTrainingValidationSamples);
  const maxValueMse = optionalFiniteOrNull(options.maxTrainingValidationValueMse);
  const maxPolicyCrossEntropy = optionalFiniteOrNull(options.maxTrainingValidationPolicyCrossEntropy);
  if (minSamples === null && maxValueMse === null && maxPolicyCrossEntropy === null) {
    return { ok: true, reason: null };
  }
  if (!validation) return { ok: false, reason: "missing_training_validation_evidence" };
  if (minSamples !== null && Number(validation.sampleCount || 0) < minSamples) {
    return { ok: false, reason: "training_validation_samples_too_few" };
  }
  if (
    maxValueMse !== null
    && (validation.value?.mse === null || validation.value?.mse === undefined || Number(validation.value.mse) > maxValueMse)
  ) {
    return { ok: false, reason: "training_validation_value_mse_too_high" };
  }
  if (
    maxPolicyCrossEntropy !== null
    && (
      validation.policy?.crossEntropy === null
      || validation.policy?.crossEntropy === undefined
      || Number(validation.policy.crossEntropy) > maxPolicyCrossEntropy
    )
  ) {
    return { ok: false, reason: "training_validation_policy_cross_entropy_too_high" };
  }
  return { ok: true, reason: null };
}

function releaseValidationError(reason, model, environment = null, featureContract = null) {
  return {
    ok: false,
    reason,
    model,
    environment,
    featureContract,
  };
}

function normalizeSubModel(value, schema) {
  if (!value || typeof value !== "object") return null;
  if (value.schema !== schema) return null;
  const weights = normalizeWeights(value.weights);
  if (!Object.keys(weights).length) return null;
  return {
    schema,
    featureScales: normalizeNumberMap(value.featureScales),
    featureKeys: Array.isArray(value.featureKeys) ? value.featureKeys.map(String) : [],
    weights,
    metrics: normalizeMetrics(value.metrics),
  };
}

function normalizeWeights(weights) {
  return normalizeNumberMap(weights);
}

function normalizeNumberMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .map(([key, raw]) => [String(key), Number(raw)])
      .filter(([, next]) => Number.isFinite(next)),
  );
}

function normalizeMetrics(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, raw]) => raw === null || Number.isFinite(Number(raw)))
      .map(([key, raw]) => [String(key), raw === null ? null : Number(raw)]),
  );
}

function finiteNumber(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function optionalFiniteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  return finiteOrNull(value);
}

function stringOrNull(value) {
  return typeof value === "string" && value ? value : null;
}

function finiteCount(value) {
  const next = Number(value);
  return Number.isFinite(next) ? Math.max(0, Math.floor(next)) : 0;
}

function normalizeSourceHash(value) {
  if (typeof value !== "string") return null;
  const next = value.trim().toLowerCase();
  return /^sha256:[a-f0-9]{64}$/.test(next) ? next : null;
}

function stableHash(text) {
  let hash = 2166136261;
  for (let index = 0; index < String(text).length; index += 1) {
    hash ^= String(text).charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function stableValue(value, seen = new WeakSet()) {
  if (value === null) return "null";
  if (typeof value === "number") return Number.isFinite(value) ? JSON.stringify(value) : "null";
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "undefined" || typeof value === "function" || typeof value === "symbol") return "null";
  if (Array.isArray(value)) return `[${value.map((entry) => stableValue(entry, seen)).join(",")}]`;
  if (typeof value !== "object") return "null";
  if (seen.has(value)) return JSON.stringify("[Circular]");
  seen.add(value);
  const output = `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${stableValue(value[key], seen)}`
  )).join(",")}}`;
  seen.delete(value);
  return output;
}
