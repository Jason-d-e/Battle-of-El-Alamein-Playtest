import {
  ALPHA_RELEASE_METADATA_SCHEMA,
  alphaModelMetadata,
  extractAlphaModelArtifact,
  normalizeAlphaTrainingProfileEvidence,
} from "./ai-alpha-model.js";

const DEFAULT_MIN_SCORE = 0.55;
const DEFAULT_MIN_SUITE_GAMES = 2;
const DEFAULT_MIN_SUITE_SIDES = 2;
const SUPPORTED_TRAINING_VALIDATION_GROUP_BY = Object.freeze(["sample", "stateHash", "side", "phase", "trajectory"]);

export function createAlphaModelRelease(artifact, options = {}) {
  const minCandidateScore = Number(options.minCandidateScore ?? DEFAULT_MIN_SCORE);
  const maxErrors = Number(options.maxErrors ?? 0);
  const minSuiteGames = Math.max(0, Number(options.minSuiteGames ?? DEFAULT_MIN_SUITE_GAMES));
  const minSuiteSides = Math.max(0, Number(options.minSuiteSides ?? DEFAULT_MIN_SUITE_SIDES));
  const minEvaluationPhases = Math.max(0, Number(options.minEvaluationPhases ?? 0));
  const minFixedPositions = Math.max(0, Number(options.minFixedPositions ?? 0));
  const minChallengePositions = Math.max(0, Number(options.minChallengePositions ?? 0));
  const minChallengeAverageRecommendationUncertainty = optionalFiniteOrNull(options.minChallengeAverageRecommendationUncertainty);
  const minContestedChallengePositions = optionalFiniteOrNull(options.minContestedChallengePositions);
  const minChallengeAverageRuntimeRisk = optionalFiniteOrNull(options.minChallengeAverageRuntimeRisk);
  const minRuntimeRiskChallengePositions = optionalFiniteOrNull(options.minRuntimeRiskChallengePositions);
  const requiredTrainingProfile = optionalStringOrNull(options.requiredTrainingProfile);
  const requireEvaluationSuite = options.requireEvaluationSuite !== false;
  const allowDirectModel = Boolean(options.allowDirectModel);
  const source = classifyReleaseSource(artifact);
  if (!source.ok) return source;
  const minSideScore = optionalFiniteOrNull(options.minSideScore ?? source.minSideScore);
  const minSideScoreDelta = optionalFiniteOrNull(options.minSideScoreDelta ?? source.minSideScoreDelta);
  const minScoreLowerBound = optionalFiniteOrNull(options.minScoreLowerBound ?? source.minScoreLowerBound);
  const minEloLowerBound = optionalFiniteOrNull(options.minEloLowerBound ?? source.minEloLowerBound);
  const maxArenaDrawRate = optionalFiniteOrNull(options.maxArenaDrawRate ?? source.maxArenaDrawRate);
  const minArenaDecisiveRate = optionalFiniteOrNull(options.minArenaDecisiveRate ?? source.minArenaDecisiveRate);
  const minAnalyzedActions = optionalFiniteOrNull(options.minAnalyzedActions ?? source.minAnalyzedActions);
  const minAverageRootVisits = optionalFiniteOrNull(options.minAverageRootVisits ?? source.minAverageRootVisits);
  const minCandidateAnalyzedActions = optionalFiniteOrNull(
    options.minCandidateAnalyzedActions ?? source.minCandidateAnalyzedActions,
  );
  const minCandidateAverageRootVisits = optionalFiniteOrNull(
    options.minCandidateAverageRootVisits ?? source.minCandidateAverageRootVisits,
  );
  const minCandidateDecisionActionTypes = optionalFiniteOrNull(
    options.minCandidateDecisionActionTypes ?? source.minCandidateDecisionActionTypes,
  );
  const minCandidateDecisionAverageRecommendationConfidence = optionalFiniteOrNull(
    options.minCandidateDecisionAverageRecommendationConfidence ?? source.minCandidateDecisionAverageRecommendationConfidence,
  );
  const maxCandidateDecisionAverageRecommendationUncertainty = optionalFiniteOrNull(
    options.maxCandidateDecisionAverageRecommendationUncertainty ?? source.maxCandidateDecisionAverageRecommendationUncertainty,
  );
  const minCandidateDecisionSelectedActionShare = optionalFiniteOrNull(
    options.minCandidateDecisionSelectedActionShare ?? source.minCandidateDecisionSelectedActionShare,
  );
  const minDecisionAverageRecommendationConfidence = optionalFiniteOrNull(
    options.minDecisionAverageRecommendationConfidence ?? source.minDecisionAverageRecommendationConfidence,
  );
  const maxDecisionAverageRecommendationUncertainty = optionalFiniteOrNull(
    options.maxDecisionAverageRecommendationUncertainty ?? source.maxDecisionAverageRecommendationUncertainty,
  );
  const minDecisionSelectedActionShare = optionalFiniteOrNull(
    options.minDecisionSelectedActionShare ?? source.minDecisionSelectedActionShare,
  );
  const minDecisionActionTypes = optionalFiniteOrNull(
    options.minDecisionActionTypes ?? source.minDecisionActionTypes,
  );
  const maxDecisionFallbackRate = optionalFiniteOrNull(
    options.maxDecisionFallbackRate ?? source.maxDecisionFallbackRate,
  );
  const minReferenceArenaScore = optionalFiniteOrNull(options.minReferenceArenaScore ?? source.minReferenceArenaScore);
  const minReferenceArenaReferences = optionalFiniteOrNull(
    options.minReferenceArenaReferences ?? source.minReferenceArenaReferences,
  );
  const maxReferenceArenaErrors = optionalFiniteOrNull(options.maxReferenceArenaErrors ?? source.maxReferenceArenaErrors);
  const trainingThresholds = {
    minTrainingSamples: optionalFiniteOrNull(options.minTrainingSamples),
    minTrainingValueSamples: optionalFiniteOrNull(options.minTrainingValueSamples),
    minTrainingOutcomeClasses: optionalFiniteOrNull(options.minTrainingOutcomeClasses),
    minTrainingPolicyRows: optionalFiniteOrNull(options.minTrainingPolicyRows),
    minTrainingPolicyActionTypes: optionalFiniteOrNull(options.minTrainingPolicyActionTypes),
    minTrainingUniqueStateHashes: optionalFiniteOrNull(options.minTrainingUniqueStateHashes),
    maxTrainingDuplicateStateRate: optionalFiniteOrNull(options.maxTrainingDuplicateStateRate),
    minTrainingSides: optionalFiniteOrNull(options.minTrainingSides),
    minTrainingSources: optionalFiniteOrNull(options.minTrainingSources),
    minTrainingReanalysisSamples: optionalFiniteOrNull(options.minTrainingReanalysisSamples),
    minTrainingStateSnapshots: optionalFiniteOrNull(options.minTrainingStateSnapshots),
    minTrainingAverageRootVisits: optionalFiniteOrNull(options.minTrainingAverageRootVisits),
    minTrainingSelectedActionShare: optionalFiniteOrNull(options.minTrainingSelectedActionShare),
    minTrainingExplorationShare: optionalFiniteOrNull(options.minTrainingExplorationShare),
    minTrainingValidationSamples: optionalFiniteOrNull(options.minTrainingValidationSamples),
    minTrainingValidationSides: optionalFiniteOrNull(options.minTrainingValidationSides),
    minTrainingValidationPhases: optionalFiniteOrNull(options.minTrainingValidationPhases),
    minTrainingValidationGroups: optionalFiniteOrNull(options.minTrainingValidationGroups),
    requiredTrainingValidationGroupBy: optionalStringOrNull(options.requiredTrainingValidationGroupBy),
    maxTrainingValidationValueMse: optionalFiniteOrNull(options.maxTrainingValidationValueMse),
    maxTrainingValidationValueCalibrationBias: optionalFiniteOrNull(options.maxTrainingValidationValueCalibrationBias),
    maxTrainingValidationPolicyCrossEntropy: optionalFiniteOrNull(options.maxTrainingValidationPolicyCrossEntropy),
    minTrainingValidationPolicyTopChoiceAccuracy: optionalFiniteOrNull(options.minTrainingValidationPolicyTopChoiceAccuracy),
  };

  const model = extractAlphaModelArtifact(source.modelSource);
  if (!model) return releaseError("invalid_alpha_model", source);
  const metadata = alphaModelMetadata(model);

  if (!allowDirectModel && source.sourceSchema === "zizi-el-alamein-alpha-model-v1") {
    return releaseError("missing_promotion_evidence", source, model);
  }

  if (source.errors > maxErrors) {
    return releaseError("release_errors_exceed_limit", source, model);
  }

  if (source.requiresPromotion && !source.promoted) {
    return releaseError("candidate_not_promoted", source, model);
  }

  const promotionVerdictGateResult = promotionVerdictGate(source.promotionVerdict);
  if (!promotionVerdictGateResult.ok) {
    return releaseError(promotionVerdictGateResult.reason, {
      ...source,
      promotionVerdict: promotionVerdictGateResult.promotionVerdict,
    }, model);
  }

  if (source.candidateScore !== null && source.candidateScore < minCandidateScore) {
    return releaseError("candidate_score_below_threshold", source, model);
  }

  if (minSideScore !== null) {
    const sideGate = sideScoreGate(source.sideScores, minSideScore);
    if (!sideGate.ok) return releaseError(sideGate.reason, source, model);
  }

  if (minSideScoreDelta !== null) {
    const sideDeltaGate = sideScoreDeltaGate(source.sideScores, minSideScoreDelta);
    if (!sideDeltaGate.ok) return releaseError(sideDeltaGate.reason, source, model);
  }

  if (minScoreLowerBound !== null) {
    const lowerBoundGate = scoreLowerBoundGate(source.arena, minScoreLowerBound, source.scoreLowerBoundPass);
    if (!lowerBoundGate.ok) return releaseError(lowerBoundGate.reason, source, model);
  }

  if (minEloLowerBound !== null) {
    const lowerBoundGate = eloLowerBoundGate(source.arena, minEloLowerBound, source.eloLowerBoundPass);
    if (!lowerBoundGate.ok) return releaseError(lowerBoundGate.reason, source, model);
  }

  if (maxArenaDrawRate !== null || minArenaDecisiveRate !== null) {
    const rateGate = arenaRateGate(source.arena, {
      maxArenaDrawRate,
      minArenaDecisiveRate,
      drawRatePass: source.drawRatePass,
      decisiveRatePass: source.decisiveRatePass,
    });
    if (!rateGate.ok) {
      return releaseError(rateGate.reason, {
        ...source,
        maxArenaDrawRate,
        minArenaDecisiveRate,
        drawRatePass: rateGate.drawRatePass,
        decisiveRatePass: rateGate.decisiveRatePass,
      }, model);
    }
  }

  if (
    source.requiresPromotion
    && (
      minAnalyzedActions !== null
      || minAverageRootVisits !== null
      || minDecisionAverageRecommendationConfidence !== null
      || maxDecisionAverageRecommendationUncertainty !== null
      || minDecisionSelectedActionShare !== null
      || minDecisionActionTypes !== null
      || maxDecisionFallbackRate !== null
    )
  ) {
    const decisionGate = decisionEvidenceGate(source.decisionEvidence, {
      minAnalyzedActions,
      minAverageRootVisits,
      minDecisionAverageRecommendationConfidence,
      maxDecisionAverageRecommendationUncertainty,
      minDecisionSelectedActionShare,
      minDecisionActionTypes,
      maxDecisionFallbackRate,
    });
    if (!decisionGate.ok) {
      return releaseError(decisionGate.reason, {
        ...source,
        minDecisionAverageRecommendationConfidence,
        maxDecisionAverageRecommendationUncertainty,
        minDecisionSelectedActionShare,
        minDecisionActionTypes,
        maxDecisionFallbackRate,
      }, model);
    }
  }

  if (
    source.requiresPromotion
    && (
      minCandidateAnalyzedActions !== null
      || minCandidateAverageRootVisits !== null
      || minCandidateDecisionActionTypes !== null
      || minCandidateDecisionAverageRecommendationConfidence !== null
      || maxCandidateDecisionAverageRecommendationUncertainty !== null
      || minCandidateDecisionSelectedActionShare !== null
    )
  ) {
    const candidateDecisionGate = candidateDecisionEvidenceGate(source.decisionEvidence, {
      minCandidateAnalyzedActions,
      minCandidateAverageRootVisits,
      minCandidateDecisionActionTypes,
      minCandidateDecisionAverageRecommendationConfidence,
      maxCandidateDecisionAverageRecommendationUncertainty,
      minCandidateDecisionSelectedActionShare,
    });
    if (!candidateDecisionGate.ok) {
      return releaseError(candidateDecisionGate.reason, {
        ...source,
        minCandidateAnalyzedActions,
        minCandidateAverageRootVisits,
        minCandidateDecisionActionTypes,
        minCandidateDecisionAverageRecommendationConfidence,
        maxCandidateDecisionAverageRecommendationUncertainty,
        minCandidateDecisionSelectedActionShare,
      }, model);
    }
  }

  const trainingGate = trainingDataGate(metadata?.training?.data, trainingThresholds);
  if (!trainingGate.ok) {
    return releaseError(trainingGate.reason, { ...source, ...trainingThresholds }, model);
  }
  const validationGate = trainingValidationGate(metadata?.training?.validation, trainingThresholds);
  if (!validationGate.ok) {
    return releaseError(validationGate.reason, { ...source, ...trainingThresholds }, model);
  }
  const referenceGate = referenceArenaGate(source.referenceArena, {
    minReferenceArenaScore,
    minReferenceArenaReferences,
    maxReferenceArenaErrors,
  });
  if (!referenceGate.ok) {
    return releaseError(referenceGate.reason, {
      ...source,
      minReferenceArenaScore,
      minReferenceArenaReferences,
      maxReferenceArenaErrors,
    }, model);
  }

  if (source.requiresPromotion && requireEvaluationSuite) {
    if (!source.evaluationSuite?.games) {
      return releaseError("missing_evaluation_suite", source, model);
    }
    if (!source.evaluationSuite.explicit) {
      return releaseError("evaluation_suite_not_explicit", source, model);
    }
    if (source.evaluationSuite.games < minSuiteGames) {
      return releaseError("evaluation_suite_too_small", source, model);
    }
    if (source.evaluationSuite.sides < minSuiteSides) {
      return releaseError("evaluation_suite_side_coverage_too_narrow", source, model);
    }
    if (source.evaluationSuite.phaseCoverage.phases < minEvaluationPhases) {
      return releaseError("evaluation_suite_phase_coverage_too_narrow", source, model);
    }
    if (source.evaluationSuite.fixedPositions < minFixedPositions) {
      return releaseError("evaluation_suite_fixed_positions_too_few", source, model);
    }
    if (source.evaluationSuite.challengePositions < minChallengePositions) {
      return releaseError("evaluation_suite_challenge_positions_too_few", source, model);
    }
    const challengeQualityGateResult = challengeQualityGate(source.evaluationSuite.challengeQuality, {
      minChallengeAverageRecommendationUncertainty,
      minContestedChallengePositions,
      minChallengeAverageRuntimeRisk,
      minRuntimeRiskChallengePositions,
    });
    if (!challengeQualityGateResult.ok) {
      return releaseError(challengeQualityGateResult.reason, {
        ...source,
        minChallengeAverageRecommendationUncertainty,
        minContestedChallengePositions,
        minChallengeAverageRuntimeRisk,
        minRuntimeRiskChallengePositions,
      }, model);
    }
  }
  const profileGate = trainingProfileGate(source.trainingProfile, requiredTrainingProfile);
  if (!profileGate.ok) {
    return releaseError(profileGate.reason, {
      ...source,
      requiredTrainingProfile,
    }, model);
  }

  return {
    ok: true,
    reason: null,
    sourceSchema: source.sourceSchema,
    promoted: source.promoted,
    candidateScore: source.candidateScore,
    promotionThreshold: source.promotionThreshold,
    minSideScore,
    minSideScoreDelta,
    minScoreLowerBound,
    minEloLowerBound,
    maxArenaDrawRate,
    minArenaDecisiveRate,
    minAnalyzedActions,
    minAverageRootVisits,
    minCandidateAnalyzedActions,
    minCandidateAverageRootVisits,
    minCandidateDecisionActionTypes,
    minCandidateDecisionAverageRecommendationConfidence,
    maxCandidateDecisionAverageRecommendationUncertainty,
    minCandidateDecisionSelectedActionShare,
    minDecisionAverageRecommendationConfidence,
    maxDecisionAverageRecommendationUncertainty,
    minDecisionSelectedActionShare,
    minDecisionActionTypes,
    maxDecisionFallbackRate,
    minReferenceArenaScore,
    minReferenceArenaReferences,
    maxReferenceArenaErrors,
    minChallengeAverageRecommendationUncertainty,
    minContestedChallengePositions,
    minChallengeAverageRuntimeRisk,
    minRuntimeRiskChallengePositions,
    minEvaluationPhases,
    requiredTrainingProfile,
    trainingProfile: normalizeAlphaTrainingProfileEvidence(source.trainingProfile),
    ...trainingThresholds,
    promotionVerdict: promotionVerdictGateResult.promotionVerdict,
    sideScoreDeltaPass: source.sideScoreDeltaPass !== false,
    scoreLowerBoundPass: source.scoreLowerBoundPass !== false,
    eloLowerBoundPass: source.eloLowerBoundPass !== false,
    drawRatePass: source.drawRatePass !== false,
    decisiveRatePass: source.decisiveRatePass !== false,
    arena: normalizeReleaseArena(source.arena),
    decisionEvidence: normalizeDecisionEvidence(source.decisionEvidence),
    errors: source.errors,
    activeGeneration: source.activeGeneration,
    evaluationSuite: source.evaluationSuite,
    referenceArena: source.referenceArena || emptyReferenceArenaEvidence(),
    sideScores: source.sideScores || [],
    model,
    metadata,
  };
}

export function createAlphaReleasedModelArtifact(release, options = {}) {
  if (!release?.ok || !release.model) return null;
  return {
    ...release.model,
    release: {
      schema: ALPHA_RELEASE_METADATA_SCHEMA,
      releasedAt: typeof options.releasedAt === "string" ? options.releasedAt : new Date().toISOString(),
      sourceSchema: release.sourceSchema || null,
      sourceArtifact: typeof options.sourceArtifact === "string" ? options.sourceArtifact : null,
      sourceHash: typeof options.sourceHash === "string" ? options.sourceHash : null,
      sourceSizeBytes: finiteOrNull(options.sourceSizeBytes),
      promoted: Boolean(release.promoted),
      candidateScore: release.candidateScore ?? null,
      promotionThreshold: release.promotionThreshold ?? null,
      minSideScore: release.minSideScore ?? null,
      minSideScoreDelta: release.minSideScoreDelta ?? null,
      minScoreLowerBound: release.minScoreLowerBound ?? null,
      minEloLowerBound: release.minEloLowerBound ?? null,
      maxArenaDrawRate: release.maxArenaDrawRate ?? null,
      minArenaDecisiveRate: release.minArenaDecisiveRate ?? null,
      minAnalyzedActions: release.minAnalyzedActions ?? null,
      minAverageRootVisits: release.minAverageRootVisits ?? null,
      minCandidateAnalyzedActions: release.minCandidateAnalyzedActions ?? null,
      minCandidateAverageRootVisits: release.minCandidateAverageRootVisits ?? null,
      minCandidateDecisionActionTypes: release.minCandidateDecisionActionTypes ?? null,
      minCandidateDecisionAverageRecommendationConfidence: release.minCandidateDecisionAverageRecommendationConfidence ?? null,
      maxCandidateDecisionAverageRecommendationUncertainty: release.maxCandidateDecisionAverageRecommendationUncertainty ?? null,
      minCandidateDecisionSelectedActionShare: release.minCandidateDecisionSelectedActionShare ?? null,
      minDecisionAverageRecommendationConfidence: release.minDecisionAverageRecommendationConfidence ?? null,
      maxDecisionAverageRecommendationUncertainty: release.maxDecisionAverageRecommendationUncertainty ?? null,
      minDecisionSelectedActionShare: release.minDecisionSelectedActionShare ?? null,
      minDecisionActionTypes: release.minDecisionActionTypes ?? null,
      maxDecisionFallbackRate: release.maxDecisionFallbackRate ?? null,
      minReferenceArenaScore: release.minReferenceArenaScore ?? null,
      minReferenceArenaReferences: release.minReferenceArenaReferences ?? null,
      maxReferenceArenaErrors: release.maxReferenceArenaErrors ?? null,
      minChallengeAverageRecommendationUncertainty: release.minChallengeAverageRecommendationUncertainty ?? null,
      minContestedChallengePositions: release.minContestedChallengePositions ?? null,
      minChallengeAverageRuntimeRisk: release.minChallengeAverageRuntimeRisk ?? null,
      minRuntimeRiskChallengePositions: release.minRuntimeRiskChallengePositions ?? null,
      minEvaluationPhases: release.minEvaluationPhases ?? null,
      requiredTrainingProfile: release.requiredTrainingProfile ?? null,
      trainingProfile: normalizeAlphaTrainingProfileEvidence(release.trainingProfile),
      minTrainingSamples: release.minTrainingSamples ?? null,
      minTrainingValueSamples: release.minTrainingValueSamples ?? null,
      minTrainingOutcomeClasses: release.minTrainingOutcomeClasses ?? null,
      minTrainingPolicyRows: release.minTrainingPolicyRows ?? null,
      minTrainingPolicyActionTypes: release.minTrainingPolicyActionTypes ?? null,
      minTrainingUniqueStateHashes: release.minTrainingUniqueStateHashes ?? null,
      maxTrainingDuplicateStateRate: release.maxTrainingDuplicateStateRate ?? null,
      minTrainingSides: release.minTrainingSides ?? null,
      minTrainingSources: release.minTrainingSources ?? null,
      minTrainingReanalysisSamples: release.minTrainingReanalysisSamples ?? null,
      minTrainingStateSnapshots: release.minTrainingStateSnapshots ?? null,
      minTrainingAverageRootVisits: release.minTrainingAverageRootVisits ?? null,
      minTrainingSelectedActionShare: release.minTrainingSelectedActionShare ?? null,
      minTrainingExplorationShare: release.minTrainingExplorationShare ?? null,
      minTrainingValidationSamples: release.minTrainingValidationSamples ?? null,
      minTrainingValidationSides: release.minTrainingValidationSides ?? null,
      minTrainingValidationPhases: release.minTrainingValidationPhases ?? null,
      minTrainingValidationGroups: release.minTrainingValidationGroups ?? null,
      requiredTrainingValidationGroupBy: release.requiredTrainingValidationGroupBy ?? null,
      maxTrainingValidationValueMse: release.maxTrainingValidationValueMse ?? null,
      maxTrainingValidationValueCalibrationBias: release.maxTrainingValidationValueCalibrationBias ?? null,
      maxTrainingValidationPolicyCrossEntropy: release.maxTrainingValidationPolicyCrossEntropy ?? null,
      minTrainingValidationPolicyTopChoiceAccuracy: release.minTrainingValidationPolicyTopChoiceAccuracy ?? null,
      promotionVerdict: normalizePromotionVerdict(release.promotionVerdict),
      sideScoreDeltaPass: release.sideScoreDeltaPass !== false,
      scoreLowerBoundPass: release.scoreLowerBoundPass !== false,
      eloLowerBoundPass: release.eloLowerBoundPass !== false,
      drawRatePass: release.drawRatePass !== false,
      decisiveRatePass: release.decisiveRatePass !== false,
      arena: release.arena || null,
      decisionEvidence: release.decisionEvidence || null,
      errors: Number(release.errors || 0),
      activeGeneration: release.activeGeneration ?? null,
      evaluationSuite: release.evaluationSuite || emptyEvaluationSuiteEvidence(),
      referenceArena: release.referenceArena || emptyReferenceArenaEvidence(),
      sideScores: release.sideScores || [],
      ...(options.runtime
        ? {
          runtime: {
            schema: "zizi-el-alamein-alpha-runtime-install-v1",
            target: typeof options.runtime.target === "string" ? options.runtime.target : null,
            modelFile: typeof options.runtime.modelFile === "string" ? options.runtime.modelFile : null,
            installApproved: Boolean(options.runtime.installApproved),
          },
        }
        : {}),
    },
  };
}

export function classifyReleaseSource(artifact) {
  if (!artifact || typeof artifact !== "object") {
    return releaseError("invalid_release_artifact");
  }
  if (artifact.schema === "zizi-el-alamein-alpha-ladder-v1") {
    const activeGeneration = artifact.activeGeneration ?? null;
    const activeCycle = activeLadderCycle(artifact, activeGeneration);
    return {
      ok: true,
      sourceSchema: artifact.schema,
      modelSource: artifact.activeModel,
      promoted: Boolean(artifact.promotions > 0 && activeGeneration !== null),
      requiresPromotion: true,
      candidateScore: finiteOrNull(activeCycle?.evaluation?.candidateScore),
      promotionThreshold: finiteOrNull(activeCycle?.evaluation?.promotionThreshold)
        ?? maxPromotionThreshold(artifact.cycles || []),
      minSideScore: optionalFiniteOrNull(activeCycle?.evaluation?.minSideScore),
      minSideScoreDelta: optionalFiniteOrNull(activeCycle?.evaluation?.minSideScoreDelta),
      minScoreLowerBound: optionalFiniteOrNull(activeCycle?.evaluation?.minScoreLowerBound),
      minEloLowerBound: optionalFiniteOrNull(activeCycle?.evaluation?.minEloLowerBound),
      maxArenaDrawRate: optionalFiniteOrNull(activeCycle?.evaluation?.maxArenaDrawRate),
      minArenaDecisiveRate: optionalFiniteOrNull(activeCycle?.evaluation?.minArenaDecisiveRate),
      minAnalyzedActions: null,
      minAverageRootVisits: null,
      minCandidateAnalyzedActions: null,
      minCandidateAverageRootVisits: null,
      minCandidateDecisionActionTypes: null,
      minCandidateDecisionAverageRecommendationConfidence: null,
      maxCandidateDecisionAverageRecommendationUncertainty: null,
      minCandidateDecisionSelectedActionShare: null,
      minDecisionAverageRecommendationConfidence: null,
      maxDecisionAverageRecommendationUncertainty: null,
      minDecisionSelectedActionShare: null,
      minDecisionActionTypes: null,
      maxDecisionFallbackRate: null,
      minReferenceArenaScore: null,
      minReferenceArenaReferences: null,
      maxReferenceArenaErrors: null,
      promotionVerdict: normalizePromotionVerdict(activeCycle?.evaluation?.promotionVerdict),
      sideScoreDeltaPass: activeCycle?.evaluation?.sideScoreDeltaPass,
      scoreLowerBoundPass: activeCycle?.evaluation?.scoreLowerBoundPass,
      eloLowerBoundPass: activeCycle?.evaluation?.eloLowerBoundPass,
      drawRatePass: activeCycle?.evaluation?.drawRatePass,
      decisiveRatePass: activeCycle?.evaluation?.decisiveRatePass,
      arena: normalizeReleaseArena(activeCycle?.evaluation?.arena),
      decisionEvidence: normalizeDecisionEvidence(activeCycle?.evaluation?.decisionEvidence),
      errors: ladderErrors(artifact),
      activeGeneration,
      evaluationSuite: evaluationSuiteEvidence(activeCycle?.evaluation),
      referenceArena: normalizeReferenceArena(activeCycle?.referenceArena),
      sideScores: normalizeReleaseSideScores(activeCycle?.evaluation?.sideScores),
      trainingProfile: normalizeAlphaTrainingProfileEvidence(activeCycle?.trainingProfile)
        || normalizeAlphaTrainingProfileEvidence(artifact.trainingProfile),
    };
  }
  if (artifact.schema === "zizi-el-alamein-alpha-cycle-v1") {
    return {
      ok: true,
      sourceSchema: artifact.schema,
      modelSource: artifact.candidateModel,
      promoted: Boolean(artifact.promoted),
      requiresPromotion: true,
      candidateScore: finiteOrNull(artifact.evaluation?.candidateScore),
      promotionThreshold: finiteOrNull(artifact.evaluation?.promotionThreshold),
      minSideScore: optionalFiniteOrNull(artifact.evaluation?.minSideScore),
      minSideScoreDelta: optionalFiniteOrNull(artifact.evaluation?.minSideScoreDelta),
      minScoreLowerBound: optionalFiniteOrNull(artifact.evaluation?.minScoreLowerBound),
      minEloLowerBound: optionalFiniteOrNull(artifact.evaluation?.minEloLowerBound),
      maxArenaDrawRate: optionalFiniteOrNull(artifact.evaluation?.maxArenaDrawRate),
      minArenaDecisiveRate: optionalFiniteOrNull(artifact.evaluation?.minArenaDecisiveRate),
      minAnalyzedActions: null,
      minAverageRootVisits: null,
      minCandidateAnalyzedActions: null,
      minCandidateAverageRootVisits: null,
      minCandidateDecisionActionTypes: null,
      minCandidateDecisionAverageRecommendationConfidence: null,
      maxCandidateDecisionAverageRecommendationUncertainty: null,
      minCandidateDecisionSelectedActionShare: null,
      minDecisionAverageRecommendationConfidence: null,
      maxDecisionAverageRecommendationUncertainty: null,
      minDecisionSelectedActionShare: null,
      minDecisionActionTypes: null,
      maxDecisionFallbackRate: null,
      minReferenceArenaScore: null,
      minReferenceArenaReferences: null,
      maxReferenceArenaErrors: null,
      promotionVerdict: normalizePromotionVerdict(artifact.evaluation?.promotionVerdict),
      sideScoreDeltaPass: artifact.evaluation?.sideScoreDeltaPass,
      scoreLowerBoundPass: artifact.evaluation?.scoreLowerBoundPass,
      eloLowerBoundPass: artifact.evaluation?.eloLowerBoundPass,
      drawRatePass: artifact.evaluation?.drawRatePass,
      decisiveRatePass: artifact.evaluation?.decisiveRatePass,
      arena: normalizeReleaseArena(artifact.evaluation?.arena),
      decisionEvidence: normalizeDecisionEvidence(artifact.evaluation?.decisionEvidence),
      errors: Number(artifact.evaluation?.errors || 0)
        + Number(artifact.iteration?.selfPlay?.errorCount || 0)
        + Number(artifact.iteration?.reanalysisSummary?.errors || 0),
      activeGeneration: null,
      evaluationSuite: evaluationSuiteEvidence(artifact.evaluation),
      referenceArena: normalizeReferenceArena(artifact.referenceArena),
      sideScores: normalizeReleaseSideScores(artifact.evaluation?.sideScores),
      trainingProfile: normalizeAlphaTrainingProfileEvidence(artifact.trainingProfile),
    };
  }
  if (artifact.schema === "zizi-el-alamein-alpha-model-v1") {
    return {
      ok: true,
      sourceSchema: artifact.schema,
      modelSource: artifact,
      promoted: false,
      requiresPromotion: false,
      candidateScore: null,
      promotionThreshold: null,
      minSideScore: null,
      minSideScoreDelta: null,
      minScoreLowerBound: null,
      minEloLowerBound: null,
      maxArenaDrawRate: null,
      minArenaDecisiveRate: null,
      minAnalyzedActions: null,
      minAverageRootVisits: null,
      minCandidateAnalyzedActions: null,
      minCandidateAverageRootVisits: null,
      minCandidateDecisionActionTypes: null,
      minCandidateDecisionAverageRecommendationConfidence: null,
      maxCandidateDecisionAverageRecommendationUncertainty: null,
      minCandidateDecisionSelectedActionShare: null,
      minDecisionAverageRecommendationConfidence: null,
      maxDecisionAverageRecommendationUncertainty: null,
      minDecisionSelectedActionShare: null,
      minDecisionActionTypes: null,
      maxDecisionFallbackRate: null,
      minReferenceArenaScore: null,
      minReferenceArenaReferences: null,
      maxReferenceArenaErrors: null,
      promotionVerdict: normalizePromotionVerdict(artifact.release?.promotionVerdict),
      sideScoreDeltaPass: true,
      scoreLowerBoundPass: true,
      eloLowerBoundPass: true,
      drawRatePass: true,
      decisiveRatePass: true,
      arena: null,
      decisionEvidence: null,
      errors: 0,
      activeGeneration: null,
      evaluationSuite: emptyEvaluationSuiteEvidence(),
      referenceArena: emptyReferenceArenaEvidence(),
      sideScores: [],
      trainingProfile: normalizeAlphaTrainingProfileEvidence(artifact.release?.trainingProfile),
    };
  }
  return releaseError("unsupported_release_artifact");
}

function releaseError(reason, source = {}, model = null) {
  return {
    ok: false,
    reason,
    sourceSchema: source.sourceSchema || null,
    promoted: Boolean(source.promoted),
    candidateScore: source.candidateScore ?? null,
    promotionThreshold: source.promotionThreshold ?? null,
    minSideScore: source.minSideScore ?? null,
    minSideScoreDelta: source.minSideScoreDelta ?? null,
    minScoreLowerBound: source.minScoreLowerBound ?? null,
    minEloLowerBound: source.minEloLowerBound ?? null,
    maxArenaDrawRate: source.maxArenaDrawRate ?? null,
    minArenaDecisiveRate: source.minArenaDecisiveRate ?? null,
    minAnalyzedActions: source.minAnalyzedActions ?? null,
    minAverageRootVisits: source.minAverageRootVisits ?? null,
    minCandidateAnalyzedActions: source.minCandidateAnalyzedActions ?? null,
    minCandidateAverageRootVisits: source.minCandidateAverageRootVisits ?? null,
    minCandidateDecisionActionTypes: source.minCandidateDecisionActionTypes ?? null,
    minCandidateDecisionAverageRecommendationConfidence: source.minCandidateDecisionAverageRecommendationConfidence ?? null,
    maxCandidateDecisionAverageRecommendationUncertainty: source.maxCandidateDecisionAverageRecommendationUncertainty ?? null,
    minCandidateDecisionSelectedActionShare: source.minCandidateDecisionSelectedActionShare ?? null,
    minDecisionAverageRecommendationConfidence: source.minDecisionAverageRecommendationConfidence ?? null,
    maxDecisionAverageRecommendationUncertainty: source.maxDecisionAverageRecommendationUncertainty ?? null,
    minDecisionSelectedActionShare: source.minDecisionSelectedActionShare ?? null,
    minDecisionActionTypes: source.minDecisionActionTypes ?? null,
    maxDecisionFallbackRate: source.maxDecisionFallbackRate ?? null,
    minReferenceArenaScore: source.minReferenceArenaScore ?? null,
    minReferenceArenaReferences: source.minReferenceArenaReferences ?? null,
    maxReferenceArenaErrors: source.maxReferenceArenaErrors ?? null,
    minChallengeAverageRecommendationUncertainty: source.minChallengeAverageRecommendationUncertainty ?? null,
    minContestedChallengePositions: source.minContestedChallengePositions ?? null,
    minChallengeAverageRuntimeRisk: source.minChallengeAverageRuntimeRisk ?? null,
    minRuntimeRiskChallengePositions: source.minRuntimeRiskChallengePositions ?? null,
    minEvaluationPhases: source.minEvaluationPhases ?? null,
    requiredTrainingProfile: source.requiredTrainingProfile ?? null,
    trainingProfile: normalizeAlphaTrainingProfileEvidence(source.trainingProfile),
    minTrainingSamples: source.minTrainingSamples ?? null,
    minTrainingValueSamples: source.minTrainingValueSamples ?? null,
    minTrainingOutcomeClasses: source.minTrainingOutcomeClasses ?? null,
    minTrainingPolicyRows: source.minTrainingPolicyRows ?? null,
    minTrainingPolicyActionTypes: source.minTrainingPolicyActionTypes ?? null,
    minTrainingUniqueStateHashes: source.minTrainingUniqueStateHashes ?? null,
    maxTrainingDuplicateStateRate: source.maxTrainingDuplicateStateRate ?? null,
    minTrainingSides: source.minTrainingSides ?? null,
    minTrainingSources: source.minTrainingSources ?? null,
    minTrainingReanalysisSamples: source.minTrainingReanalysisSamples ?? null,
    minTrainingStateSnapshots: source.minTrainingStateSnapshots ?? null,
    minTrainingAverageRootVisits: source.minTrainingAverageRootVisits ?? null,
    minTrainingSelectedActionShare: source.minTrainingSelectedActionShare ?? null,
    minTrainingExplorationShare: source.minTrainingExplorationShare ?? null,
    minTrainingValidationSamples: source.minTrainingValidationSamples ?? null,
    minTrainingValidationSides: source.minTrainingValidationSides ?? null,
    minTrainingValidationPhases: source.minTrainingValidationPhases ?? null,
    minTrainingValidationGroups: source.minTrainingValidationGroups ?? null,
    requiredTrainingValidationGroupBy: source.requiredTrainingValidationGroupBy ?? null,
    maxTrainingValidationValueMse: source.maxTrainingValidationValueMse ?? null,
    maxTrainingValidationValueCalibrationBias: source.maxTrainingValidationValueCalibrationBias ?? null,
    maxTrainingValidationPolicyCrossEntropy: source.maxTrainingValidationPolicyCrossEntropy ?? null,
    minTrainingValidationPolicyTopChoiceAccuracy: source.minTrainingValidationPolicyTopChoiceAccuracy ?? null,
    promotionVerdict: normalizePromotionVerdict(source.promotionVerdict),
    sideScoreDeltaPass: source.sideScoreDeltaPass !== false,
    scoreLowerBoundPass: source.scoreLowerBoundPass !== false,
    eloLowerBoundPass: source.eloLowerBoundPass !== false,
    drawRatePass: source.drawRatePass !== false,
    decisiveRatePass: source.decisiveRatePass !== false,
    arena: normalizeReleaseArena(source.arena),
    decisionEvidence: normalizeDecisionEvidence(source.decisionEvidence),
    errors: Number(source.errors || 0),
    activeGeneration: source.activeGeneration ?? null,
    evaluationSuite: source.evaluationSuite || emptyEvaluationSuiteEvidence(),
    referenceArena: source.referenceArena || emptyReferenceArenaEvidence(),
    sideScores: source.sideScores || [],
    model,
    metadata: model ? alphaModelMetadata(model) : null,
  };
}

function activeLadderCycle(ladder, activeGeneration) {
  const cycles = ladder.cycles || [];
  const generation = Number(activeGeneration);
  return cycles.find((entry) => Number(entry.generation) === generation)?.cycle || null;
}

function evaluationSuiteEvidence(evaluation) {
  const entries = evaluation?.evaluationSuite?.entries || [];
  if (!Array.isArray(entries) || !entries.length) return emptyEvaluationSuiteEvidence();
  const sides = new Set(entries.map((entry) => entry.candidateSide).filter((side) => side === "axis" || side === "allied"));
  const seeds = new Set(entries.map((entry) => Number(entry.seed)).filter(Number.isFinite));
  const fixedPositions = entries.filter((entry) => entry.initialState).length;
  const challengePositions = entries.filter((entry) => entry.sourceStateHash || entry.sourceReplay).length;
  const challengeQuality = challengeQualityEvidence(entries);
  const phaseCoverage = phaseCoverageEvidence(entries);
  return {
    schema: "zizi-el-alamein-alpha-release-suite-evidence-v1",
    explicit: Boolean(evaluation.evaluationSuite.explicit),
    games: entries.length,
    sides: sides.size,
    seeds: seeds.size,
    phaseCoverage,
    fixedPositions,
    challengePositions,
    challengeQuality,
    challengeSelection: normalizeChallengeSelection(evaluation.evaluationSuite.challengeSelection),
  };
}

function emptyEvaluationSuiteEvidence() {
  return {
    schema: "zizi-el-alamein-alpha-release-suite-evidence-v1",
    explicit: false,
    games: 0,
    sides: 0,
    seeds: 0,
    phaseCoverage: emptyPhaseCoverageEvidence(),
    fixedPositions: 0,
    challengePositions: 0,
    challengeQuality: null,
    challengeSelection: null,
  };
}

function phaseCoverageEvidence(entries) {
  const counts = countBy((entries || []).map(phaseIdForSuiteEntry).filter(Boolean), (phaseId) => phaseId);
  const ids = Object.keys(counts).sort();
  return {
    schema: "zizi-el-alamein-alpha-release-phase-coverage-v1",
    phases: ids.length,
    ids,
    counts: Object.fromEntries(ids.map((id) => [id, counts[id]])),
  };
}

function emptyPhaseCoverageEvidence() {
  return {
    schema: "zizi-el-alamein-alpha-release-phase-coverage-v1",
    phases: 0,
    ids: [],
    counts: {},
  };
}

function phaseIdForSuiteEntry(entry) {
  const phaseId = entry?.sourcePhaseId
    || entry?.phaseId
    || entry?.initialState?.phaseId
    || entry?.initialState?.currentPhaseId
    || entry?.initialState?.phase?.id;
  return typeof phaseId === "string" && phaseId ? phaseId : null;
}

function emptyReferenceArenaEvidence() {
  return {
    schema: "zizi-el-alamein-alpha-reference-arena-evidence-v1",
    references: 0,
    scoredReferences: 0,
    minCandidateScore: null,
    averageCandidateScore: null,
    errors: 0,
    entries: [],
  };
}

function normalizeReferenceArena(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return emptyReferenceArenaEvidence();
  const entries = Array.isArray(value.entries)
    ? value.entries.map(normalizeReferenceArenaEntry).filter(Boolean)
    : [];
  return {
    schema: "zizi-el-alamein-alpha-reference-arena-evidence-v1",
    references: finiteNumber(value.references, entries.length),
    scoredReferences: finiteNumber(value.scoredReferences, entries.filter((entry) => entry.scoredGames > 0).length),
    minCandidateScore: finiteOrNull(value.minCandidateScore),
    averageCandidateScore: finiteOrNull(value.averageCandidateScore),
    errors: finiteNumber(value.errors, entries.reduce((sum, entry) => sum + Number(entry.errors || 0), 0)),
    entries,
  };
}

function normalizeReferenceArenaEntry(value, index = 0) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const evaluation = value.evaluation || {};
  return {
    schema: "zizi-el-alamein-alpha-reference-arena-entry-evidence-v1",
    index: finiteNumber(value.index, index),
    label: typeof value.label === "string" && value.label ? value.label : `reference-${index + 1}`,
    source: typeof value.source === "string" ? value.source : null,
    modelPresent: Boolean(value.modelPresent),
    games: finiteNumber(evaluation.games, 0),
    scoredGames: finiteNumber(evaluation.scoredGames, 0),
    candidateScore: finiteOrNull(evaluation.candidateScore),
    errors: finiteNumber(evaluation.errors, 0),
    arena: normalizeReleaseArena(evaluation.arena),
  };
}

function challengeQualityEvidence(entries) {
  const challengeEntries = (entries || []).filter((entry) => entry.sourceStateHash || entry.sourceReplay);
  if (!challengeEntries.length) return null;
  return {
    schema: "zizi-el-alamein-alpha-challenge-quality-v1",
    samples: challengeEntries.length,
    averagePriority: averageFinite(challengeEntries.map((entry) => entry.sourcePriority)),
    averageRuntimeRisk: averageFinite(challengeEntries.map(challengeRuntimeRisk)),
    runtimeRiskPositions: challengeEntries.filter((entry) => challengeRuntimeRisk(entry) > 0).length,
    runtimeRecommendations: challengeEntries.filter((entry) => entry.sourceAlphaRecommendation).length,
    runtimeRejectedRecommendations: challengeEntries.filter((entry) => entry.sourceAlphaRecommendation?.ok === false).length,
    runtimeCandidateFallbacks: challengeEntries.filter((entry) => {
      const source = entry.sourceAlphaRecommendation?.selectedSource;
      return source && source !== "direct";
    }).length,
    runtimeIllegalCandidatesSkipped: challengeEntries.reduce((sum, entry) => (
      sum + Math.max(0, Number(entry.sourceAlphaRecommendation?.illegalCandidateCount || 0))
    ), 0),
    averagePolicyEntropy: averageFinite(challengeEntries.map((entry) => entry.sourcePolicyEntropy)),
    averageRecommendationConfidence: averageFinite(challengeEntries.map((entry) => entry.sourceRecommendationConfidence)),
    averageRecommendationVisitMargin: averageFinite(challengeEntries.map((entry) => entry.sourceRecommendationVisitMargin)),
    averageRecommendationQMargin: averageFinite(challengeEntries.map((entry) => entry.sourceRecommendationQMargin)),
    averageRecommendationUncertainty: averageFinite(challengeEntries.map(challengeRecommendationUncertainty)),
    averageTemperature: averageFinite(challengeEntries.map((entry) => entry.sourceTemperature)),
    averageRootNoiseWeight: averageFinite(challengeEntries.map((entry) => entry.sourceRootNoiseWeight)),
    averageSearchIterations: averageFinite(challengeEntries.map((entry) => entry.sourceSearchIterations)),
    averageRootVisits: averageFinite(challengeEntries.map((entry) => entry.sourceRootVisits)),
    recommendationLabels: countBy(challengeEntries, (entry) => entry.sourceRecommendationLabel || "unknown"),
    selectionModes: countBy(challengeEntries, (entry) => entry.sourceSelectionMode || "unknown"),
  };
}

function challengeRecommendationUncertainty(entry) {
  const confidence = Number(entry?.sourceRecommendationConfidence);
  if (Number.isFinite(confidence)) return 1 - clamp01(confidence);
  const visitMargin = Number(entry?.sourceRecommendationVisitMargin);
  const entropy = Number(entry?.sourcePolicyEntropy);
  if (Number.isFinite(visitMargin) && Number.isFinite(entropy)) {
    return ((1 - clamp01(Math.abs(visitMargin))) + clamp01(entropy)) / 2;
  }
  if (Number.isFinite(visitMargin)) return 1 - clamp01(Math.abs(visitMargin));
  if (Number.isFinite(entropy)) return clamp01(entropy);
  return null;
}

function challengeRuntimeRisk(entry) {
  const recommendation = entry?.sourceAlphaRecommendation || {};
  if (recommendation.ok === false || entry?.sourceSelectionMode === "rejected") return 1;
  let risk = 0;
  const selectedSource = recommendation.selectedSource || null;
  if (selectedSource && selectedSource !== "direct") risk += 0.35;
  risk += Math.min(0.3, Math.max(0, Number(recommendation.illegalCandidateCount || 0)) * 0.1);
  const confidence = Number(entry?.sourceRecommendationConfidence);
  if (Number.isFinite(confidence)) risk += 0.25 * (1 - clamp01(confidence));
  if (recommendation.runtimeActionMode === "direct" || recommendation.runtimeAnalysisMode === "direct") risk += 0.1;
  return round(clamp01(risk), 6);
}

function challengeQualityGate(quality, options = {}) {
  const minAverageUncertainty = optionalFiniteOrNull(options.minChallengeAverageRecommendationUncertainty);
  const minContestedPositions = optionalFiniteOrNull(options.minContestedChallengePositions);
  const minAverageRuntimeRisk = optionalFiniteOrNull(options.minChallengeAverageRuntimeRisk);
  const minRuntimeRiskPositions = optionalFiniteOrNull(options.minRuntimeRiskChallengePositions);
  if (
    minAverageUncertainty === null
    && minContestedPositions === null
    && minAverageRuntimeRisk === null
    && minRuntimeRiskPositions === null
  ) return { ok: true, reason: null };
  if (!quality) return { ok: false, reason: "missing_challenge_quality_evidence" };
  if (
    minAverageUncertainty !== null
    && (
      quality.averageRecommendationUncertainty === null
      || quality.averageRecommendationUncertainty === undefined
      || Number(quality.averageRecommendationUncertainty) < minAverageUncertainty
    )
  ) {
    return { ok: false, reason: "challenge_average_uncertainty_too_low" };
  }
  if (
    minContestedPositions !== null
    && Number(quality.recommendationLabels?.contested || 0) < minContestedPositions
  ) {
    return { ok: false, reason: "challenge_contested_positions_too_few" };
  }
  if (
    minAverageRuntimeRisk !== null
    && (
      quality.averageRuntimeRisk === null
      || quality.averageRuntimeRisk === undefined
      || Number(quality.averageRuntimeRisk) < minAverageRuntimeRisk
    )
  ) {
    return { ok: false, reason: "challenge_average_runtime_risk_too_low" };
  }
  if (
    minRuntimeRiskPositions !== null
    && Number(quality.runtimeRiskPositions || 0) < minRuntimeRiskPositions
  ) {
    return { ok: false, reason: "challenge_runtime_risk_positions_too_few" };
  }
  return { ok: true, reason: null };
}

function trainingProfileGate(profile, requiredProfile = null) {
  const required = optionalStringOrNull(requiredProfile);
  if (required === null) return { ok: true, reason: null };
  const evidence = normalizeAlphaTrainingProfileEvidence(profile);
  if (!evidence) return { ok: false, reason: "missing_training_profile_evidence" };
  if (evidence.profile !== required) return { ok: false, reason: "training_profile_mismatch" };
  return { ok: true, reason: null };
}

function referenceArenaGate(arena, options = {}) {
  const minScore = optionalFiniteOrNull(options.minReferenceArenaScore);
  const minReferences = optionalFiniteOrNull(options.minReferenceArenaReferences);
  const maxErrors = optionalFiniteOrNull(options.maxReferenceArenaErrors);
  if (minScore === null && minReferences === null && maxErrors === null) return { ok: true, reason: null };
  const evidence = normalizeReferenceArena(arena);
  if (!evidence.references) return { ok: false, reason: "missing_reference_arena_evidence" };
  if (minReferences !== null && Number(evidence.scoredReferences || 0) < minReferences) {
    return { ok: false, reason: "reference_arena_too_few_references" };
  }
  if (maxErrors !== null && Number(evidence.errors || 0) > maxErrors) {
    return { ok: false, reason: "reference_arena_errors_exceed_limit" };
  }
  if (
    minScore !== null
    && (
      evidence.minCandidateScore === null
      || evidence.minCandidateScore === undefined
      || Number(evidence.minCandidateScore) < minScore
    )
  ) {
    return { ok: false, reason: "reference_arena_score_below_threshold" };
  }
  return { ok: true, reason: null };
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

function sideScoreGate(sideScores, minSideScore) {
  const scores = normalizeReleaseSideScores(sideScores);
  if (!scores.length) return { ok: false, reason: "missing_side_score_evidence" };
  const weakSide = scores.find((score) => (
    Number(score.scoredGames || 0) < 1 || Number(score.candidateScore || 0) < minSideScore
  ));
  if (weakSide) return { ok: false, reason: "candidate_side_score_below_threshold" };
  return { ok: true, reason: null };
}

function sideScoreDeltaGate(sideScores, minSideScoreDelta) {
  if (!Array.isArray(sideScores) || !sideScores.length) {
    return { ok: false, reason: "missing_side_score_evidence" };
  }
  const weakSide = sideScores.find((score) => (
    Number(score.scoredGames || 0) < 1
    || optionalFiniteOrNull(score.scoreDelta) === null
    || Number(score.scoreDelta) < minSideScoreDelta
  ));
  return weakSide
    ? { ok: false, reason: "candidate_side_score_delta_below_threshold" }
    : { ok: true, reason: null };
}

function scoreLowerBoundGate(arena, minScoreLowerBound, sourcePass = true) {
  const metrics = normalizeReleaseArena(arena);
  if (!metrics?.scoreInterval95) return { ok: false, reason: "missing_arena_evidence" };
  if (sourcePass === false || Number(metrics.scoreInterval95.low || 0) < minScoreLowerBound) {
    return { ok: false, reason: "candidate_score_lower_bound_below_threshold" };
  }
  return { ok: true, reason: null };
}

function eloLowerBoundGate(arena, minEloLowerBound, sourcePass = true) {
  const metrics = normalizeReleaseArena(arena);
  if (!metrics?.eloDiffInterval95) return { ok: false, reason: "missing_arena_evidence" };
  if (sourcePass === false || Number(metrics.eloDiffInterval95.low || 0) < minEloLowerBound) {
    return { ok: false, reason: "candidate_elo_lower_bound_below_threshold" };
  }
  return { ok: true, reason: null };
}

function arenaRateGate(arena, options = {}) {
  const maxArenaDrawRate = optionalFiniteOrNull(options.maxArenaDrawRate);
  const minArenaDecisiveRate = optionalFiniteOrNull(options.minArenaDecisiveRate);
  if (maxArenaDrawRate === null && minArenaDecisiveRate === null) {
    return { ok: true, reason: null, drawRatePass: true, decisiveRatePass: true };
  }
  const metrics = normalizeReleaseArena(arena);
  if (!metrics || Number(metrics.scoredGames || 0) < 1) {
    return { ok: false, reason: "missing_arena_evidence", drawRatePass: false, decisiveRatePass: false };
  }
  const drawRatePass = maxArenaDrawRate === null
    ? true
    : options.drawRatePass !== false && Number(metrics.drawRate || 0) <= maxArenaDrawRate;
  if (!drawRatePass) {
    return { ok: false, reason: "arena_draw_rate_too_high", drawRatePass, decisiveRatePass: true };
  }
  const decisiveRatePass = minArenaDecisiveRate === null
    ? true
    : options.decisiveRatePass !== false && Number(metrics.decisiveRate || 0) >= minArenaDecisiveRate;
  if (!decisiveRatePass) {
    return { ok: false, reason: "arena_decisive_rate_too_low", drawRatePass, decisiveRatePass };
  }
  return { ok: true, reason: null, drawRatePass, decisiveRatePass };
}

function decisionEvidenceGate(decisionEvidence, options = {}) {
  const evidence = normalizeDecisionEvidence(decisionEvidence);
  if (!evidence) return { ok: false, reason: "missing_decision_evidence" };
  const minAnalyzedActions = optionalFiniteOrNull(options.minAnalyzedActions);
  const minAverageRootVisits = optionalFiniteOrNull(options.minAverageRootVisits);
  const minConfidence = optionalFiniteOrNull(options.minDecisionAverageRecommendationConfidence);
  const maxUncertainty = optionalFiniteOrNull(options.maxDecisionAverageRecommendationUncertainty);
  const minSelectedActionShare = optionalFiniteOrNull(options.minDecisionSelectedActionShare);
  const minActionTypes = optionalFiniteOrNull(options.minDecisionActionTypes);
  const maxFallbackRate = optionalFiniteOrNull(options.maxDecisionFallbackRate);
  if (minAnalyzedActions !== null && Number(evidence.analyzedActions || 0) < minAnalyzedActions) {
    return { ok: false, reason: "decision_evidence_too_few_analyzed_actions" };
  }
  if (minAverageRootVisits !== null && Number(evidence.averageRootVisits || 0) < minAverageRootVisits) {
    return { ok: false, reason: "decision_evidence_root_visits_too_low" };
  }
  if (minActionTypes !== null && countActionTypes(evidence.actionTypes) < minActionTypes) {
    return { ok: false, reason: "decision_evidence_action_coverage_too_narrow" };
  }
  if (
    minConfidence !== null
    && (
      evidence.averageRecommendationConfidence === null
      || evidence.averageRecommendationConfidence === undefined
      || Number(evidence.averageRecommendationConfidence) < minConfidence
    )
  ) {
    return { ok: false, reason: "decision_evidence_confidence_too_low" };
  }
  if (
    maxUncertainty !== null
    && (
      evidence.averageRecommendationUncertainty === null
      || evidence.averageRecommendationUncertainty === undefined
      || Number(evidence.averageRecommendationUncertainty) > maxUncertainty
    )
  ) {
    return { ok: false, reason: "decision_evidence_uncertainty_too_high" };
  }
  if (
    minSelectedActionShare !== null
    && (
      evidence.selectedActionShare === null
      || evidence.selectedActionShare === undefined
      || Number(evidence.selectedActionShare) < minSelectedActionShare
    )
  ) {
    return { ok: false, reason: "decision_evidence_selected_share_too_low" };
  }
  if (maxFallbackRate !== null) {
    const fallbackRate = decisionFallbackRate(evidence);
    if (fallbackRate === null) return { ok: false, reason: "decision_evidence_too_few_decisions" };
    if (fallbackRate > maxFallbackRate) return { ok: false, reason: "decision_evidence_fallback_rate_too_high" };
  }
  return { ok: true, reason: null };
}

function candidateDecisionEvidenceGate(decisionEvidence, options = {}) {
  const evidence = normalizeDecisionEvidence(decisionEvidence);
  if (!evidence) return { ok: false, reason: "missing_decision_evidence" };
  const candidate = evidence.roles?.candidate;
  if (!candidate || Number(candidate.actionCount || 0) < 1) {
    return { ok: false, reason: "missing_candidate_decision_evidence" };
  }
  const minAnalyzedActions = optionalFiniteOrNull(options.minCandidateAnalyzedActions);
  const minAverageRootVisits = optionalFiniteOrNull(options.minCandidateAverageRootVisits);
  const minActionTypes = optionalFiniteOrNull(options.minCandidateDecisionActionTypes);
  const minConfidence = optionalFiniteOrNull(options.minCandidateDecisionAverageRecommendationConfidence);
  const maxUncertainty = optionalFiniteOrNull(options.maxCandidateDecisionAverageRecommendationUncertainty);
  const minSelectedActionShare = optionalFiniteOrNull(options.minCandidateDecisionSelectedActionShare);
  if (minAnalyzedActions !== null && Number(candidate.analyzedActions || 0) < minAnalyzedActions) {
    return { ok: false, reason: "candidate_decision_evidence_too_few_analyzed_actions" };
  }
  if (minAverageRootVisits !== null && Number(candidate.averageRootVisits || 0) < minAverageRootVisits) {
    return { ok: false, reason: "candidate_decision_evidence_root_visits_too_low" };
  }
  if (minActionTypes !== null && countActionTypes(candidate.actionTypes) < minActionTypes) {
    return { ok: false, reason: "candidate_decision_evidence_action_coverage_too_narrow" };
  }
  if (
    minConfidence !== null
    && (
      candidate.averageRecommendationConfidence === null
      || candidate.averageRecommendationConfidence === undefined
      || Number(candidate.averageRecommendationConfidence) < minConfidence
    )
  ) {
    return { ok: false, reason: "candidate_decision_evidence_confidence_too_low" };
  }
  if (
    maxUncertainty !== null
    && (
      candidate.averageRecommendationUncertainty === null
      || candidate.averageRecommendationUncertainty === undefined
      || Number(candidate.averageRecommendationUncertainty) > maxUncertainty
    )
  ) {
    return { ok: false, reason: "candidate_decision_evidence_uncertainty_too_high" };
  }
  if (
    minSelectedActionShare !== null
    && (
      candidate.selectedActionShare === null
      || candidate.selectedActionShare === undefined
      || Number(candidate.selectedActionShare) < minSelectedActionShare
    )
  ) {
    return { ok: false, reason: "candidate_decision_evidence_selected_share_too_low" };
  }
  return { ok: true, reason: null };
}

function decisionFallbackRate(evidence) {
  const decisions = Number(evidence?.decisionCount ?? evidence?.actionCount ?? 0);
  if (!(decisions > 0)) return null;
  return Number(evidence?.fallbackActions || 0) / decisions;
}

function countActionTypes(actionTypes) {
  return Object.values(actionTypes || {}).filter((count) => Number(count) > 0).length;
}

function countPositiveKeys(value) {
  return Object.values(value || {}).filter((count) => Number(count) > 0).length;
}

function trainingDataGate(data, options = {}) {
  const thresholds = [
    options.minTrainingSamples,
    options.minTrainingValueSamples,
    options.minTrainingOutcomeClasses,
    options.minTrainingPolicyRows,
    options.minTrainingPolicyActionTypes,
    options.minTrainingUniqueStateHashes,
    options.maxTrainingDuplicateStateRate,
    options.minTrainingSides,
    options.minTrainingSources,
    options.minTrainingReanalysisSamples,
    options.minTrainingStateSnapshots,
    options.minTrainingAverageRootVisits,
    options.minTrainingSelectedActionShare,
    options.minTrainingExplorationShare,
  ];
  if (thresholds.every((value) => value === null)) return { ok: true, reason: null };
  if (!data) return { ok: false, reason: "missing_training_data_evidence" };
  if (options.minTrainingSamples !== null && Number(data.sampleCount || 0) < options.minTrainingSamples) {
    return { ok: false, reason: "training_samples_too_few" };
  }
  if (
    options.minTrainingValueSamples !== null
    && Number(data.valueSamples || 0) < options.minTrainingValueSamples
  ) {
    return { ok: false, reason: "training_value_samples_too_few" };
  }
  if (
    options.minTrainingOutcomeClasses !== null
    && Number(data.outcomeClassCount ?? countPositiveKeys(data.outcomeBuckets)) < options.minTrainingOutcomeClasses
  ) {
    return { ok: false, reason: "training_outcome_coverage_too_narrow" };
  }
  if (
    options.minTrainingPolicyRows !== null
    && Number(data.policyRows || 0) < options.minTrainingPolicyRows
  ) {
    return { ok: false, reason: "training_policy_rows_too_few" };
  }
  if (
    options.minTrainingPolicyActionTypes !== null
    && Number(data.policyActionTypeCount ?? countPositiveKeys(data.policyActionTypes)) < options.minTrainingPolicyActionTypes
  ) {
    return { ok: false, reason: "training_policy_action_coverage_too_narrow" };
  }
  if (
    options.minTrainingUniqueStateHashes !== null
    && Number(data.uniqueStateHashes || 0) < options.minTrainingUniqueStateHashes
  ) {
    return { ok: false, reason: "training_unique_states_too_few" };
  }
  if (
    options.maxTrainingDuplicateStateRate !== null
    && (
      data.duplicateStateRate === null
      || data.duplicateStateRate === undefined
      || Number(data.duplicateStateRate) > options.maxTrainingDuplicateStateRate
    )
  ) {
    return { ok: false, reason: "training_duplicate_state_rate_too_high" };
  }
  if (options.minTrainingSides !== null && Object.keys(data.sides || {}).length < options.minTrainingSides) {
    return { ok: false, reason: "training_side_coverage_too_narrow" };
  }
  if (options.minTrainingSources !== null && (data.sources || []).length < options.minTrainingSources) {
    return { ok: false, reason: "training_sources_too_few" };
  }
  if (
    options.minTrainingReanalysisSamples !== null
    && Number(data.reanalysisSamples || 0) < options.minTrainingReanalysisSamples
  ) {
    return { ok: false, reason: "training_reanalysis_samples_too_few" };
  }
  if (
    options.minTrainingStateSnapshots !== null
    && Number(data.samplesWithStateSnapshot || 0) < options.minTrainingStateSnapshots
  ) {
    return { ok: false, reason: "training_state_snapshots_too_few" };
  }
  if (
    options.minTrainingAverageRootVisits !== null
    && (data.averageRootVisits === null || Number(data.averageRootVisits || 0) < options.minTrainingAverageRootVisits)
  ) {
    return { ok: false, reason: "training_root_visits_too_low" };
  }
  if (
    options.minTrainingSelectedActionShare !== null
    && (
      data.averageSelectedVisitShare === null
      || data.averageSelectedVisitShare === undefined
      || Number(data.averageSelectedVisitShare || 0) < options.minTrainingSelectedActionShare
    )
  ) {
    return { ok: false, reason: "training_selected_action_share_too_low" };
  }
  const explorationShare = trainingExplorationShare(data);
  if (
    options.minTrainingExplorationShare !== null
    && (explorationShare === null || explorationShare < options.minTrainingExplorationShare)
  ) {
    return { ok: false, reason: "training_exploration_share_too_low" };
  }
  return { ok: true, reason: null };
}

function trainingExplorationShare(data) {
  const recorded = Number(data?.explorationDecisionShare);
  if (Number.isFinite(recorded)) return recorded;
  const decisions = Number(data?.samplesWithDecision || 0);
  if (!(decisions > 0)) return null;
  const modes = data?.selectionModes || {};
  const sampled = Number(data?.sampledDecisionCount ?? 0)
    || Number(modes.sampled || 0) + Number(modes.sampled_best || 0);
  const explored = Number(data?.explorationDecisionCount ?? sampled);
  if (!Number.isFinite(explored)) return null;
  return explored / decisions;
}

function trainingValidationGate(validation, options = {}) {
  const minSamples = options.minTrainingValidationSamples;
  const minSides = options.minTrainingValidationSides;
  const minPhases = options.minTrainingValidationPhases;
  const minGroups = options.minTrainingValidationGroups;
  const requiredGroupBy = optionalStringOrNull(options.requiredTrainingValidationGroupBy);
  const maxValueMse = options.maxTrainingValidationValueMse;
  const maxValueCalibrationBias = options.maxTrainingValidationValueCalibrationBias;
  const maxPolicyCrossEntropy = options.maxTrainingValidationPolicyCrossEntropy;
  const minPolicyTopChoiceAccuracy = options.minTrainingValidationPolicyTopChoiceAccuracy;
  if (requiredGroupBy !== null && !SUPPORTED_TRAINING_VALIDATION_GROUP_BY.includes(requiredGroupBy)) {
    return { ok: false, reason: "unsupported_training_validation_group_by" };
  }
  const trajectoryValidation = validation?.validationGroupBy === "trajectory";
  if (
    [
      minSamples,
      minSides,
      minPhases,
      minGroups,
      requiredGroupBy,
      maxValueMse,
      maxValueCalibrationBias,
      maxPolicyCrossEntropy,
      minPolicyTopChoiceAccuracy,
    ].every((value) => value === null)
    && !trajectoryValidation
  ) {
    return { ok: true, reason: null };
  }
  if (!validation) return { ok: false, reason: "missing_training_validation_evidence" };
  if (trajectoryValidation && validation.trajectoryOverlapCount === null) {
    return { ok: false, reason: "missing_training_validation_trajectory_overlap_evidence" };
  }
  if (trajectoryValidation && Number(validation.trajectoryOverlapCount) !== 0) {
    return { ok: false, reason: "training_validation_trajectory_overlap" };
  }
  if (trajectoryValidation && Number(validation.trainingTrajectories || 0) < 1) {
    return { ok: false, reason: "training_validation_training_trajectories_too_few" };
  }
  if (trajectoryValidation && Number(validation.validationTrajectories || 0) < 1) {
    return { ok: false, reason: "training_validation_validation_trajectories_too_few" };
  }
  if (minSamples !== null && Number(validation.sampleCount || 0) < minSamples) {
    return { ok: false, reason: "training_validation_samples_too_few" };
  }
  if (minSides !== null && countPositiveKeys(validation.sides) < minSides) {
    return { ok: false, reason: "training_validation_side_coverage_too_narrow" };
  }
  if (minPhases !== null && countPositiveKeys(validation.phases) < minPhases) {
    return { ok: false, reason: "training_validation_phase_coverage_too_narrow" };
  }
  if (minGroups !== null && Number(validation.validationGroups || 0) < minGroups) {
    return { ok: false, reason: "training_validation_groups_too_few" };
  }
  if (requiredGroupBy !== null && validation.validationGroupBy !== requiredGroupBy) {
    return { ok: false, reason: "training_validation_group_by_mismatch" };
  }
  if (
    maxValueMse !== null
    && (validation.value?.mse === null || validation.value?.mse === undefined || Number(validation.value.mse) > maxValueMse)
  ) {
    return { ok: false, reason: "training_validation_value_mse_too_high" };
  }
  if (
    maxValueCalibrationBias !== null
    && (
      validation.value?.calibrationBias === null
      || validation.value?.calibrationBias === undefined
      || Math.abs(Number(validation.value.calibrationBias)) > maxValueCalibrationBias
    )
  ) {
    return { ok: false, reason: "training_validation_value_calibration_bias_too_high" };
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
  if (
    minPolicyTopChoiceAccuracy !== null
    && (
      validation.policy?.topChoiceAccuracy === null
      || validation.policy?.topChoiceAccuracy === undefined
      || Number(validation.policy.topChoiceAccuracy) < minPolicyTopChoiceAccuracy
    )
  ) {
    return { ok: false, reason: "training_validation_policy_top_choice_accuracy_too_low" };
  }
  return { ok: true, reason: null };
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
      baselineScore: optionalFiniteOrNull(entry?.baselineScore),
      scoreDelta: optionalFiniteOrNull(entry?.scoreDelta),
      arena: normalizeReleaseArena(entry?.arena),
    }))
    .filter((entry) => entry.candidateSide);
}

function normalizePromotionVerdict(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const gates = Array.isArray(value.gates)
    ? value.gates
      .map(normalizePromotionGate)
      .filter(Boolean)
    : [];
  return {
    schema: typeof value.schema === "string" ? value.schema : "zizi-el-alamein-alpha-promotion-verdict-v1",
    ok: typeof value.ok === "boolean" ? value.ok : (gates.length > 0 && gates.every((gate) => gate.ok)),
    reason: typeof value.reason === "string" && value.reason ? value.reason : null,
    gates,
  };
}

function promotionVerdictGate(value) {
  const promotionVerdict = normalizePromotionVerdict(value);
  if (!promotionVerdict) {
    return { ok: true, reason: null, promotionVerdict: null };
  }
  const failedGate = promotionVerdict.gates.find((gate) => gate.ok === false);
  if (!promotionVerdict.ok || failedGate) {
    return {
      ok: false,
      reason: "promotion_verdict_failed",
      promotionVerdict,
    };
  }
  return {
    ok: true,
    reason: null,
    promotionVerdict,
  };
}

function normalizePromotionGate(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || typeof value.key !== "string") return null;
  return {
    key: value.key,
    ok: Boolean(value.ok),
    required: value.required !== false,
    actual: finiteOrNull(value.actual),
    threshold: finiteOrNull(value.threshold),
    comparison: typeof value.comparison === "string" && value.comparison ? value.comparison : "min",
    reason: typeof value.reason === "string" && value.reason ? value.reason : null,
  };
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
    averageRecommendationConfidence: finiteOrNull(value.averageRecommendationConfidence),
    averageRecommendationVisitMargin: finiteOrNull(value.averageRecommendationVisitMargin),
    averageRecommendationQMargin: finiteOrNull(value.averageRecommendationQMargin),
    averageRecommendationUncertainty: finiteOrNull(value.averageRecommendationUncertainty),
    averageSearchIterations: finiteOrNull(value.averageSearchIterations),
    averageRootVisits: finiteOrNull(value.averageRootVisits),
    selectedActionShare: finiteOrNull(value.selectedActionShare),
    actionTypes: countByObject(value.actionTypes),
    selectionModes: countByObject(value.selectionModes),
    recommendationLabels: countByObject(value.recommendationLabels),
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
      averageRecommendationConfidence: null,
      averageRecommendationVisitMargin: null,
      averageRecommendationQMargin: null,
      averageRecommendationUncertainty: null,
      averageSearchIterations: null,
      averageRootVisits: null,
      selectedActionShare: null,
      recommendationLabels: {},
      actionTypes: {},
    };
  }
  return {
    actionCount: finiteNumber(value.actionCount, 0),
    decisionCount: finiteNumber(value.decisionCount, 0),
    analyzedActions: finiteNumber(value.analyzedActions, 0),
    averagePolicyEntropy: finiteOrNull(value.averagePolicyEntropy),
    averageRecommendationConfidence: finiteOrNull(value.averageRecommendationConfidence),
    averageRecommendationVisitMargin: finiteOrNull(value.averageRecommendationVisitMargin),
    averageRecommendationQMargin: finiteOrNull(value.averageRecommendationQMargin),
    averageRecommendationUncertainty: finiteOrNull(value.averageRecommendationUncertainty),
    averageSearchIterations: finiteOrNull(value.averageSearchIterations),
    averageRootVisits: finiteOrNull(value.averageRootVisits),
    selectedActionShare: finiteOrNull(value.selectedActionShare),
    recommendationLabels: countByObject(value.recommendationLabels),
    actionTypes: countByObject(value.actionTypes),
  };
}

function ladderErrors(ladder) {
  return (ladder.cycles || []).reduce((sum, cycle) => sum + Number(cycle.summary?.errors || 0), 0);
}

function maxPromotionThreshold(cycles) {
  const thresholds = cycles
    .map((cycle) => finiteOrNull(cycle.cycle?.evaluation?.promotionThreshold))
    .filter((value) => value !== null);
  return thresholds.length ? Math.max(...thresholds) : null;
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function optionalFiniteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  return finiteOrNull(value);
}

function optionalStringOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  return String(value);
}

function finiteNumber(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function round(value, digits = 6) {
  const next = Number(value);
  if (!Number.isFinite(next)) return 0;
  return Number(next.toFixed(digits));
}

function averageFinite(values) {
  const clean = (values || []).map(Number).filter(Number.isFinite);
  if (!clean.length) return null;
  return Number((clean.reduce((sum, value) => sum + value, 0) / clean.length).toFixed(6));
}

function countBy(items, selector) {
  return (items || []).reduce((counts, item) => {
    const key = selector(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function countByObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .map(([key, raw]) => [String(key), Math.max(0, finiteNumber(raw, 0))])
      .filter(([, count]) => count > 0),
  );
}
