import {
  ALPHA_RELEASE_METADATA_SCHEMA,
  alphaModelMetadata,
  extractAlphaModelArtifact,
} from "./ai-alpha-model.js";

const DEFAULT_MIN_SCORE = 0.55;
const DEFAULT_MIN_SUITE_GAMES = 2;
const DEFAULT_MIN_SUITE_SIDES = 2;

export function createAlphaModelRelease(artifact, options = {}) {
  const minCandidateScore = Number(options.minCandidateScore ?? DEFAULT_MIN_SCORE);
  const maxErrors = Number(options.maxErrors ?? 0);
  const minSuiteGames = Math.max(0, Number(options.minSuiteGames ?? DEFAULT_MIN_SUITE_GAMES));
  const minSuiteSides = Math.max(0, Number(options.minSuiteSides ?? DEFAULT_MIN_SUITE_SIDES));
  const minFixedPositions = Math.max(0, Number(options.minFixedPositions ?? 0));
  const minChallengePositions = Math.max(0, Number(options.minChallengePositions ?? 0));
  const requireEvaluationSuite = options.requireEvaluationSuite !== false;
  const allowDirectModel = Boolean(options.allowDirectModel);
  const source = classifyReleaseSource(artifact);
  if (!source.ok) return source;
  const minSideScore = optionalFiniteOrNull(options.minSideScore ?? source.minSideScore);
  const minScoreLowerBound = optionalFiniteOrNull(options.minScoreLowerBound ?? source.minScoreLowerBound);
  const minAnalyzedActions = optionalFiniteOrNull(options.minAnalyzedActions ?? source.minAnalyzedActions);
  const minAverageRootVisits = optionalFiniteOrNull(options.minAverageRootVisits ?? source.minAverageRootVisits);
  const trainingThresholds = {
    minTrainingSamples: optionalFiniteOrNull(options.minTrainingSamples),
    minTrainingSides: optionalFiniteOrNull(options.minTrainingSides),
    minTrainingSources: optionalFiniteOrNull(options.minTrainingSources),
    minTrainingReanalysisSamples: optionalFiniteOrNull(options.minTrainingReanalysisSamples),
    minTrainingStateSnapshots: optionalFiniteOrNull(options.minTrainingStateSnapshots),
    minTrainingAverageRootVisits: optionalFiniteOrNull(options.minTrainingAverageRootVisits),
    minTrainingValidationSamples: optionalFiniteOrNull(options.minTrainingValidationSamples),
    maxTrainingValidationValueMse: optionalFiniteOrNull(options.maxTrainingValidationValueMse),
    maxTrainingValidationPolicyCrossEntropy: optionalFiniteOrNull(options.maxTrainingValidationPolicyCrossEntropy),
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

  if (source.candidateScore !== null && source.candidateScore < minCandidateScore) {
    return releaseError("candidate_score_below_threshold", source, model);
  }

  if (minSideScore !== null) {
    const sideGate = sideScoreGate(source.sideScores, minSideScore);
    if (!sideGate.ok) return releaseError(sideGate.reason, source, model);
  }

  if (minScoreLowerBound !== null) {
    const lowerBoundGate = scoreLowerBoundGate(source.arena, minScoreLowerBound, source.scoreLowerBoundPass);
    if (!lowerBoundGate.ok) return releaseError(lowerBoundGate.reason, source, model);
  }

  if (source.requiresPromotion && (minAnalyzedActions !== null || minAverageRootVisits !== null)) {
    const decisionGate = decisionEvidenceGate(source.decisionEvidence, {
      minAnalyzedActions,
      minAverageRootVisits,
    });
    if (!decisionGate.ok) return releaseError(decisionGate.reason, source, model);
  }

  const trainingGate = trainingDataGate(metadata?.training?.data, trainingThresholds);
  if (!trainingGate.ok) {
    return releaseError(trainingGate.reason, { ...source, ...trainingThresholds }, model);
  }
  const validationGate = trainingValidationGate(metadata?.training?.validation, trainingThresholds);
  if (!validationGate.ok) {
    return releaseError(validationGate.reason, { ...source, ...trainingThresholds }, model);
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
    if (source.evaluationSuite.fixedPositions < minFixedPositions) {
      return releaseError("evaluation_suite_fixed_positions_too_few", source, model);
    }
    if (source.evaluationSuite.challengePositions < minChallengePositions) {
      return releaseError("evaluation_suite_challenge_positions_too_few", source, model);
    }
  }

  return {
    ok: true,
    reason: null,
    sourceSchema: source.sourceSchema,
    promoted: source.promoted,
    candidateScore: source.candidateScore,
    promotionThreshold: source.promotionThreshold,
    minSideScore,
    minScoreLowerBound,
    minAnalyzedActions,
    minAverageRootVisits,
    ...trainingThresholds,
    scoreLowerBoundPass: source.scoreLowerBoundPass !== false,
    arena: normalizeReleaseArena(source.arena),
    decisionEvidence: normalizeDecisionEvidence(source.decisionEvidence),
    errors: source.errors,
    activeGeneration: source.activeGeneration,
    evaluationSuite: source.evaluationSuite,
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
      minScoreLowerBound: release.minScoreLowerBound ?? null,
      minAnalyzedActions: release.minAnalyzedActions ?? null,
      minAverageRootVisits: release.minAverageRootVisits ?? null,
      minTrainingSamples: release.minTrainingSamples ?? null,
      minTrainingSides: release.minTrainingSides ?? null,
      minTrainingSources: release.minTrainingSources ?? null,
      minTrainingReanalysisSamples: release.minTrainingReanalysisSamples ?? null,
      minTrainingStateSnapshots: release.minTrainingStateSnapshots ?? null,
      minTrainingAverageRootVisits: release.minTrainingAverageRootVisits ?? null,
      minTrainingValidationSamples: release.minTrainingValidationSamples ?? null,
      maxTrainingValidationValueMse: release.maxTrainingValidationValueMse ?? null,
      maxTrainingValidationPolicyCrossEntropy: release.maxTrainingValidationPolicyCrossEntropy ?? null,
      scoreLowerBoundPass: release.scoreLowerBoundPass !== false,
      arena: release.arena || null,
      decisionEvidence: release.decisionEvidence || null,
      errors: Number(release.errors || 0),
      activeGeneration: release.activeGeneration ?? null,
      evaluationSuite: release.evaluationSuite || emptyEvaluationSuiteEvidence(),
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
      minScoreLowerBound: optionalFiniteOrNull(activeCycle?.evaluation?.minScoreLowerBound),
      minAnalyzedActions: null,
      minAverageRootVisits: null,
      scoreLowerBoundPass: activeCycle?.evaluation?.scoreLowerBoundPass,
      arena: normalizeReleaseArena(activeCycle?.evaluation?.arena),
      decisionEvidence: normalizeDecisionEvidence(activeCycle?.evaluation?.decisionEvidence),
      errors: ladderErrors(artifact),
      activeGeneration,
      evaluationSuite: evaluationSuiteEvidence(activeCycle?.evaluation),
      sideScores: normalizeReleaseSideScores(activeCycle?.evaluation?.sideScores),
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
      minScoreLowerBound: optionalFiniteOrNull(artifact.evaluation?.minScoreLowerBound),
      minAnalyzedActions: null,
      minAverageRootVisits: null,
      scoreLowerBoundPass: artifact.evaluation?.scoreLowerBoundPass,
      arena: normalizeReleaseArena(artifact.evaluation?.arena),
      decisionEvidence: normalizeDecisionEvidence(artifact.evaluation?.decisionEvidence),
      errors: Number(artifact.evaluation?.errors || 0)
        + Number(artifact.iteration?.selfPlay?.errorCount || 0)
        + Number(artifact.iteration?.reanalysisSummary?.errors || 0),
      activeGeneration: null,
      evaluationSuite: evaluationSuiteEvidence(artifact.evaluation),
      sideScores: normalizeReleaseSideScores(artifact.evaluation?.sideScores),
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
      minScoreLowerBound: null,
      minAnalyzedActions: null,
      minAverageRootVisits: null,
      scoreLowerBoundPass: true,
      arena: null,
      decisionEvidence: null,
      errors: 0,
      activeGeneration: null,
      evaluationSuite: emptyEvaluationSuiteEvidence(),
      sideScores: [],
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
    minScoreLowerBound: source.minScoreLowerBound ?? null,
    minAnalyzedActions: source.minAnalyzedActions ?? null,
    minAverageRootVisits: source.minAverageRootVisits ?? null,
    minTrainingSamples: source.minTrainingSamples ?? null,
    minTrainingSides: source.minTrainingSides ?? null,
    minTrainingSources: source.minTrainingSources ?? null,
    minTrainingReanalysisSamples: source.minTrainingReanalysisSamples ?? null,
    minTrainingStateSnapshots: source.minTrainingStateSnapshots ?? null,
    minTrainingAverageRootVisits: source.minTrainingAverageRootVisits ?? null,
    minTrainingValidationSamples: source.minTrainingValidationSamples ?? null,
    maxTrainingValidationValueMse: source.maxTrainingValidationValueMse ?? null,
    maxTrainingValidationPolicyCrossEntropy: source.maxTrainingValidationPolicyCrossEntropy ?? null,
    scoreLowerBoundPass: source.scoreLowerBoundPass !== false,
    arena: normalizeReleaseArena(source.arena),
    decisionEvidence: normalizeDecisionEvidence(source.decisionEvidence),
    errors: Number(source.errors || 0),
    activeGeneration: source.activeGeneration ?? null,
    evaluationSuite: source.evaluationSuite || emptyEvaluationSuiteEvidence(),
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
  return {
    schema: "zizi-el-alamein-alpha-release-suite-evidence-v1",
    explicit: Boolean(evaluation.evaluationSuite.explicit),
    games: entries.length,
    sides: sides.size,
    seeds: seeds.size,
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
    fixedPositions: 0,
    challengePositions: 0,
    challengeQuality: null,
    challengeSelection: null,
  };
}

function challengeQualityEvidence(entries) {
  const challengeEntries = (entries || []).filter((entry) => entry.sourceStateHash || entry.sourceReplay);
  if (!challengeEntries.length) return null;
  return {
    schema: "zizi-el-alamein-alpha-challenge-quality-v1",
    samples: challengeEntries.length,
    averagePriority: averageFinite(challengeEntries.map((entry) => entry.sourcePriority)),
    averagePolicyEntropy: averageFinite(challengeEntries.map((entry) => entry.sourcePolicyEntropy)),
    averageTemperature: averageFinite(challengeEntries.map((entry) => entry.sourceTemperature)),
    averageRootNoiseWeight: averageFinite(challengeEntries.map((entry) => entry.sourceRootNoiseWeight)),
    averageSearchIterations: averageFinite(challengeEntries.map((entry) => entry.sourceSearchIterations)),
    averageRootVisits: averageFinite(challengeEntries.map((entry) => entry.sourceRootVisits)),
    selectionModes: countBy(challengeEntries, (entry) => entry.sourceSelectionMode || "unknown"),
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

function sideScoreGate(sideScores, minSideScore) {
  const scores = normalizeReleaseSideScores(sideScores);
  if (!scores.length) return { ok: false, reason: "missing_side_score_evidence" };
  const weakSide = scores.find((score) => (
    Number(score.scoredGames || 0) < 1 || Number(score.candidateScore || 0) < minSideScore
  ));
  if (weakSide) return { ok: false, reason: "candidate_side_score_below_threshold" };
  return { ok: true, reason: null };
}

function scoreLowerBoundGate(arena, minScoreLowerBound, sourcePass = true) {
  const metrics = normalizeReleaseArena(arena);
  if (!metrics?.scoreInterval95) return { ok: false, reason: "missing_arena_evidence" };
  if (sourcePass === false || Number(metrics.scoreInterval95.low || 0) < minScoreLowerBound) {
    return { ok: false, reason: "candidate_score_lower_bound_below_threshold" };
  }
  return { ok: true, reason: null };
}

function decisionEvidenceGate(decisionEvidence, options = {}) {
  const evidence = normalizeDecisionEvidence(decisionEvidence);
  if (!evidence) return { ok: false, reason: "missing_decision_evidence" };
  const minAnalyzedActions = optionalFiniteOrNull(options.minAnalyzedActions);
  const minAverageRootVisits = optionalFiniteOrNull(options.minAverageRootVisits);
  if (minAnalyzedActions !== null && Number(evidence.analyzedActions || 0) < minAnalyzedActions) {
    return { ok: false, reason: "decision_evidence_too_few_analyzed_actions" };
  }
  if (minAverageRootVisits !== null && Number(evidence.averageRootVisits || 0) < minAverageRootVisits) {
    return { ok: false, reason: "decision_evidence_root_visits_too_low" };
  }
  return { ok: true, reason: null };
}

function trainingDataGate(data, options = {}) {
  const thresholds = [
    options.minTrainingSamples,
    options.minTrainingSides,
    options.minTrainingSources,
    options.minTrainingReanalysisSamples,
    options.minTrainingStateSnapshots,
    options.minTrainingAverageRootVisits,
  ];
  if (thresholds.every((value) => value === null)) return { ok: true, reason: null };
  if (!data) return { ok: false, reason: "missing_training_data_evidence" };
  if (options.minTrainingSamples !== null && Number(data.sampleCount || 0) < options.minTrainingSamples) {
    return { ok: false, reason: "training_samples_too_few" };
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
  return { ok: true, reason: null };
}

function trainingValidationGate(validation, options = {}) {
  const minSamples = options.minTrainingValidationSamples;
  const maxValueMse = options.maxTrainingValidationValueMse;
  const maxPolicyCrossEntropy = options.maxTrainingValidationPolicyCrossEntropy;
  if ([minSamples, maxValueMse, maxPolicyCrossEntropy].every((value) => value === null)) {
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
    actionTypes: countByObject(value.actionTypes),
    selectionModes: countByObject(value.selectionModes),
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

function finiteNumber(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
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
