import {
  runAlphaModelMatchBatch,
  runAlphaReferenceArena,
} from "./ai-alpha-evaluation.js";
import { runAlphaTrainingIteration } from "./ai-alpha-iteration.js";

export function runAlphaTrainingCycle({
  scenario,
  rules,
  board = null,
  initialState = null,
  baseModel = null,
  seed = 1942,
  iteration = {},
  evaluation = {},
} = {}) {
  const candidate = runAlphaTrainingIteration({
    scenario,
    rules,
    board,
    initialState,
    baseModel,
    seed,
    games: Number(iteration.games || 1),
    maxPlies: Number(iteration.maxPlies || 80),
    searchOptions: iteration.searchOptions || {},
    selfPlayOptions: iteration.selfPlayOptions || {},
    selfPlayBatch: iteration.selfPlayBatch || null,
    replayInputs: iteration.replayInputs || [],
    replayBufferOptions: iteration.replayBufferOptions || {},
    reanalysisOptions: iteration.reanalysisOptions || {},
    reanalysisBatch: iteration.reanalysisBatch || null,
    trainingOptions: iteration.trainingOptions || {},
  });
  const match = runAlphaModelMatchBatch({
    scenario,
    rules,
    board,
    initialState,
    candidateModel: candidate.model,
    baselineModel: baseModel,
    seed: Number(evaluation.seed ?? Number(seed || 0) + 10000),
    games: Number(evaluation.games || 2),
    maxPlies: Number(evaluation.maxPlies || iteration.maxPlies || 80),
    candidateSide: evaluation.candidateSide || "axis",
    alternateSides: evaluation.alternateSides !== false,
    promotionThreshold: Number(evaluation.promotionThreshold || 0.55),
    minSideScore: evaluation.minSideScore,
    sideScoreBaselines: evaluation.sideScoreBaselines,
    minSideScoreDelta: evaluation.minSideScoreDelta,
    minScoreLowerBound: evaluation.minScoreLowerBound,
    minEloLowerBound: evaluation.minEloLowerBound,
    maxArenaDrawRate: evaluation.maxArenaDrawRate,
    minArenaDecisiveRate: evaluation.minArenaDecisiveRate,
    suite: evaluation.suite || null,
    guardAdjudication: evaluation.guardAdjudication || "situation",
    guardAdjudicationMargin: evaluation.guardAdjudicationMargin,
    requireRulesWinner: evaluation.requireRulesWinner !== false,
    searchOptions: evaluation.searchOptions || iteration.searchOptions || {},
  });
  const referenceArena = Array.isArray(evaluation.referenceModels) && evaluation.referenceModels.length
    ? runAlphaReferenceArena({
      scenario,
      rules,
      board,
      initialState,
      candidateModel: candidate.model,
      references: evaluation.referenceModels,
      seed: Number(evaluation.referenceSeed ?? Number(evaluation.seed ?? Number(seed || 0) + 20000)),
      games: Number(evaluation.referenceGames || evaluation.games || 2),
      maxPlies: Number(evaluation.referenceMaxPlies || evaluation.maxPlies || iteration.maxPlies || 80),
      candidateSide: evaluation.referenceCandidateSide || evaluation.candidateSide || "axis",
      alternateSides: evaluation.referenceAlternateSides ?? (evaluation.alternateSides !== false),
      suite: evaluation.referenceSuite || null,
      searchOptions: evaluation.referenceSearchOptions || evaluation.searchOptions || iteration.searchOptions || {},
      promotionThreshold: Number(evaluation.referencePromotionThreshold || evaluation.promotionThreshold || 0.55),
      guardAdjudication: evaluation.referenceGuardAdjudication || evaluation.guardAdjudication || "situation",
      guardAdjudicationMargin: evaluation.referenceGuardAdjudicationMargin ?? evaluation.guardAdjudicationMargin,
      requireRulesWinner: evaluation.referenceRequireRulesWinner ?? (evaluation.requireRulesWinner !== false),
    })
    : null;
  const referenceGate = referenceArenaPromotionGate(referenceArena, {
    minReferenceArenaScore: evaluation.minReferenceArenaScore,
    minReferenceArenaReferences: evaluation.minReferenceArenaReferences,
    maxReferenceArenaErrors: evaluation.maxReferenceArenaErrors,
  });
  const promoted = Boolean(match.promote && referenceGate.ok);

  return {
    schema: "zizi-el-alamein-alpha-cycle-v1",
    generatedAt: new Date().toISOString(),
    seed,
    promoted,
    promotionBlockedReason: match.promote ? referenceGate.reason : null,
    referenceArenaPass: referenceGate.ok,
    referenceArenaGate: referenceGate,
    activeModel: promoted ? candidate.model : baseModel,
    candidateModel: candidate.model,
    baseModel,
    replayBuffer: candidate.trainingReplayBuffer || candidate.replayBuffer,
    iteration: {
      schema: candidate.schema,
      generatedAt: candidate.generatedAt,
      seed: candidate.seed,
      games: candidate.games,
      maxPlies: candidate.maxPlies,
      selfPlay: candidate.selfPlay,
      replayBufferSummary: candidate.replayBufferSummary,
      trainingReplayBufferSummary: candidate.trainingReplayBufferSummary,
      trainingSampleSelection: candidate.trainingSampleSelection,
      trainingBalance: candidate.trainingBalance,
      reanalysisSummary: candidate.reanalysisSummary,
      training: candidate.training,
    },
    evaluation: match,
    referenceArena,
  };
}

export function summarizeAlphaTrainingCycle(cycle) {
  return {
    schema: "zizi-el-alamein-alpha-cycle-summary-v1",
    promoted: Boolean(cycle?.promoted),
    candidateScore: Number(cycle?.evaluation?.candidateScore || 0),
    candidateEloDiff: cycle?.evaluation?.arena?.eloDiff ?? null,
    promotionThreshold: Number(cycle?.evaluation?.promotionThreshold || 0),
    minSideScore: cycle?.evaluation?.minSideScore ?? null,
    sideScorePass: cycle?.evaluation?.sideScorePass !== false,
    minSideScoreDelta: cycle?.evaluation?.minSideScoreDelta ?? null,
    sideScoreDeltaPass: cycle?.evaluation?.sideScoreDeltaPass !== false,
    minScoreLowerBound: cycle?.evaluation?.minScoreLowerBound ?? null,
    scoreLowerBoundPass: cycle?.evaluation?.scoreLowerBoundPass !== false,
    minEloLowerBound: cycle?.evaluation?.minEloLowerBound ?? null,
    eloLowerBoundPass: cycle?.evaluation?.eloLowerBoundPass !== false,
    maxArenaDrawRate: cycle?.evaluation?.maxArenaDrawRate ?? null,
    drawRatePass: cycle?.evaluation?.drawRatePass !== false,
    minArenaDecisiveRate: cycle?.evaluation?.minArenaDecisiveRate ?? null,
    decisiveRatePass: cycle?.evaluation?.decisiveRatePass !== false,
    promotionVerdict: cycle?.evaluation?.promotionVerdict || null,
    promotionBlockedReason: cycle?.promotionBlockedReason || null,
    referenceArenaPass: cycle?.referenceArenaPass !== false,
    minReferenceArenaScore: cycle?.referenceArenaGate?.minReferenceArenaScore ?? null,
    minReferenceArenaReferences: cycle?.referenceArenaGate?.minReferenceArenaReferences ?? null,
    maxReferenceArenaErrors: cycle?.referenceArenaGate?.maxReferenceArenaErrors ?? null,
    referenceArenaReferences: Number(cycle?.referenceArena?.references || 0),
    referenceArenaScoredReferences: Number(cycle?.referenceArena?.scoredReferences || 0),
    referenceArenaMinScore: cycle?.referenceArena?.minCandidateScore ?? null,
    referenceArenaAverageScore: cycle?.referenceArena?.averageCandidateScore ?? null,
    selfPlayGames: Number(cycle?.iteration?.selfPlay?.games || cycle?.iteration?.games || 0),
    evaluationGames: Number(cycle?.evaluation?.games || 0),
    trainingSamples: Number(cycle?.iteration?.training?.samples || 0),
    reanalysisSamples: Number(cycle?.iteration?.reanalysisSummary?.sampleCount || 0),
    errors: Number(cycle?.evaluation?.errors || 0)
      + Number(cycle?.iteration?.selfPlay?.errorCount || 0)
      + Number(cycle?.iteration?.reanalysisSummary?.errors || 0),
  };
}

function referenceArenaPromotionGate(arena, options = {}) {
  const minScore = optionalFiniteOrNull(options.minReferenceArenaScore);
  const minReferences = optionalFiniteOrNull(options.minReferenceArenaReferences);
  const maxErrors = optionalFiniteOrNull(options.maxReferenceArenaErrors);
  if (minScore === null && minReferences === null && maxErrors === null) {
    return {
      ok: true,
      reason: null,
      minReferenceArenaScore: null,
      minReferenceArenaReferences: null,
      maxReferenceArenaErrors: null,
    };
  }
  if (!arena?.references) {
    return {
      ok: false,
      reason: "missing_reference_arena_evidence",
      minReferenceArenaScore: minScore,
      minReferenceArenaReferences: minReferences,
      maxReferenceArenaErrors: maxErrors,
    };
  }
  if (minReferences !== null && Number(arena.scoredReferences || 0) < minReferences) {
    return {
      ok: false,
      reason: "reference_arena_too_few_references",
      minReferenceArenaScore: minScore,
      minReferenceArenaReferences: minReferences,
      maxReferenceArenaErrors: maxErrors,
    };
  }
  if (maxErrors !== null && Number(arena.errors || 0) > maxErrors) {
    return {
      ok: false,
      reason: "reference_arena_errors_exceed_limit",
      minReferenceArenaScore: minScore,
      minReferenceArenaReferences: minReferences,
      maxReferenceArenaErrors: maxErrors,
    };
  }
  if (
    minScore !== null
    && (
      arena.minCandidateScore === null
      || arena.minCandidateScore === undefined
      || Number(arena.minCandidateScore) < minScore
    )
  ) {
    return {
      ok: false,
      reason: "reference_arena_score_below_threshold",
      minReferenceArenaScore: minScore,
      minReferenceArenaReferences: minReferences,
      maxReferenceArenaErrors: maxErrors,
    };
  }
  return {
    ok: true,
    reason: null,
    minReferenceArenaScore: minScore,
    minReferenceArenaReferences: minReferences,
    maxReferenceArenaErrors: maxErrors,
  };
}

function optionalFiniteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}
