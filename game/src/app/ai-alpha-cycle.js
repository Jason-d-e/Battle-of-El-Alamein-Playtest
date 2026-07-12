import { runAlphaModelMatchBatch } from "./ai-alpha-evaluation.js";
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
    replayInputs: iteration.replayInputs || [],
    replayBufferOptions: iteration.replayBufferOptions || {},
    reanalysisOptions: iteration.reanalysisOptions || {},
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
    minScoreLowerBound: evaluation.minScoreLowerBound,
    suite: evaluation.suite || null,
    searchOptions: evaluation.searchOptions || iteration.searchOptions || {},
  });

  return {
    schema: "zizi-el-alamein-alpha-cycle-v1",
    generatedAt: new Date().toISOString(),
    seed,
    promoted: Boolean(match.promote),
    activeModel: match.promote ? candidate.model : baseModel,
    candidateModel: candidate.model,
    baseModel,
    replayBuffer: candidate.replayBuffer,
    iteration: {
      schema: candidate.schema,
      generatedAt: candidate.generatedAt,
      seed: candidate.seed,
      games: candidate.games,
      maxPlies: candidate.maxPlies,
      selfPlay: candidate.selfPlay,
      replayBufferSummary: candidate.replayBufferSummary,
      trainingSampleSelection: candidate.trainingSampleSelection,
      reanalysisSummary: candidate.reanalysisSummary,
      training: candidate.training,
    },
    evaluation: match,
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
    minScoreLowerBound: cycle?.evaluation?.minScoreLowerBound ?? null,
    scoreLowerBoundPass: cycle?.evaluation?.scoreLowerBoundPass !== false,
    selfPlayGames: Number(cycle?.iteration?.selfPlay?.games || cycle?.iteration?.games || 0),
    evaluationGames: Number(cycle?.evaluation?.games || 0),
    trainingSamples: Number(cycle?.iteration?.training?.samples || 0),
    reanalysisSamples: Number(cycle?.iteration?.reanalysisSummary?.sampleCount || 0),
    errors: Number(cycle?.evaluation?.errors || 0)
      + Number(cycle?.iteration?.selfPlay?.errorCount || 0)
      + Number(cycle?.iteration?.reanalysisSummary?.errors || 0),
  };
}
