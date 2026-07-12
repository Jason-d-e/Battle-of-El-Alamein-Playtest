import {
  runAlphaTrainingCycle,
  summarizeAlphaTrainingCycle,
} from "./ai-alpha-cycle.js";
import { summarizeAlphaReplayBuffer } from "./ai-alpha-replay-buffer.js";

export function runAlphaTrainingLadder({
  scenario,
  rules,
  board = null,
  initialState = null,
  baseModel = null,
  seed = 1942,
  generations = 1,
  seedStride = 1000,
  iteration = {},
  evaluation = {},
} = {}) {
  const count = Math.max(1, Number(generations || 1));
  const cycles = [];
  let activeModel = baseModel;
  let activeGeneration = baseModel ? 0 : null;
  let rollingReplayBuffer = null;
  const initialReplayInputs = iteration.replayInputs || [];
  const replayBufferOptions = iteration.replayBufferOptions || {};
  const replayEnabled = initialReplayInputs.length > 0 || replayBufferOptions.enabled === true;

  for (let index = 0; index < count; index += 1) {
    const generation = index + 1;
    const generationSeed = Number(seed || 0) + index * Number(seedStride || 1000);
    const generationIteration = replayEnabled
      ? {
        ...iteration,
        replayInputs: rollingReplayBuffer ? [rollingReplayBuffer] : initialReplayInputs,
        replayBufferOptions: {
          ...replayBufferOptions,
          enabled: true,
        },
      }
      : iteration;
    const cycle = runAlphaTrainingCycle({
      scenario,
      rules,
      board,
      initialState,
      baseModel: activeModel,
      seed: generationSeed,
      iteration: generationIteration,
      evaluation: {
        ...evaluation,
        seed: evaluation.seed === undefined
          ? undefined
          : Number(evaluation.seed || 0) + index * Number(seedStride || 1000),
      },
    });
    if (cycle.promoted) {
      activeModel = cycle.candidateModel;
      activeGeneration = generation;
    }
    if (cycle.replayBuffer) rollingReplayBuffer = cycle.replayBuffer;
    cycles.push({
      generation,
      seed: generationSeed,
      promoted: cycle.promoted,
      summary: summarizeAlphaTrainingCycle(cycle),
      cycle: omitReplayBuffer(cycle),
    });
  }

  return {
    schema: "zizi-el-alamein-alpha-ladder-v1",
    generatedAt: new Date().toISOString(),
    seed,
    generations: count,
    seedStride: Number(seedStride || 1000),
    activeGeneration,
    activeModel,
    baseModel,
    replayBuffer: rollingReplayBuffer,
    replayBufferSummary: rollingReplayBuffer ? summarizeAlphaReplayBuffer(rollingReplayBuffer) : null,
    promotions: cycles.filter((cycle) => cycle.promoted).length,
    bestCandidateScore: bestCandidateScore(cycles),
    cycles,
  };
}

export function summarizeAlphaTrainingLadder(ladder) {
  const cycles = ladder?.cycles || [];
  return {
    schema: "zizi-el-alamein-alpha-ladder-summary-v1",
    generations: Number(ladder?.generations || cycles.length || 0),
    promotions: Number(ladder?.promotions || 0),
    activeGeneration: ladder?.activeGeneration ?? null,
    hasActiveModel: Boolean(ladder?.activeModel),
    bestCandidateScore: Number(ladder?.bestCandidateScore || 0),
    totalSelfPlayGames: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.selfPlayGames || 0), 0),
    totalEvaluationGames: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.evaluationGames || 0), 0),
    totalTrainingSamples: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.trainingSamples || 0), 0),
    totalReanalysisSamples: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.reanalysisSamples || 0), 0),
    replayBufferSamples: Number(ladder?.replayBufferSummary?.sampleCount || 0),
    errors: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.errors || 0), 0),
  };
}

function bestCandidateScore(cycles) {
  if (!cycles.length) return 0;
  return Math.max(...cycles.map((cycle) => Number(cycle.summary?.candidateScore || 0)));
}

function omitReplayBuffer(cycle) {
  if (!cycle || typeof cycle !== "object") return cycle;
  return {
    ...cycle,
    replayBuffer: cycle.replayBuffer ? null : cycle.replayBuffer,
  };
}
