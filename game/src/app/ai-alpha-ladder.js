import {
  runAlphaTrainingCycle,
  summarizeAlphaTrainingCycle,
} from "./ai-alpha-cycle.js";
import { summarizeAlphaReplayBuffer } from "./ai-alpha-replay-buffer.js";
import { extractAlphaModelArtifact } from "./ai-alpha-model.js";

export const ALPHA_LADDER_CHECKPOINT_SCHEMA = "zizi-el-alamein-alpha-ladder-checkpoint-v1";

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
  const state = createAlphaTrainingLadderState({
    scenario,
    rules,
    board,
    initialState,
    baseModel,
    seed,
    generations,
    seedStride,
    iteration,
    evaluation,
  });
  while (state.cycles.length < state.count) runAlphaTrainingLadderGeneration(state);
  return completeAlphaTrainingLadder(state);
}

export function createAlphaTrainingLadderState({
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
  const replayInputs = iteration.replayInputs || [];
  const replayBufferOptions = iteration.replayBufferOptions || {};
  return {
    scenario,
    rules,
    board,
    initialState,
    iteration,
    evaluation,
    seed,
    count: Math.max(1, Number(generations || 1)),
    seedStride: Number(seedStride || 1000),
    baseModel,
    activeModel: baseModel,
    activeGeneration: baseModel ? 0 : null,
    rollingReplayBuffer: null,
    hallOfFame: baseModel ? [{ generation: 0, label: "generation-0-active", model: baseModel }] : [],
    initialReplayInputs: replayInputs,
    replayBufferOptions,
    replayEnabled: replayInputs.length > 0 || replayBufferOptions.enabled === true,
    cycles: [],
  };
}

export function alphaTrainingLadderGenerationPlan(state) {
  if (!state || state.cycles.length >= state.count) return null;
  const index = state.cycles.length;
  const generation = index + 1;
  const generationSeed = Number(state.seed || 0) + index * state.seedStride;
  const generationIteration = state.replayEnabled
    ? {
      ...state.iteration,
      replayInputs: state.rollingReplayBuffer ? [state.rollingReplayBuffer] : state.initialReplayInputs,
      replayBufferOptions: {
        ...state.replayBufferOptions,
        enabled: true,
      },
    }
    : state.iteration;
  return {
    index,
    generation,
    seed: generationSeed,
    baseModel: state.activeModel,
    iteration: generationIteration,
    evaluation: {
      ...state.evaluation,
      referenceModels: buildAlphaHallOfFameReferenceModels({
        staticReferences: state.evaluation.referenceModels,
        hallOfFame: state.hallOfFame,
        activeGeneration: state.activeGeneration,
        includeHallOfFameReferences: state.evaluation.includeHallOfFameReferences,
        hallOfFameLimit: state.evaluation.hallOfFameLimit,
      }),
      seed: state.evaluation.seed === undefined
        ? undefined
        : Number(state.evaluation.seed || 0) + index * state.seedStride,
    },
  };
}

export function runAlphaTrainingLadderGeneration(state, options = {}) {
  const plan = alphaTrainingLadderGenerationPlan(state);
  if (!plan) throw new Error("Alpha training ladder has no pending generation");
  const cycle = runAlphaTrainingCycle({
    scenario: state.scenario,
    rules: state.rules,
    board: state.board,
    initialState: state.initialState,
    baseModel: plan.baseModel,
    seed: plan.seed,
    iteration: options.selfPlayBatch || options.reanalysisBatch
      ? {
        ...plan.iteration,
        ...(options.selfPlayBatch ? { selfPlayBatch: options.selfPlayBatch } : {}),
        ...(options.reanalysisBatch ? { reanalysisBatch: options.reanalysisBatch } : {}),
      }
      : plan.iteration,
    evaluation: plan.evaluation,
  });
  if (cycle.promoted) {
    state.activeModel = cycle.candidateModel;
    state.activeGeneration = plan.generation;
    state.hallOfFame.push({
      generation: plan.generation,
      label: `generation-${plan.generation}-active`,
      model: state.activeModel,
    });
  }
  if (cycle.replayBuffer) state.rollingReplayBuffer = cycle.replayBuffer;
  const entry = {
    generation: plan.generation,
    seed: plan.seed,
    promoted: cycle.promoted,
    summary: summarizeAlphaTrainingCycle(cycle),
    cycle: omitReplayBuffer(cycle),
  };
  state.cycles.push(entry);
  return entry;
}

export function completeAlphaTrainingLadder(state) {
  if (!state || state.cycles.length !== state.count) {
    throw new Error(`Alpha training ladder incomplete: ${state?.cycles?.length || 0}/${state?.count || 0} generations`);
  }
  return {
    schema: "zizi-el-alamein-alpha-ladder-v1",
    generatedAt: new Date().toISOString(),
    seed: state.seed,
    generations: state.count,
    seedStride: state.seedStride,
    activeGeneration: state.activeGeneration,
    activeModel: state.activeModel,
    baseModel: state.baseModel,
    hallOfFame: state.hallOfFame.map((entry) => ({
      generation: entry.generation,
      label: entry.label,
      modelPresent: Boolean(entry.model),
    })),
    replayBuffer: state.rollingReplayBuffer,
    replayBufferSummary: state.rollingReplayBuffer ? summarizeAlphaReplayBuffer(state.rollingReplayBuffer) : null,
    promotions: state.cycles.filter((cycle) => cycle.promoted).length,
    bestCandidateScore: bestCandidateScore(state.cycles),
    cycles: state.cycles,
  };
}

export function createAlphaTrainingLadderCheckpoint(state) {
  if (!state || !Array.isArray(state.cycles)) throw new Error("Invalid Alpha training ladder state");
  return {
    schema: ALPHA_LADDER_CHECKPOINT_SCHEMA,
    generatedAt: new Date().toISOString(),
    seed: state.seed,
    seedStride: state.seedStride,
    targetGenerations: state.count,
    completedGenerations: state.cycles.length,
    activeGeneration: state.activeGeneration,
    promotions: state.cycles.filter((entry) => entry.promoted).length,
    baseModel: state.baseModel,
    configuration: {
      initialState: state.initialState,
      iteration: state.iteration,
      evaluation: state.evaluation,
    },
    replayBuffer: state.rollingReplayBuffer,
    cycles: state.cycles,
  };
}

export function restoreAlphaTrainingLadderState(checkpoint, config = {}) {
  if (!checkpoint || checkpoint.schema !== ALPHA_LADDER_CHECKPOINT_SCHEMA) {
    throw new Error("Invalid Alpha training ladder checkpoint schema");
  }
  if (!Array.isArray(checkpoint.cycles)) throw new Error("Alpha ladder checkpoint cycles are missing");
  if (!checkpoint.configuration || typeof checkpoint.configuration !== "object") {
    throw new Error("Alpha ladder checkpoint training configuration is missing");
  }
  const completed = Math.max(0, Math.floor(Number(checkpoint.completedGenerations || 0)));
  if (checkpoint.cycles.length !== completed) throw new Error("Alpha ladder checkpoint completed generation mismatch");
  const seed = Number(checkpoint.seed);
  const seedStride = Number(checkpoint.seedStride);
  if (!Number.isFinite(seed) || !Number.isFinite(seedStride) || seedStride <= 0) {
    throw new Error("Alpha ladder checkpoint seed configuration is invalid");
  }
  if (config.seed !== undefined && Number(config.seed) !== seed) {
    throw new Error("Alpha ladder checkpoint seed does not match requested seed");
  }
  if (config.seedStride !== undefined && Number(config.seedStride) !== seedStride) {
    throw new Error("Alpha ladder checkpoint seed stride does not match requested stride");
  }
  const target = Math.max(1, Math.floor(Number(config.generations ?? checkpoint.targetGenerations ?? completed)));
  if (target < completed) throw new Error("Alpha ladder checkpoint has more completed generations than requested");
  const validatedBaseModel = checkpoint.baseModel ? extractAlphaModelArtifact(checkpoint.baseModel) : null;
  if (checkpoint.baseModel && !validatedBaseModel) throw new Error("Alpha ladder checkpoint base model is invalid");
  const checkpointBaseModel = checkpoint.baseModel ? cloneJsonLike(checkpoint.baseModel) : null;
  if (config.baseModel) {
    const requestedBase = extractAlphaModelArtifact(config.baseModel);
    if (!requestedBase || stableModelValue(requestedBase) !== stableModelValue(validatedBaseModel)) {
      throw new Error("Alpha ladder checkpoint base model does not match requested base model");
    }
  }
  if (
    checkpoint.replayBuffer
    && checkpoint.replayBuffer.schema !== "zizi-el-alamein-alpha-replay-buffer-v1"
  ) {
    throw new Error("Alpha ladder checkpoint replay buffer is invalid");
  }
  const state = createAlphaTrainingLadderState({
    ...config,
    initialState: cloneJsonLike(checkpoint.configuration.initialState),
    iteration: cloneJsonLike(checkpoint.configuration.iteration || {}),
    evaluation: cloneJsonLike(checkpoint.configuration.evaluation || {}),
    baseModel: checkpointBaseModel,
    seed,
    generations: target,
    seedStride,
  });
  state.cycles = cloneJsonLike(checkpoint.cycles);
  state.rollingReplayBuffer = checkpoint.replayBuffer ? cloneJsonLike(checkpoint.replayBuffer) : null;
  state.activeModel = checkpointBaseModel;
  state.activeGeneration = checkpointBaseModel ? 0 : null;
  state.hallOfFame = checkpointBaseModel
    ? [{ generation: 0, label: "generation-0-active", model: checkpointBaseModel }]
    : [];
  for (let index = 0; index < state.cycles.length; index += 1) {
    const entry = state.cycles[index];
    const generation = index + 1;
    const expectedSeed = seed + index * seedStride;
    if (Number(entry?.generation) !== generation || Number(entry?.seed) !== expectedSeed) {
      throw new Error(`Alpha ladder checkpoint generation sequence mismatch at ${generation}`);
    }
    if (entry?.cycle?.schema !== "zizi-el-alamein-alpha-cycle-v1") {
      throw new Error(`Alpha ladder checkpoint cycle is invalid at ${generation}`);
    }
    if (Boolean(entry.promoted) !== Boolean(entry.cycle.promoted)) {
      throw new Error(`Alpha ladder checkpoint promotion mismatch at ${generation}`);
    }
    if (entry.promoted) {
      const validatedPromotedModel = extractAlphaModelArtifact(entry.cycle.candidateModel);
      if (!validatedPromotedModel) throw new Error(`Alpha ladder checkpoint promoted model is missing at ${generation}`);
      const promotedModel = entry.cycle.candidateModel;
      state.activeModel = promotedModel;
      state.activeGeneration = generation;
      state.hallOfFame.push({
        generation,
        label: `generation-${generation}-active`,
        model: promotedModel,
      });
    }
  }
  if (Number(checkpoint.promotions || 0) !== state.cycles.filter((entry) => entry.promoted).length) {
    throw new Error("Alpha ladder checkpoint promotion count mismatch");
  }
  if ((checkpoint.activeGeneration ?? null) !== state.activeGeneration) {
    throw new Error("Alpha ladder checkpoint active generation mismatch");
  }
  return state;
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
    totalReferenceArenaReferences: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.referenceArenaReferences || 0), 0),
    hallOfFameSize: Array.isArray(ladder?.hallOfFame) ? ladder.hallOfFame.length : 0,
    weakestReferenceArenaScore: weakestReferenceArenaScore(cycles),
    replayBufferSamples: Number(ladder?.replayBufferSummary?.sampleCount || 0),
    errors: cycles.reduce((sum, cycle) => sum + Number(cycle.summary?.errors || 0), 0),
  };
}

export function buildAlphaHallOfFameReferenceModels({
  staticReferences = [],
  hallOfFame = [],
  activeGeneration = null,
  includeHallOfFameReferences = false,
  hallOfFameLimit = null,
} = {}) {
  const references = Array.isArray(staticReferences) ? staticReferences.slice() : [];
  if (!includeHallOfFameReferences) return references;
  const limit = optionalPositiveInteger(hallOfFameLimit);
  const historical = (Array.isArray(hallOfFame) ? hallOfFame : [])
    .filter((entry) => entry && entry.model && Number(entry.generation) !== Number(activeGeneration))
    .slice(-(limit ?? Number.MAX_SAFE_INTEGER))
    .map((entry) => ({
      label: entry.label || `generation-${entry.generation}-active`,
      source: "hall-of-fame",
      model: entry.model,
    }));
  return [...references, ...historical];
}

function bestCandidateScore(cycles) {
  if (!cycles.length) return 0;
  return Math.max(...cycles.map((cycle) => Number(cycle.summary?.candidateScore || 0)));
}

function weakestReferenceArenaScore(cycles) {
  const scores = (cycles || [])
    .map((cycle) => cycle.summary?.referenceArenaMinScore)
    .map(Number)
    .filter(Number.isFinite);
  return scores.length ? Math.min(...scores) : null;
}

function optionalPositiveInteger(value) {
  if (value === null || value === undefined || value === "") return null;
  const next = Math.floor(Number(value));
  return Number.isFinite(next) && next > 0 ? next : null;
}

function omitReplayBuffer(cycle) {
  if (!cycle || typeof cycle !== "object") return cycle;
  return {
    ...cycle,
    replayBuffer: cycle.replayBuffer ? null : cycle.replayBuffer,
  };
}

function stableModelValue(model) {
  return JSON.stringify(model || null);
}

function cloneJsonLike(value) {
  return value === null || value === undefined ? value : JSON.parse(JSON.stringify(value));
}
