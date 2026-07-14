import {
  ENV_ACTION,
  activeSide,
  applyEnvironmentAction,
  createEnvironment,
  generateLegalActions,
  stateHash,
} from "../core/index.js";
import {
  actionKey,
  analyzePosition,
  makeSearchTrainingSample,
  matchingAlphaPhasePlanHint,
  nextAlphaReusablePhasePlan,
} from "./ai-alpha-search.js";
import { evaluateSituation } from "./ai-situation.js";
import { canonicalSha256 } from "../../../shared/wargame-alpha/fingerprint.js";

const DEFAULT_MAX_PLIES = 160;
const DEFAULT_GUARD_OUTCOME_MODE = "policy-only";
const DEFAULT_GUARD_OUTCOME_WEIGHT = 0.35;
const DEFAULT_GUARD_OUTCOME_DISCOUNT = 0.985;

export function runAlphaSelfPlayGame({
  scenario,
  rules,
  board = null,
  initialState = null,
  model = null,
  seed = 1942,
  maxPlies = DEFAULT_MAX_PLIES,
  searchOptions = {},
  temperature = 0,
  temperaturePlies = 12,
  rootNoiseWeight = 0,
  rootDirichletAlpha = 0.3,
  rootNoisePlies = temperaturePlies,
  guardOutcomeMode = DEFAULT_GUARD_OUTCOME_MODE,
  guardOutcomeWeight = DEFAULT_GUARD_OUTCOME_WEIGHT,
  guardOutcomeDiscount = DEFAULT_GUARD_OUTCOME_DISCOUNT,
  includeStateSnapshots = false,
  rollDie = null,
  random = null,
} = {}) {
  const dieRoll = rollDie || makeSeededDieRoller(seed);
  const policyRandom = random || mulberry32(Number(seed || 0) ^ 0xA5A5A5A5);
  let environment = createEnvironment({ scenario, rules, board, state: initialState });
  const initialStateFingerprint = canonicalSha256(environment.state, "trajectory initial state");
  const start = selfPlayStartMetadata(environment);
  const decisions = [];
  const actions = [];
  const errors = [];
  let reusablePhasePlan = null;
  let plies = 0;

  while (!environment.state.winner && plies < maxPlies) {
    const legalActions = generateLegalActions(environment, { includeChanceActions: true });
    if (!legalActions.length) {
      errors.push({ reason: "no_legal_actions", stateHash: stateHash(environment) });
      break;
    }

    let analysis = null;
    let action = null;
    const decisionTemperature = plies < Number(temperaturePlies || 0) ? Number(temperature || 0) : 0;
    const decisionRootNoiseWeight = plies < Number(rootNoisePlies || 0) ? Number(rootNoiseWeight || 0) : 0;
    if (legalActions.every(isChanceAction)) {
      action = chooseChanceAction(legalActions, dieRoll());
    } else {
      const side = activeSide(environment);
      const phasePlanHint = matchingAlphaPhasePlanHint(reusablePhasePlan, environment, side, legalActions);
      analysis = analyzePosition(environment, {
        side,
        model,
        ...searchOptions,
        ...(phasePlanHint ? { phasePlanHint } : {}),
        rootNoiseWeight: decisionRootNoiseWeight,
        rootDirichletAlpha: Number(rootDirichletAlpha || 0.3),
        rootNoiseRandom: decisionRootNoiseWeight > 0 ? policyRandom : null,
        gumbelRandom: String(searchOptions.rootSelectionMode || "").toLowerCase() === "gumbel"
          ? policyRandom
          : searchOptions.gumbelRandom,
        includeStateSnapshot: Boolean(includeStateSnapshots),
      });
      action = !analysis.requiresChance
        ? selectAlphaPolicyAction(analysis, {
          temperature: decisionTemperature,
          random: policyRandom,
        })
        : fallbackAction(legalActions);
      reusablePhasePlan = nextAlphaReusablePhasePlan(analysis, action, environment, side);
    }

    if (!action) {
      errors.push({ reason: "no_action_selected", stateHash: stateHash(environment) });
      break;
    }

    const applied = applyEnvironmentAction(environment, action, {
      enrichEvents: false,
      previousState: false,
      cloneResultState: false,
    });
    if (!applied.ok) {
      errors.push({
        reason: "selected_action_failed",
        applyReason: applied.reason,
        action,
        stateHash: stateHash(environment),
      });
      break;
    }

    const selection = analysis ? {
      mode: selectedPolicyMode(analysis, action, {
        temperature: decisionTemperature,
      }),
      temperature: decisionTemperature,
      rootNoiseWeight: decisionRootNoiseWeight,
    } : { mode: "chance", temperature: 0 };
    if (analysis) {
      decisions.push({
        analysis,
        action: applied.action,
        selection,
        ply: plies,
      });
    }
    actions.push({
      ply: plies,
      side: activeSide(environment),
      stateHashBefore: stateHash(environment),
      action: applied.action,
      selection,
      stateHashAfter: stateHash(applied.environment),
    });
    environment = applied.environment;
    plies += 1;
  }

  const winner = environment.state.winner || null;
  const guardHit = !winner && plies >= maxPlies;
  const rawSamples = makeSelfPlayTrainingSamples(decisions, {
    environment,
    winner,
    guardHit,
    guardOutcomeMode,
    guardOutcomeWeight,
    guardOutcomeDiscount,
    finalPly: plies,
  });
  const finalStateHash = stateHash(environment);
  const finalStateFingerprint = canonicalSha256(environment.state, "trajectory final state");
  const trajectoryId = alphaSelfPlayTrajectoryId({
    scenario,
    rules,
    seed,
    start,
    initialStateFingerprint,
    finalStateHash,
    finalStateFingerprint,
    plies,
    maxPlies,
    guardHit,
    guardOutcomeMode,
    guardOutcomeWeight,
    guardOutcomeDiscount,
    winner,
    errors,
    actions,
  });
  const samples = rawSamples.map((sample) => ({
    ...sample,
    trajectoryIds: [trajectoryId],
  }));
  return {
    schema: "zizi-el-alamein-alpha-self-play-v1",
    seed,
    trajectoryId,
    initialStateHash: start.stateHash,
    initialTurn: start.turn,
    initialPhaseId: start.phaseId,
    initialSide: start.side,
    plies,
    maxPlies,
    guardHit,
    guardOutcomeMode: normalizeGuardOutcomeMode(guardOutcomeMode),
    guardOutcomeWeight: normalizedGuardOutcomeWeight(guardOutcomeWeight),
    guardOutcomeDiscount: normalizedGuardOutcomeDiscount(guardOutcomeDiscount),
    includeStateSnapshots: Boolean(includeStateSnapshots),
    winner,
    finalStateHash,
    sampleOutcomeSources: countBy(samples, (sample) => sample.outcomeSource || "unknown"),
    averageOutcomeWeight: averageOutcomeWeight(samples),
    samples,
    actions,
    errors,
  };
}

export function alphaSelfPlayTrajectoryId({
  scenario,
  rules,
  seed,
  start,
  initialStateFingerprint,
  finalStateHash,
  finalStateFingerprint,
  plies,
  maxPlies,
  guardHit,
  guardOutcomeMode,
  guardOutcomeWeight,
  guardOutcomeDiscount,
  winner,
  errors,
  actions,
} = {}) {
  return canonicalSha256({
    schema: "wargame-alpha-trajectory-fingerprint-v1",
    environment: canonicalSha256({ scenario, rules }, "trajectory environment"),
    seed: Number(seed ?? 0),
    initial: {
      stateHash: start?.stateHash || null,
      stateFingerprint: initialStateFingerprint || null,
      turn: Number(start?.turn ?? 1),
      phaseId: start?.phaseId || null,
      side: start?.side || null,
    },
    final: {
      stateHash: finalStateHash || null,
      stateFingerprint: finalStateFingerprint || null,
      plies: Number(plies || 0),
      maxPlies: Number(maxPlies || 0),
      guardHit: Boolean(guardHit),
      guardOutcomeMode: normalizeGuardOutcomeMode(guardOutcomeMode),
      guardOutcomeWeight: normalizedGuardOutcomeWeight(guardOutcomeWeight),
      guardOutcomeDiscount: normalizedGuardOutcomeDiscount(guardOutcomeDiscount),
      winner: winner || null,
      errors: (errors || []).map((error) => ({
        reason: error?.reason || null,
        applyReason: error?.applyReason || null,
        stateHash: error?.stateHash || null,
      })),
    },
    trace: (actions || []).map((entry) => ({
      ply: Number(entry?.ply || 0),
      side: entry?.side || null,
      stateHashBefore: entry?.stateHashBefore || null,
      action: entry?.action || null,
      stateHashAfter: entry?.stateHashAfter || null,
    })),
  }, "self-play trajectory");
}

export function runAlphaSelfPlayBatch({
  games = 1,
  seed = 1942,
  initialState = null,
  initialStates = null,
  ...options
} = {}) {
  const count = Math.max(1, Number(games || 1));
  const results = [];
  for (let index = 0; index < count; index += 1) {
    results.push(runAlphaSelfPlayGame({
      ...options,
      initialState: alphaSelfPlayInitialState({ initialState, initialStates }, index),
      seed: Number(seed || 0) + index,
    }));
  }
  return makeAlphaSelfPlayBatch(results, { seed });
}

export function makeAlphaSelfPlayBatch(results = [], options = {}) {
  const games = Array.isArray(results) ? results.slice() : [];
  return {
    schema: "zizi-el-alamein-alpha-self-play-batch-v1",
    generatedAt: options.generatedAt || new Date().toISOString(),
    seed: Number(options.seed ?? games[0]?.seed ?? 1942),
    games: games.length,
    wins: countBy(games, (result) => result.winner?.side || "unresolved"),
    initialSides: countBy(games, (result) => result.initialSide || "unknown"),
    initialPhases: countBy(games, (result) => result.initialPhaseId || "unknown"),
    uniqueInitialPositions: new Set(games.map((result) => result.initialStateHash).filter(Boolean)).size,
    sampleCount: games.reduce((sum, result) => sum + (result.samples || []).length, 0),
    sampleOutcomeSources: mergeCounts(games.map((result) => result.sampleOutcomeSources || {})),
    averageOutcomeWeight: averageOutcomeWeight(games.flatMap((result) => result.samples || [])),
    includeStateSnapshots: games.some((result) => result.includeStateSnapshots),
    actionCount: games.reduce((sum, result) => sum + (result.actions || []).length, 0),
    errorCount: games.reduce((sum, result) => sum + (result.errors || []).length, 0),
    results: games,
  };
}

export function alphaSelfPlayInitialState({ initialState = null, initialStates = null } = {}, gameIndex = 0) {
  if (initialStates !== null && initialStates !== undefined && !Array.isArray(initialStates)) {
    throw new TypeError("Alpha self-play initialStates must be an array");
  }
  const states = Array.isArray(initialStates) ? initialStates : [];
  if (states.some((state) => !state || typeof state !== "object" || Array.isArray(state))) {
    throw new TypeError("Alpha self-play initialStates contains an invalid state");
  }
  if (!states.length) return initialState;
  const index = Math.max(0, Math.floor(Number(gameIndex || 0))) % states.length;
  return states[index];
}

export function mulberry32(initialSeed) {
  let state = Number(initialSeed || 0) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let next = state;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeSeededDieRoller(seed) {
  const rng = mulberry32(seed);
  return () => Math.floor(rng() * 6) + 1;
}

export function selectAlphaPolicyAction(analysis, options = {}) {
  const policy = (analysis?.policy || []).filter((entry) => entry?.action);
  if (!policy.length || analysis?.requiresChance) return null;
  const temperature = Number(options.temperature || 0);
  if (!(temperature > 0)) return analysis.bestAction || policy[0].action;
  const random = options.random || (() => 0);
  const weighted = policy.map((entry) => ({
    action: entry.action,
    weight: policyWeight(entry, temperature),
  }));
  const total = weighted.reduce((sum, entry) => sum + entry.weight, 0);
  if (!(total > 0)) return analysis.bestAction || policy[0].action;
  let cursor = clamp01(random()) * total;
  for (const entry of weighted) {
    cursor -= entry.weight;
    if (cursor <= 0) return entry.action;
  }
  return weighted[weighted.length - 1].action;
}

function makeSelfPlayTrainingSamples(decisions, options) {
  const winner = options.winner || null;
  const guardHit = Boolean(options.guardHit);
  const guardOutcomeMode = normalizeGuardOutcomeMode(options.guardOutcomeMode);
  const guardOutcomeWeight = normalizedGuardOutcomeWeight(options.guardOutcomeWeight);
  const guardOutcomeDiscount = normalizedGuardOutcomeDiscount(options.guardOutcomeDiscount);
  const guardAxisValue = !winner && guardHit && guardOutcomeMode === "final-evaluation"
    ? evaluateSituation(options.environment, { side: "axis" })
    : null;
  const outcomeSide = winner?.side || null;
  return (decisions || []).map((decision) => {
    const analysis = decision?.analysis || decision;
    const sample = makeSearchTrainingSample(analysis, outcomeSide);
    const decisionMetadata = makeSelfPlayDecisionMetadata(decision);
    if (!winner && guardHit && guardOutcomeMode === "final-evaluation") {
      const horizonPlies = Math.max(0, Number(options.finalPly || 0) - Number(decision?.ply || 0));
      const confidenceDiscount = guardOutcomeDiscount ** horizonPlies;
      const finalValue = analysis.side === "axis" ? guardAxisValue : -guardAxisValue;
      return {
        ...sample,
        decision: decisionMetadata,
        outcome: round(finalValue, 6),
        outcomeSource: "guard_final_evaluation",
        outcomeWeight: round(guardOutcomeWeight * confidenceDiscount, 6),
        outcomeHorizonPlies: horizonPlies,
        outcomeDiscount: guardOutcomeDiscount,
      };
    }
    return {
      ...sample,
      decision: decisionMetadata,
      outcome: winner ? sample.outcome : null,
      outcomeSource: winner ? "terminal_result" : guardHit ? "policy_only_guard" : "policy_only_unresolved",
      outcomeWeight: winner ? 1 : 0,
    };
  });
}

function selfPlayStartMetadata(environment) {
  const phase = environment?.rules?.phases?.[environment?.state?.phaseIndex] || null;
  return {
    stateHash: stateHash(environment),
    turn: Number(environment?.state?.turn || 1),
    phaseId: phase?.id || null,
    side: phase?.side || activeSide(environment),
  };
}

function makeSelfPlayDecisionMetadata(decision) {
  const analysis = decision?.analysis || decision;
  const selection = decision?.selection || {};
  const selectedAction = decision?.action || analysis?.bestAction || null;
  const selectedPolicy = selectedPolicyEntry(analysis, selectedAction);
  return {
    schema: "zizi-el-alamein-alpha-self-play-decision-v1",
    selectedAction,
    selectedVisitShare: finiteNumber(selectedPolicy?.visitShare, 0),
    selectedVisits: finiteNumber(selectedPolicy?.visits, 0),
    selectedQ: finiteOrNull(selectedPolicy?.q),
    selectedPrior: finiteOrNull(selectedPolicy?.prior),
    selectedPolicyRank: finiteOrNull(selectedPolicy?.rank),
    selectedIsBest: selectedPolicy?.rank === 1,
    selectionMode: selection.mode || "unknown",
    temperature: finiteNumber(selection.temperature, 0),
    rootNoiseWeight: finiteNumber(selection.rootNoiseWeight, 0),
    policyEntropy: policyEntropy(analysis?.policy || []),
    recommendationConfidence: finiteOrNull(analysis?.recommendation?.confidence),
    recommendationVisitMargin: finiteOrNull(analysis?.recommendation?.visitMargin),
    recommendationQMargin: finiteOrNull(analysis?.recommendation?.qMargin),
    recommendationLabel: typeof analysis?.recommendation?.label === "string" ? analysis.recommendation.label : null,
    policySize: Array.isArray(analysis?.policy) ? analysis.policy.length : 0,
    searchIterations: finiteNumber(analysis?.search?.iterations, 0),
    rootVisits: finiteNumber(analysis?.search?.rootVisits, 0),
    rootSelectionMode: analysis?.search?.rootSelection?.mode || "puct",
    rootSelectionInitialCandidates: finiteNumber(analysis?.search?.rootSelection?.initialCandidates, 0),
    rootSelectionRounds: Array.isArray(analysis?.search?.rootSelection?.rounds)
      ? analysis.search.rootSelection.rounds.length
      : 0,
    phasePlanActions: finiteNumber(analysis?.search?.phasePlan?.actions, 0),
    phasePlanReused: Boolean(analysis?.search?.phasePlan?.reused),
    phasePlanSearchedNodes: finiteNumber(analysis?.search?.phasePlan?.searchedNodes, 0),
  };
}

function selectedPolicyEntry(analysis, action) {
  if (!analysis || !action || !Array.isArray(analysis.policy)) return null;
  const selectedKey = actionKey(action);
  const index = analysis.policy.findIndex((entry) => actionKey(entry.action) === selectedKey);
  return index >= 0 ? { ...analysis.policy[index], rank: index + 1 } : null;
}

function normalizeGuardOutcomeMode(value) {
  return value === "final-evaluation" ? "final-evaluation" : DEFAULT_GUARD_OUTCOME_MODE;
}

function normalizedGuardOutcomeWeight(value) {
  const next = Number(value ?? DEFAULT_GUARD_OUTCOME_WEIGHT);
  if (!Number.isFinite(next)) return DEFAULT_GUARD_OUTCOME_WEIGHT;
  return Math.min(1, Math.max(0, next));
}

function normalizedGuardOutcomeDiscount(value) {
  const next = Number(value ?? DEFAULT_GUARD_OUTCOME_DISCOUNT);
  if (!Number.isFinite(next)) return DEFAULT_GUARD_OUTCOME_DISCOUNT;
  return Math.min(1, Math.max(0, next));
}

function policyWeight(entry, temperature) {
  const visits = Number(entry.visits || 0);
  const share = Number(entry.visitShare || 0);
  const base = visits > 0 ? visits : share > 0 ? share : Number(entry.prior || 0);
  if (!(base > 0)) return 0;
  return base ** (1 / Math.max(0.01, temperature));
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

function selectedPolicyMode(analysis, action, options = {}) {
  if (!analysis || !action) return "fallback";
  if (analysis.bestAction && JSON.stringify(analysis.bestAction) === JSON.stringify(action)) {
    return Number(options.temperature || 0) > 0 ? "sampled_best" : "best";
  }
  return Number(options.temperature || 0) > 0 ? "sampled" : "fallback";
}

function chooseChanceAction(actions, roll) {
  const dieRoll = Math.max(1, Math.min(6, Number(roll || 1)));
  return actions.find((action) => Number(action.dieRoll) === dieRoll) || actions[0] || null;
}

function fallbackAction(actions) {
  return actions.find((action) => action.type !== ENV_ACTION.RESOLVE_COMBAT) || actions[0] || null;
}

function isChanceAction(action) {
  return action?.type === ENV_ACTION.RESOLVE_COMBAT && Number.isInteger(Number(action.dieRoll));
}

function clamp01(value) {
  const next = Number(value);
  if (!Number.isFinite(next)) return 0;
  return Math.min(0.999999999, Math.max(0, next));
}

function countBy(items, selector) {
  return items.reduce((counts, item) => {
    const key = selector(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function mergeCounts(items) {
  return (items || []).reduce((merged, counts) => {
    for (const [key, value] of Object.entries(counts || {})) {
      merged[key] = (merged[key] || 0) + Number(value || 0);
    }
    return merged;
  }, {});
}

function averageOutcomeWeight(samples) {
  const weights = (samples || []).map((sample) => Number(sample.outcomeWeight ?? 1)).filter(Number.isFinite);
  if (!weights.length) return 0;
  return round(weights.reduce((sum, value) => sum + value, 0) / weights.length, 4);
}

function finiteNumber(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}
