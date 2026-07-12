import {
  ENV_ACTION,
  activeSide,
  applyEnvironmentAction,
  generateLegalActions,
  hexDistance,
  liveUnits,
  neighborsOf,
  unitById,
} from "../core/index.js";
import { combatEliminationProfile } from "./ai-tactics.js";

const DEFAULT_BEAM_WIDTH = 20;
const DEFAULT_CANDIDATE_LIMIT = 36;
const DEFAULT_MAX_ACTIONS = 12;
const DEFAULT_NODE_LIMIT = 180;
const DEFAULT_TIME_BUDGET_MS = 60;
const DEFAULT_MIN_NODES = 8;
const DEFAULT_PROJECTION_WEIGHT = 0.12;
const DEFAULT_PROJECTION_CANDIDATE_LIMIT = 18;
const DEFAULT_PROJECTION_BEAM_LIMIT = 4;

export const DEFAULT_PHASE_SEARCH_WEIGHTS = Object.freeze({
  axisProgress: 220,
  axisMobileFactor: 0.25,
  axisCombatFactor: 0.18,
  axisObjective: 2600,
  axisObjectiveSupport: 30,
  axisObjectiveCounterThreat: -60,
  axisObjectiveUnsupportedPenalty: -360,
  axisAdjacent: 420,
  axisNear: 160,
  axisReserve: 8,
  axisStateObjectiveHeld: 2400,
  axisStateObjectiveDistance: -42,
  alliedForwardBand: 540,
  alliedForwardBandPenalty: -52,
  alliedObjectiveHugPenalty: -320,
  alliedContact: 260,
  alliedContactPenalty: -38,
  alliedLine: 95,
  alliedDrift: -42,
  alliedCombat: 18,
  alliedStateAxisDistance: 34,
  alliedStateWallLink: 38,
});

export function beamSearchMovementPhase(environment, options = {}) {
  const side = options.side || activeSide(environment);
  const beamWidth = Number(options.beamWidth || DEFAULT_BEAM_WIDTH);
  const candidateLimit = Number(options.candidateLimit || DEFAULT_CANDIDATE_LIMIT);
  const maxActions = Number(options.maxActions || DEFAULT_MAX_ACTIONS);
  const nodeLimit = Number(options.nodeLimit || DEFAULT_NODE_LIMIT);
  const timeBudgetMs = Number(options.timeBudgetMs ?? DEFAULT_TIME_BUDGET_MS);
  const minNodes = Number(options.minNodes ?? DEFAULT_MIN_NODES);
  const scoreAction = options.scoreAction || defaultMovementActionScore;
  const scoreState = options.scoreState || defaultMovementStateScore;
  const weights = mergePhaseSearchWeights(options.weights);
  const projectPhaseEnd = options.projectPhaseEnd !== false;
  const projectionWeight = Number(options.projectionWeight ?? DEFAULT_PROJECTION_WEIGHT);
  const projectionCandidateLimit = Number(options.projectionCandidateLimit || DEFAULT_PROJECTION_CANDIDATE_LIMIT);
  const projectionBeamLimit = Number(options.projectionBeamLimit || DEFAULT_PROJECTION_BEAM_LIMIT);
  const distanceCache = options.distanceCache || new Map();
  const startedAt = Date.now();
  let searchedNodes = 0;

  const overBudget = () => (
    searchedNodes >= nodeLimit
    || (searchedNodes >= minNodes && timeBudgetMs > 0 && Date.now() - startedAt >= timeBudgetMs)
  );

  let beams = [{
    environment,
    actions: [],
    score: scoreState(environment, { side, distanceCache, weights }),
    ended: false,
  }];

  for (let depth = 0; depth < maxActions; depth += 1) {
    const nextBeams = [];
    for (const beam of beams) {
      if (beam.ended || overBudget()) {
        nextBeams.push(beam);
        continue;
      }
      const moveActions = generateLegalActions(beam.environment, { includeChanceActions: false })
        .filter((action) => action.type === ENV_ACTION.MOVE_UNIT)
        .map((action) => ({
          action,
          score: quickMoveActionScore(beam.environment, action, side, distanceCache, weights),
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, candidateLimit)
        .map((candidate) => candidate.action);

      if (!moveActions.length) {
        nextBeams.push({ ...beam, ended: true });
        continue;
      }

      for (const action of moveActions) {
        if (overBudget()) break;
        const applied = applyEnvironmentAction(beam.environment, action, {
          enrichEvents: false,
          previousState: false,
          cloneResultState: false,
        });
        searchedNodes += 1;
        if (!applied.ok) continue;
        const actionScore = scoreAction(beam.environment, action, applied, { side, depth, distanceCache, weights });
        const stateScore = scoreState(applied.environment, { side, depth: depth + 1, distanceCache, weights });
        nextBeams.push({
          environment: applied.environment,
          actions: beam.actions.concat(applied.action),
          score: beam.score + actionScore + stateScore * 0.08,
          ended: Boolean(applied.state.winner),
          searchedNodes,
        });
      }

      nextBeams.push({
        ...beam,
        score: beam.score + scoreState(beam.environment, { side, depth, stopped: true, distanceCache, weights }) * 0.04,
        ended: true,
      });
    }

    beams = nextBeams
      .sort((a, b) => b.score - a.score || a.actions.length - b.actions.length)
      .slice(0, beamWidth);

    if (beams.every((beam) => beam.ended)) break;
  }

  if (projectPhaseEnd) {
    beams = beams.slice(0, projectionBeamLimit).map((beam) => {
      const projection = phaseEndProjectionScore(beam.environment, {
        side,
        distanceCache,
        weights,
        candidateLimit: projectionCandidateLimit,
      });
      return {
        ...beam,
        projection,
        projectedScore: beam.score + projection.score * projectionWeight,
      };
    }).sort((a, b) => (b.projectedScore ?? b.score) - (a.projectedScore ?? a.score) || a.actions.length - b.actions.length);
  }

  return beams[0] || null;
}

export function phaseEndProjectionScore(environment, options = {}) {
  const side = options.side || activeSide(environment);
  const endPhase = generateLegalActions(environment, { includeChanceActions: false })
    .find((action) => action.type === ENV_ACTION.END_PHASE);
  if (!side || !endPhase) {
    return { score: 0, reason: "no_phase_end" };
  }

  const ended = applyEnvironmentAction(environment, endPhase, quietApplyOptions());
  if (!ended.ok) return { score: 0, reason: "phase_end_illegal" };
  if (ended.state.winner?.side === side) return { score: 9000, reason: "phase_end_win" };
  if (ended.state.winner && ended.state.winner.side !== side) return { score: -9000, reason: "phase_end_loss" };

  let score = defaultMovementStateScore(ended.environment, options) * 0.08;
  let reason = "state";
  const nextSide = activeSide(ended.environment);
  if (nextSide === side) {
    const combat = combatProjectionScore(ended.environment, side, options);
    score += combat.score;
    reason = combat.reason;
    const opponentAfterPass = opponentImmediateAfterPassingOwnPhaseScore(ended.environment, side, options);
    score += opponentAfterPass.score;
    if (opponentAfterPass.score < 0) reason = `${reason}+${opponentAfterPass.reason}`;
  } else if (nextSide && nextSide !== side) {
    const opponent = opponentImmediateWinScore(ended.environment, side, options);
    score += opponent.score;
    if (opponent.score < 0) reason = opponent.reason;
  }
  return { score, reason };
}

export function defaultMovementActionScore(environment, action, applied, options = {}) {
  const side = options.side || activeSide(environment);
  const unitBefore = unitById(environment.state.units, action.unitId);
  const unitAfter = unitById(applied.state.units, action.unitId);
  if (!unitBefore || !unitAfter) return 0;
  if (applied.state.winner?.side === side) return 10000;
  return scoreFeatureVector(movementActionFeatureVector(environment, action, {
    side,
    unitBefore,
    unitAfter,
    distanceCache: options.distanceCache,
  }), options.weights);
}

function combatProjectionScore(environment, side, options = {}) {
  const actions = generateLegalActions(environment, { includeChanceActions: false })
    .filter((action) => action.type === ENV_ACTION.DECLARE_COMBAT)
    .slice(0, Number(options.candidateLimit || DEFAULT_PROJECTION_CANDIDATE_LIMIT));
  let best = null;
  for (const action of actions) {
    const profile = combatEliminationProfile(environment, action);
    if (!profile) continue;
    const sealed = Number(profile.sealedRetreatRolls || 0);
    const direct = Number(profile.directDefenderEliminationRolls || 0);
    const adverse = Number(profile.attackerAdverseRolls || 0);
    const target = Number(profile.targetValue || 0);
    let score = direct * 130 + sealed * 210 + target * 8 - adverse * (side === "axis" ? 105 : 75);
    if (profile.guaranteedDefenderElimination) score += 1850 + target * 10;
    else if (sealed > 0 && profile.defenderEliminationRolls >= adverse) score += 760 + sealed * 110;
    if (!best || score > best.score) {
      best = {
        score,
        action,
        profile,
        reason: profile.guaranteedDefenderElimination ? "projected_guaranteed_elimination" : sealed > 0 ? "projected_sealed_retreat" : "projected_combat",
      };
    }
  }
  return best || { score: 0, reason: "no_projected_combat" };
}

function opponentImmediateAfterPassingOwnPhaseScore(environment, side, options = {}) {
  const passed = passPhaseWithoutActions(environment);
  if (!passed.ok) return { score: 0, reason: "own_phase_pass_illegal" };
  if (passed.state.winner?.side === side) return { score: 2000, reason: "own_phase_pass_win" };
  if (passed.state.winner && passed.state.winner.side !== side) return { score: -7000, reason: "opponent_immediate_win_after_pass" };
  return opponentImmediateWinScore(passed.environment, side, options);
}

function passPhaseWithoutActions(environment) {
  let current = environment;
  let actions = generateLegalActions(current, { includeChanceActions: false });
  const finishDeclarations = actions.find((action) => action.type === ENV_ACTION.FINISH_DECLARATIONS);
  if (finishDeclarations) {
    const finished = applyEnvironmentAction(current, finishDeclarations, quietApplyOptions());
    if (!finished.ok) return finished;
    current = finished.environment;
    if (finished.state.winner) return finished;
    actions = generateLegalActions(current, { includeChanceActions: false });
  }
  const endPhase = actions.find((action) => action.type === ENV_ACTION.END_PHASE);
  if (!endPhase) return { ok: false, reason: "phase_has_required_actions", state: current.state, environment: current };
  return applyEnvironmentAction(current, endPhase, quietApplyOptions());
}

function opponentImmediateWinScore(environment, side, options = {}) {
  const opponent = side === "axis" ? "allied" : "axis";
  if (activeSide(environment) !== opponent) return { score: 0, reason: "opponent_not_active" };
  const actions = generateLegalActions(environment, { includeChanceActions: false })
    .slice(0, Number(options.candidateLimit || DEFAULT_PROJECTION_CANDIDATE_LIMIT));
  for (const action of actions) {
    const applied = applyEnvironmentAction(environment, action, quietApplyOptions());
    if (applied.ok && applied.state.winner?.side === opponent) {
      return {
        score: -8200,
        reason: "opponent_immediate_win",
        action,
      };
    }
  }
  return { score: 0, reason: "opponent_no_immediate_win" };
}

function quietApplyOptions() {
  return {
    enrichEvents: false,
    previousState: false,
    cloneResultState: false,
  };
}

export function defaultMovementStateScore(environment, options = {}) {
  const side = options.side || activeSide(environment);
  if (environment.state.winner?.side === side) return 10000;
  if (environment.state.winner && environment.state.winner.side !== side) return -10000;

  return scoreFeatureVector(movementStateFeatureVector(environment, {
    side,
    distanceCache: options.distanceCache,
  }), options.weights);
}

export function mergePhaseSearchWeights(overrides = null) {
  return Object.freeze({
    ...DEFAULT_PHASE_SEARCH_WEIGHTS,
    ...(overrides || {}),
  });
}

export function scoreFeatureVector(features, weights = null) {
  const merged = mergePhaseSearchWeights(weights);
  return Object.entries(features || {}).reduce((sum, [key, value]) => (
    sum + Number(value || 0) * Number(merged[key] || 0)
  ), 0);
}

export function movementStateFeatureVector(environment, options = {}) {
  const side = options.side || activeSide(environment);
  if (side === "axis") {
    const objectives = axisObjectives(environment);
    const axisUnits = liveUnits(environment.state.units).filter((unit) => unit.side === "axis");
    const minDistance = Math.min(Infinity, ...axisUnits.map((unit) => nearestDistance(environment, unit.hexId, objectives, options.distanceCache)));
    const occupied = axisUnits.some((unit) => objectives.includes(unit.hexId));
    return {
      axisStateObjectiveHeld: occupied ? 1 : 0,
      axisStateObjectiveDistance: Math.min(18, minDistance),
    };
  }

  const objectives = axisObjectives(environment);
  const axisUnits = liveUnits(environment.state.units).filter((unit) => unit.side === "axis");
  const minAxisDistance = Math.min(Infinity, ...axisUnits.map((unit) => nearestDistance(environment, unit.hexId, objectives, options.distanceCache)));
  const wallLinks = linkedWallScore(environment, "allied", options.distanceCache);
  return {
    alliedStateAxisDistance: Math.min(18, minAxisDistance),
    alliedStateWallLink: wallLinks,
  };
}

export function movementActionFeatureVector(environment, action, options = {}) {
  const side = options.side || activeSide(environment);
  const unitBefore = options.unitBefore || unitById(environment.state.units, action.unitId);
  const unitAfter = options.unitAfter || { ...unitBefore, hexId: action.toHexId };
  if (!unitBefore || !unitAfter) return {};
  if (side === "axis") return axisMoveActionFeatures(environment, action, unitBefore, unitAfter, options.distanceCache);
  return alliedMoveActionFeatures(environment, action, unitBefore, unitAfter, options.distanceCache);
}

function axisMoveActionFeatures(environment, action, unitBefore, unitAfter, distanceCache) {
  const objectives = axisObjectives(environment);
  const currentDistance = nearestDistance(environment, unitBefore.hexId, objectives, distanceCache);
  const nextDistance = nearestDistance(environment, unitAfter.hexId, objectives, distanceCache);
  const progress = currentDistance - nextDistance;
  const mobileFit = Number(unitBefore.movement || 0) >= 8 ? 1 + DEFAULT_PHASE_SEARCH_WEIGHTS.axisMobileFactor : 0.82;
  const combatFit = Number(unitBefore.combat || 0) >= 4 ? 1 + DEFAULT_PHASE_SEARCH_WEIGHTS.axisCombatFactor : 0.86;
  const objectiveSecurity = objectives.includes(unitAfter.hexId)
    ? axisObjectiveSecurityFeatures(environment, unitBefore, unitAfter.hexId)
    : {};
  return {
    axisProgress: progress * mobileFit * combatFit,
    axisObjective: objectives.includes(unitAfter.hexId) ? 1 : 0,
    ...objectiveSecurity,
    axisAdjacent: nextDistance === 1 ? 1 : 0,
    axisNear: nextDistance === 2 ? 1 : 0,
    axisReserve: Number(action.route?.remaining || 0),
  };
}

function axisObjectiveSecurityFeatures(environment, movingUnit, objectiveHexId) {
  let supportStrength = 0;
  let supportCount = 0;
  let counterThreat = 0;
  for (const unit of liveUnits(environment.state.units)) {
    if (unit.id === movingUnit.id || unit.eliminated || unit.disrupted) continue;
    const range = distanceBetween(environment, unit.hexId, objectiveHexId);
    if (range > 2) continue;
    if (unit.side === movingUnit.side) {
      supportStrength += Number(unit.combat || 0);
      supportCount += 1;
    } else {
      counterThreat += Number(unit.combat || 0);
    }
  }
  const holdStrength = Number(movingUnit.combat || 0) + supportStrength;
  return {
    axisObjectiveSupport: supportStrength,
    axisObjectiveCounterThreat: Math.max(0, counterThreat - holdStrength),
    axisObjectiveUnsupportedPenalty: supportCount <= 0 && counterThreat > holdStrength ? 1 : 0,
  };
}

function alliedMoveActionFeatures(environment, action, unitBefore, unitAfter, distanceCache) {
  const objectives = axisObjectives(environment);
  const currentObjectiveDistance = nearestDistance(environment, unitBefore.hexId, objectives, distanceCache);
  const nextObjectiveDistance = nearestDistance(environment, unitAfter.hexId, objectives, distanceCache);
  const axisPressure = nearestAxisPressure(environment, unitAfter.hexId, distanceCache);
  return {
    alliedForwardBand: nextObjectiveDistance >= 4 && nextObjectiveDistance <= 8 ? 1 : 0,
    alliedForwardBandPenalty: nextObjectiveDistance >= 4 && nextObjectiveDistance <= 8 ? Math.abs(6 - nextObjectiveDistance) : 0,
    alliedObjectiveHugPenalty: nextObjectiveDistance <= 2 ? 1 : 0,
    alliedContact: axisPressure >= 2 && axisPressure <= 5 ? 1 : 0,
    alliedContactPenalty: axisPressure >= 2 && axisPressure <= 5 ? Math.abs(3 - axisPressure) : 0,
    alliedLine: linkedWallScoreForHex(environment, "allied", unitAfter.hexId, unitBefore.id, distanceCache),
    alliedDrift: Math.max(-2, currentObjectiveDistance - nextObjectiveDistance),
    alliedCombat: Number(unitBefore.combat || 0),
  };
}

function quickMoveActionScore(environment, action, side, distanceCache, weights) {
  const unit = unitById(environment.state.units, action.unitId);
  if (!unit) return 0;
  if (side === "axis") {
    const objectives = axisObjectives(environment);
    return (nearestDistance(environment, unit.hexId, objectives, distanceCache) - nearestDistance(environment, action.toHexId, objectives, distanceCache)) * 100
      + (objectives.includes(action.toHexId) ? 10000 : 0)
      + Number(unit.movement || 0) * 2
      + Number(unit.combat || 0);
  }
  return linkedWallScoreForHex(environment, side, action.toHexId, unit.id, distanceCache) * 80
    + Math.max(0, 8 - nearestDistance(environment, action.toHexId, axisObjectives(environment), distanceCache)) * 20
    + Number(unit.combat || 0) * 5;
}

function axisObjectives(environment) {
  return [
    ...(environment.scenario?.objectives?.alamHalfaRidge || []),
    ...(environment.scenario?.objectives?.coastalRoadEast || []),
  ];
}

function nearestDistance(environment, fromHexId, targets, distanceCache = null) {
  if (!targets?.length) return Infinity;
  return Math.min(...targets.map((hexId) => distanceBetween(environment, fromHexId, hexId, distanceCache)));
}

function distanceBetween(environment, fromHexId, toHexId, distanceCache = null) {
  if (!distanceCache) return hexDistance(environment.board, fromHexId, toHexId);
  const key = `from:${toHexId}`;
  if (!distanceCache.has(key)) {
    distanceCache.set(key, distanceMapFrom(environment, toHexId));
  }
  return distanceCache.get(key).get(fromHexId) ?? Infinity;
}

function distanceMapFrom(environment, originHexId) {
  const distances = new Map([[originHexId, 0]]);
  const queue = [originHexId];
  while (queue.length) {
    const current = queue.shift();
    const nextDistance = distances.get(current) + 1;
    for (const next of neighborsOf(environment.board, current)) {
      if (distances.has(next)) continue;
      distances.set(next, nextDistance);
      queue.push(next);
    }
  }
  return distances;
}

function nearestAxisPressure(environment, hexId, distanceCache = null) {
  const axisUnits = liveUnits(environment.state.units).filter((unit) => unit.side === "axis" && !unit.disrupted);
  return Math.min(Infinity, ...axisUnits.map((unit) => distanceBetween(environment, unit.hexId, hexId, distanceCache)));
}

function linkedWallScore(environment, side, distanceCache = null) {
  return liveUnits(environment.state.units)
    .filter((unit) => unit.side === side && !unit.disrupted)
    .reduce((sum, unit) => sum + linkedWallScoreForHex(environment, side, unit.hexId, unit.id, distanceCache), 0);
}

function linkedWallScoreForHex(environment, side, hexId, ignoreUnitId = null, distanceCache = null) {
  return liveUnits(environment.state.units)
    .filter((unit) => unit.id !== ignoreUnitId && unit.side === side && !unit.disrupted)
    .reduce((sum, unit) => {
      const distance = distanceBetween(environment, unit.hexId, hexId, distanceCache);
      if (distance === 2) return sum + 2;
      if (distance === 3) return sum + 1;
      if (distance === 1) return sum - 1;
      return sum;
    }, 0);
}
