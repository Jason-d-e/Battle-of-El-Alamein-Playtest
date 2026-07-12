import {
  activeSide,
  currentPhase,
  environmentContext,
  environmentMetrics,
  hexDistance,
  liveUnits,
} from "../core/index.js";

const OPPOSITE_SIDE = Object.freeze({ axis: "allied", allied: "axis" });
const LOCAL_FORCE_RADIUS = 3;

export const DEFAULT_SITUATION_WEIGHTS = Object.freeze({
  materialBalance: 0.018,
  unitBalance: 0.028,
  axisObjectiveHeld: 0.72,
  axisObjectiveProgress: 0.052,
  axisObjectivePressure: 0.046,
  axisObjectiveLocalAdvantage: 0.014,
  alliedExitPressure: 0.055,
  alliedExitLocalAdvantage: 0.016,
  friendlyCohesion: 0.012,
  enemyCohesion: -0.01,
  friendlyThreats: 0.018,
  enemyThreats: -0.022,
});

export function analyzeSituation(environment, options = {}) {
  const side = options.side || activeSide(environment);
  const enemy = OPPOSITE_SIDE[side] || null;
  const metrics = environmentMetrics(environment);
  const context = environmentContext(environment);
  const live = liveUnits(environment.state.units);
  const friendly = live.filter((unit) => unit.side === side);
  const enemies = enemy ? live.filter((unit) => unit.side === enemy) : [];
  const axisUnits = live.filter((unit) => unit.side === "axis");
  const alliedUnits = live.filter((unit) => unit.side === "allied");
  const objectiveHexes = axisObjectiveHexes(environment);
  const exitHexes = environment.scenario?.objectives?.alliedWestExitEdge || [];
  const axisObjectiveDistance = nearestUnitDistance(environment, axisUnits, objectiveHexes);
  const alliedExitDistance = nearestUnitDistance(environment, alliedUnits, exitHexes);
  const axisObjectiveLocalAdvantage = localForceAdvantage(environment, axisUnits, alliedUnits, objectiveHexes);
  const alliedExitLocalAdvantage = localForceAdvantage(environment, alliedUnits, axisUnits, exitHexes);
  const objectiveStatus = metrics.objectiveStatus || {};
  const axisObjectiveHeld = Number(Boolean(
    objectiveStatus.ridgeOccupied
    || objectiveStatus.roadOccupied
    || objectiveStatus.ridgeControl
    || objectiveStatus.roadCut
  ));
  const material = metrics.combatStrength || { axis: 0, allied: 0 };
  const friendlyStrength = Number(material[side] || 0);
  const enemyStrength = enemy ? Number(material[enemy] || 0) : 0;
  const friendlyThreats = adjacentThreatCount(environment, friendly, enemies);
  const enemyThreats = adjacentThreatCount(environment, enemies, friendly);
  const friendlyCohesion = cohesionScore(environment, friendly);
  const enemyCohesion = cohesionScore(environment, enemies);

  return {
    side,
    enemy,
    turn: Number(environment.state.turn || 1),
    phaseId: currentPhase(environment)?.id || null,
    phaseType: currentPhase(environment)?.type || null,
    activeSide: activeSide(environment),
    winner: environment.state.winner ? { ...environment.state.winner } : null,
    objectiveStatus,
    objectiveHexes,
    exitHexes,
    features: {
      materialBalance: friendlyStrength - enemyStrength,
      unitBalance: friendly.length - enemies.length,
      axisObjectiveHeld,
      axisObjectiveDistance: finiteOrCap(axisObjectiveDistance, 24),
      axisObjectiveProgress: objectiveProgress(axisObjectiveDistance),
      axisObjectivePressure: objectivePressure(axisObjectiveDistance),
      axisObjectiveLocalAdvantage,
      alliedExitDistance: finiteOrCap(alliedExitDistance, 24),
      alliedExitPressure: objectivePressure(alliedExitDistance),
      alliedExitLocalAdvantage,
      friendlyCohesion,
      enemyCohesion,
      friendlyThreats,
      enemyThreats,
    },
    metrics,
    context,
  };
}

export function evaluateSituation(environment, options = {}) {
  const side = options.side || activeSide(environment);
  const winner = environment.state?.winner;
  if (winner?.side === side) return 1;
  if (winner && winner.side !== side) return -1;

  const analysis = options.analysis || analyzeSituation(environment, { side });
  const weights = {
    ...DEFAULT_SITUATION_WEIGHTS,
    ...(options.weights || {}),
  };
  const features = analysis.features;
  let score = 0;
  score += features.materialBalance * weights.materialBalance;
  score += features.unitBalance * weights.unitBalance;
  score += features.friendlyCohesion * weights.friendlyCohesion;
  score += features.enemyCohesion * weights.enemyCohesion;
  score += features.friendlyThreats * weights.friendlyThreats;
  score += features.enemyThreats * weights.enemyThreats;

  if (side === "axis") {
    score += features.axisObjectiveHeld * weights.axisObjectiveHeld;
    score += features.axisObjectiveProgress * weights.axisObjectiveProgress;
    score += features.axisObjectivePressure * weights.axisObjectivePressure;
    score += features.axisObjectiveLocalAdvantage * weights.axisObjectiveLocalAdvantage;
    score -= features.alliedExitPressure * weights.alliedExitPressure;
    score -= features.alliedExitLocalAdvantage * weights.alliedExitLocalAdvantage;
  } else {
    score -= features.axisObjectiveHeld * weights.axisObjectiveHeld;
    score -= features.axisObjectiveProgress * weights.axisObjectiveProgress;
    score -= features.axisObjectivePressure * weights.axisObjectivePressure;
    score -= features.axisObjectiveLocalAdvantage * weights.axisObjectiveLocalAdvantage;
    score += features.alliedExitPressure * weights.alliedExitPressure;
    score += features.alliedExitLocalAdvantage * weights.alliedExitLocalAdvantage;
  }

  return clamp(score, -1, 1);
}

export function axisObjectiveHexes(environment) {
  const objectives = environment.scenario?.objectives || {};
  return [
    ...(objectives.alamHalfaRidge || []),
    ...(objectives.coastalRoadEast || []),
  ];
}

export function nearestDistanceToAny(environment, fromHexId, targets) {
  if (!fromHexId || !targets?.length) return Infinity;
  return Math.min(...targets.map((targetHexId) => hexDistance(environment.board, fromHexId, targetHexId)));
}

function nearestUnitDistance(environment, units, targets) {
  if (!units.length || !targets?.length) return Infinity;
  return Math.min(...units.map((unit) => nearestDistanceToAny(environment, unit.hexId, targets)));
}

function objectiveProgress(distance) {
  if (!Number.isFinite(distance)) return 0;
  return Math.max(0, 18 - distance);
}

function objectivePressure(distance) {
  if (!Number.isFinite(distance)) return 0;
  return Math.max(0, 8 - distance);
}

function cohesionScore(environment, units) {
  let score = 0;
  for (let outer = 0; outer < units.length; outer += 1) {
    for (let inner = outer + 1; inner < units.length; inner += 1) {
      const distance = hexDistance(environment.board, units[outer].hexId, units[inner].hexId);
      if (distance === 2) score += 2;
      else if (distance === 3) score += 1;
      else if (distance === 1) score -= 0.5;
    }
  }
  return score;
}

function adjacentThreatCount(environment, attackers, defenders) {
  let count = 0;
  for (const attacker of attackers) {
    for (const defender of defenders) {
      if (hexDistance(environment.board, attacker.hexId, defender.hexId) === 1) {
        count += Number(attacker.combat || 0);
      }
    }
  }
  return count;
}

function localForceAdvantage(environment, friendly, enemies, targets) {
  if (!targets?.length) return 0;
  return localForceMass(environment, friendly, targets) - localForceMass(environment, enemies, targets);
}

function localForceMass(environment, units, targets) {
  let mass = 0;
  for (const unit of units || []) {
    const distance = nearestDistanceToAny(environment, unit.hexId, targets);
    if (distance <= LOCAL_FORCE_RADIUS) {
      const proximity = (LOCAL_FORCE_RADIUS + 1 - distance) / (LOCAL_FORCE_RADIUS + 1);
      mass += Number(unit.combat ?? unit.attack ?? 0) * proximity;
    }
  }
  return round(mass, 4);
}

function finiteOrCap(value, cap) {
  return Number.isFinite(value) ? Math.min(cap, value) : cap;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}
