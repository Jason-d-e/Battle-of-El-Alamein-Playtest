import {
  ENV_ACTION,
  activeSide,
  applyEnvironmentAction,
  calculateOdds,
  compactAction,
  generateLegalActions,
  stateHash,
  unitById,
} from "../core/index.js";
import {
  analyzeSituation,
  axisObjectiveHexes,
  evaluateSituation,
  nearestDistanceToAny,
} from "./ai-situation.js";
import {
  actionPolicyFeatures,
  scoreAlphaPolicyLogit,
  scoreAlphaValueSample,
} from "./ai-alpha-training.js";

const DEFAULT_ITERATIONS = 96;
const DEFAULT_MAX_DEPTH = 5;
const DEFAULT_ACTION_LIMIT = 24;
const DEFAULT_PRE_APPLY_LIMIT = 24;
const DEFAULT_EXPLORATION = 1.35;
const DEFAULT_ROOT_DIRICHLET_ALPHA = 0.3;
const OPPOSITE_SIDE = Object.freeze({ axis: "allied", allied: "axis" });

export function analyzePosition(environment, options = {}) {
  const side = options.side || activeSide(environment);
  const valueModel = options.model?.value || options.weights?.valueModel || null;
  const valueWeights = options.weights?.value || null;
  const root = createNode({
    environment,
    side,
    depth: 0,
    prior: 1,
    action: null,
    parent: null,
    seedValue: evaluateAlphaSearchValue(environment, { side, valueModel, valueWeights }),
  });
  const startedAt = now();
  const iterations = Math.max(1, Number(options.iterations || DEFAULT_ITERATIONS));
  const timeBudgetMs = Number(options.timeBudgetMs || 0);
  const maxDepth = Math.max(1, Number(options.maxDepth || DEFAULT_MAX_DEPTH));
  const actionLimit = Number(options.actionLimit || DEFAULT_ACTION_LIMIT);
  const preApplyLimit = Number(options.preApplyLimit || defaultPreApplyLimit(actionLimit));
  const pvLimit = Math.max(0, Number(options.pvLimit || maxDepth));
  const candidateLineLimit = Math.max(0, Number(options.candidateLineLimit ?? options.lineLimit ?? 3));
  const progressInterval = Math.max(0, Math.floor(Number(options.progressInterval ?? 0)));
  const progressLimit = Math.max(0, Math.floor(Number(options.progressLimit ?? 6)));
  const policyModel = options.model?.policy || options.weights?.policy || null;
  const policyWeight = Number(options.policyWeight ?? (policyModel ? 1.6 : 0));
  const priorSideMode = "active-side";
  let completedIterations = 0;
  const progress = [];

  for (let index = 0; index < iterations; index += 1) {
    if (timeBudgetMs > 0 && now() - startedAt >= timeBudgetMs) break;
    const path = selectPath(root, {
      side,
      maxDepth,
      actionLimit,
      preApplyLimit,
      exploration: Number(options.exploration || DEFAULT_EXPLORATION),
      weights: options.weights || null,
      valueModel,
      valueWeights,
      policyModel,
      policyWeight,
      rootNoiseWeight: Number(options.rootNoiseWeight || 0),
      rootDirichletAlpha: Number(options.rootDirichletAlpha || DEFAULT_ROOT_DIRICHLET_ALPHA),
      rootNoise: options.rootNoise || null,
      rootNoiseRandom: typeof options.rootNoiseRandom === "function" ? options.rootNoiseRandom : null,
      priorSideMode,
    });
    const leaf = path[path.length - 1];
    const value = evaluateAlphaSearchValue(leaf.environment, { side, valueModel, valueWeights });
    backpropagate(path, value);
    completedIterations += 1;
    if (progressInterval > 0 && completedIterations % progressInterval === 0) {
      appendProgressSnapshot(progress, root, {
        iterations: completedIterations,
        elapsedMs: now() - startedAt,
        pvLimit,
        candidateLineLimit,
        progressLimit,
      });
    }
  }

  if (!root.expanded) {
    expandNode(root, {
      side,
      maxDepth,
      actionLimit,
      preApplyLimit,
      weights: options.weights || null,
      valueModel,
      valueWeights,
      policyModel,
      policyWeight,
      rootNoiseWeight: Number(options.rootNoiseWeight || 0),
      rootDirichletAlpha: Number(options.rootDirichletAlpha || DEFAULT_ROOT_DIRICHLET_ALPHA),
      rootNoise: options.rootNoise || null,
      rootNoiseRandom: typeof options.rootNoiseRandom === "function" ? options.rootNoiseRandom : null,
      priorSideMode,
    });
  }

  if (progressInterval > 0) {
    appendProgressSnapshot(progress, root, {
      iterations: completedIterations,
      elapsedMs: now() - startedAt,
      pvLimit,
      candidateLineLimit,
      progressLimit,
      force: true,
    });
  }

  const policy = rootPolicy(root);
  const best = policy[0] || null;
  const elapsedMs = now() - startedAt;

  return {
    schema: "zizi-el-alamein-alpha-analysis-v1",
    side,
    stateHash: stateHash(environment),
    rootValue: evaluateAlphaSearchValue(environment, { side, valueModel, valueWeights }),
    situation: analyzeSituation(environment, { side }),
    ...(options.includeStateSnapshot ? { initialState: cloneJsonLike(environment.state) } : {}),
    bestAction: best?.action || null,
    policy,
    principalVariation: principalVariation(root, pvLimit),
    candidateLines: candidateLines(root, candidateLineLimit, pvLimit),
    ...(progress.length ? { progress } : {}),
    search: {
      iterations: completedIterations,
      rootVisits: root.visits,
      rootChildren: root.children.length,
      maxDepth,
      actionLimit,
      preApplyLimit,
      elapsedMs,
      priorSideMode,
      rootNoise: Number(options.rootNoiseWeight || 0) > 0
        ? {
          weight: Number(options.rootNoiseWeight || 0),
          alpha: Number(options.rootDirichletAlpha || DEFAULT_ROOT_DIRICHLET_ALPHA),
        }
        : null,
    },
    requiresChance: isChanceOnlyRoot(root),
  };
}

export function chooseAlphaAction(environment, options = {}) {
  const analysis = analyzePosition(environment, options);
  return analysis.requiresChance ? null : analysis.bestAction;
}

export function makeSearchTrainingSample(analysis, outcomeSide = null) {
  const policyVisits = (analysis.policy || []).reduce((sum, item) => sum + Number(item.visits || 0), 0);
  const policyPrior = (analysis.policy || []).reduce((sum, item) => sum + Number(item.prior || 0), 0);
  return {
    schema: "zizi-el-alamein-alpha-training-sample-v1",
    stateHash: analysis.stateHash,
    side: analysis.side,
    turn: analysis.situation?.turn ?? null,
    phaseId: analysis.situation?.phaseId || null,
    ...(analysis.initialState ? { initialState: cloneJsonLike(analysis.initialState) } : {}),
    features: { ...(analysis.situation?.features || {}) },
    rootValue: analysis.rootValue,
    outcome: outcomeSide ? outcomeValue(outcomeSide, analysis.side) : null,
    policy: (analysis.policy || []).map((item) => ({
      action: item.action,
      visitShare: policyTargetShare(item, {
        visits: policyVisits,
        prior: policyPrior,
        count: analysis.policy?.length || 0,
      }),
      visits: item.visits,
      q: item.q,
      prior: item.prior,
    })),
  };
}

function policyTargetShare(item, totals) {
  if (totals.visits > 0) return Number((Number(item.visits || 0) / totals.visits).toFixed(6));
  if (totals.prior > 0) return Number((Number(item.prior || 0) / totals.prior).toFixed(6));
  return totals.count > 0 ? Number((1 / totals.count).toFixed(6)) : 0;
}

export function actionKey(action) {
  if (!action) return "";
  const compact = compactAction(action);
  const keys = Object.keys(compact).sort();
  return keys.map((key) => `${key}:${stableValue(compact[key])}`).join("|");
}

export function mixAlphaRootPriors(priors, noise, weight = 0.25) {
  const base = normalizeProbabilityVector(priors);
  if (!base.length) return [];
  const blendWeight = clamp(Number(weight || 0), 0, 1);
  if (!(blendWeight > 0)) return base;
  const perturbation = normalizeProbabilityVector(padVector(noise, base.length));
  if (!perturbation.length) return base;
  return normalizeProbabilityVector(base.map((prior, index) => (
    prior * (1 - blendWeight) + perturbation[index] * blendWeight
  )));
}

export function generateAlphaRootNoise(count, alpha = DEFAULT_ROOT_DIRICHLET_ALPHA, random = null) {
  const length = Math.max(0, Math.floor(Number(count || 0)));
  if (!length) return [];
  if (typeof random !== "function") return Array.from({ length }, () => 1 / length);
  const concentration = Math.max(0.001, Number(alpha || DEFAULT_ROOT_DIRICHLET_ALPHA));
  const raw = Array.from({ length }, () => sampleGamma(concentration, random));
  return normalizeProbabilityVector(raw);
}

function createNode({ environment, side, depth, prior, action, parent, seedValue = null }) {
  return {
    environment,
    side,
    depth,
    prior,
    action,
    parent,
    seedValue: seedValue ?? evaluateSituation(environment, { side }),
    valueSum: 0,
    visits: 0,
    children: [],
    expanded: false,
    terminal: Boolean(environment.state?.winner),
  };
}

function selectPath(root, options) {
  const path = [root];
  let node = root;
  while (!node.terminal && node.depth < options.maxDepth) {
    if (!node.expanded) {
      expandNode(node, options);
      break;
    }
    if (!node.children.length) break;
    node = selectChild(node, options);
    path.push(node);
  }
  return path;
}

function expandNode(node, options) {
  if (node.expanded || node.terminal || node.depth >= options.maxDepth) {
    node.expanded = true;
    return;
  }

  const legalActions = generateLegalActions(node.environment, { includeChanceActions: true });
  const priorSide = priorSideForNode(node, options);
  const priorOptions = {
    ...options,
    side: priorSide,
  };
  const candidateActions = preselectActions(node.environment, legalActions, priorOptions);
  const currentValue = evaluateAlphaSearchValue(node.environment, {
    side: priorSide,
    valueModel: options.valueModel,
    valueWeights: options.valueWeights || null,
  });
  const candidates = [];
  for (const action of candidateActions) {
    const applied = applyEnvironmentAction(node.environment, action, quietApplyOptions());
    if (!applied.ok) continue;
    const priorScore = actionPriorScore(node.environment, action, {
      ...priorOptions,
      side: priorSide,
      currentValue,
      appliedEnvironment: applied.environment,
    });
    candidates.push({
      action: applied.action,
      environment: applied.environment,
      priorScore,
      seedValue: evaluateAlphaSearchValue(applied.environment, {
        side: options.side,
        valueModel: options.valueModel,
        valueWeights: options.valueWeights || null,
      }),
    });
  }

  const limited = candidates
    .sort((a, b) => b.priorScore - a.priorScore || actionKey(a.action).localeCompare(actionKey(b.action)))
    .slice(0, Number(options.actionLimit || DEFAULT_ACTION_LIMIT));
  const priors = rootAdjustedPriors(
    normalizePriors(limited.map((candidate) => candidate.priorScore)),
    node,
    options,
  );
  node.children = limited.map((candidate, index) => createNode({
    environment: candidate.environment,
    side: options.side,
    depth: node.depth + 1,
    prior: priors[index],
    action: candidate.action,
    parent: node,
    seedValue: candidate.seedValue,
  }));
  node.expanded = true;
}

function priorSideForNode(node, options) {
  return activeSide(node.environment) || options.side;
}

function selectChild(node, options) {
  if (isChanceNode(node)) {
    return node.children
      .slice()
      .sort((a, b) => a.visits - b.visits || actionKey(a.action).localeCompare(actionKey(b.action)))[0];
  }

  const maximizing = activeSide(node.environment) === options.side;
  let best = null;
  for (const child of node.children) {
    const q = child.visits ? child.valueSum / child.visits : child.seedValue;
    const exploration = options.exploration * child.prior * Math.sqrt(node.visits + 1) / (1 + child.visits);
    const score = maximizing ? q + exploration : -q + exploration;
    if (!best || score > best.score) best = { child, score };
  }
  return best.child;
}

function actionPriorScore(environment, action, options) {
  if (isChanceAction(action)) return 0;
  const afterValue = evaluateAlphaSearchValue(options.appliedEnvironment, {
    side: options.side,
    valueModel: options.valueModel,
    valueWeights: options.valueWeights || null,
  });
  let score = (afterValue - options.currentValue) * 4;
  if (options.appliedEnvironment.state.winner?.side === options.side) score += 8;
  else if (options.appliedEnvironment.state.winner) score -= 8;

  return score + preActionPriorScore(environment, action, options);
}

function evaluateAlphaSearchValue(environment, options = {}) {
  const side = options.side || activeSide(environment);
  if (options.valueModel?.weights && Object.keys(options.valueModel.weights).length) {
    const situation = analyzeSituation(environment, { side });
    return scoreAlphaValueSample({
      side,
      turn: situation.turn,
      phaseId: situation.phaseId,
      features: situation.features,
    }, options.valueModel);
  }
  return evaluateSituation(environment, { side, weights: options.valueWeights || null });
}

function preselectActions(environment, legalActions, options) {
  if (legalActions.every(isChanceAction)) return legalActions;
  return legalActions
    .map((action) => ({
      action,
      score: preActionPriorScore(environment, action, options),
    }))
    .sort((a, b) => b.score - a.score || actionKey(a.action).localeCompare(actionKey(b.action)))
    .slice(0, Number(options.preApplyLimit || DEFAULT_PRE_APPLY_LIMIT))
    .map((candidate) => candidate.action);
}

function preActionPriorScore(environment, action, options) {
  let score = 0;
  if (action.type === ENV_ACTION.MOVE_UNIT) score += movePriorScore(environment, action, options.side);
  else if (action.type === ENV_ACTION.DECLARE_COMBAT) score += combatPriorScore(environment, action, options.side);
  else if (action.type === ENV_ACTION.END_PHASE) score -= 0.15;
  else if (action.type === ENV_ACTION.FINISH_DECLARATIONS) score -= 0.05;
  else if (action.type === ENV_ACTION.ADVANCE_UNIT) score += 0.25;
  else if (action.type === ENV_ACTION.SKIP_ADVANCE) score -= 0.1;
  return score + modelPolicyPriorScore(environment, action, options);
}

function movePriorScore(environment, action, side) {
  const unit = unitById(environment.state.units, action.unitId);
  if (!unit) return 0;
  const objectives = axisObjectiveHexes(environment);
  const exitHexes = environment.scenario?.objectives?.alliedWestExitEdge || [];
  const targetHexes = unit.side === "axis" ? objectives : exitHexes;
  const currentDistance = nearestDistanceToAny(environment, unit.hexId, targetHexes);
  const nextDistance = nearestDistanceToAny(environment, action.toHexId, targetHexes);
  const progress = finiteProgress(currentDistance, nextDistance);
  let score = progress * 0.34 + Number(action.route?.remaining || 0) * 0.025;
  if (unit.side === "axis" && objectives.includes(action.toHexId)) score += 4.5;
  if (unit.side === "allied" && exitHexes.includes(action.toHexId)) score += 3.2;
  if (unit.side !== side) score *= -1;
  return score;
}

function combatPriorScore(environment, action, side) {
  const defender = unitById(environment.state.units, action.defenderId);
  const attackers = (action.attackerIds || []).map((id) => unitById(environment.state.units, id)).filter(Boolean);
  if (!defender || !attackers.length) return 0;
  const odds = calculateOdds({
    board: environment.board,
    scenario: environment.scenario,
    rules: environment.rules,
    units: environment.state.units,
    state: environment.state,
  }, attackers, defender);
  const attackerSide = attackers[0]?.side || activeSide(environment);
  const targetValue = Number(defender.combat || 0) * 0.12 + Number(defender.movement || 0) * 0.035;
  const overcommit = Math.max(0, attackers.length - 2) * 0.12;
  const score = odds.columnIndex * 0.22 + targetValue - overcommit;
  return attackerSide === side ? score : -score;
}

function modelPolicyPriorScore(environment, action, options) {
  if (!options.policyModel || !options.policyWeight) return 0;
  const situation = analyzeSituation(environment, { side: options.side });
  const logit = scoreAlphaPolicyLogit(
    actionPolicyFeatures(action, {
      side: options.side,
      turn: situation.turn,
      phaseId: situation.phaseId,
      features: situation.features,
    }, environment.scenario),
    options.policyModel,
  );
  return clamp(logit, -8, 8) * options.policyWeight;
}

function backpropagate(path, value) {
  for (const node of path) {
    node.visits += 1;
    node.valueSum += value;
  }
}

function policyEntry(child, totalVisits = 0) {
  return {
    action: child.action,
    prior: Number(child.prior.toFixed(6)),
    visits: child.visits,
    visitShare: policyVisitShare(child, totalVisits),
    q: Number((child.visits ? child.valueSum / child.visits : child.seedValue).toFixed(6)),
    value: Number(child.seedValue.toFixed(6)),
  };
}

function rootPolicy(root) {
  const totalChildVisits = root.children.reduce((sum, child) => sum + Number(child.visits || 0), 0);
  return root.children
    .map((child) => policyEntry(child, totalChildVisits))
    .sort((a, b) => b.visits - a.visits || b.q - a.q || b.prior - a.prior || actionKey(a.action).localeCompare(actionKey(b.action)));
}

function appendProgressSnapshot(progress, root, options) {
  if (!Array.isArray(progress) || !(Number(options.progressLimit || 0) > 0)) return;
  const snapshot = searchProgressSnapshot(root, options);
  const lastIndex = progress.length - 1;
  if (lastIndex >= 0 && progress[lastIndex].iterations === snapshot.iterations) {
    progress[lastIndex] = snapshot;
    return;
  }
  if (progress.length >= options.progressLimit) {
    if (options.force) progress[lastIndex] = snapshot;
    return;
  }
  progress.push(snapshot);
}

function searchProgressSnapshot(root, options) {
  const policy = rootPolicy(root);
  return {
    schema: "zizi-el-alamein-alpha-progress-v1",
    iterations: Math.max(0, Math.floor(Number(options.iterations || 0))),
    elapsedMs: Math.max(0, Math.round(Number(options.elapsedMs || 0))),
    rootVisits: root.visits,
    rootChildren: root.children.length,
    bestAction: policy[0]?.action || null,
    topPolicy: policy.slice(0, 4),
    principalVariation: principalVariation(root, options.pvLimit),
    candidateLines: candidateLines(root, options.candidateLineLimit, options.pvLimit),
  };
}

function principalVariation(root, limit) {
  return variationFromNode(root, limit, false);
}

function candidateLines(root, limit, variationLimit) {
  const lineCount = Math.max(0, Math.floor(Number(limit || 0)));
  if (!lineCount || !root.children.length) return [];
  const totalVisits = root.children.reduce((sum, child) => sum + Number(child.visits || 0), 0);
  return root.children
    .slice()
    .sort((a, b) => b.visits - a.visits || b.seedValue - a.seedValue || b.prior - a.prior || actionKey(a.action).localeCompare(actionKey(b.action)))
    .slice(0, lineCount)
    .map((child) => ({
      ...policyEntry(child, totalVisits),
      principalVariation: variationFromNode(child, variationLimit, true),
    }));
}

function variationFromNode(start, limit, includeStart) {
  const variation = [];
  let node = start;
  if (includeStart && node.action && variation.length < limit) {
    variation.push(variationEntry(node));
  }
  while (node.children.length && variation.length < limit) {
    node = node.children
      .slice()
      .sort((a, b) => b.visits - a.visits || b.seedValue - a.seedValue || actionKey(a.action).localeCompare(actionKey(b.action)))[0];
    variation.push(variationEntry(node));
  }
  return variation;
}

function variationEntry(node) {
  return {
    action: node.action,
    visits: node.visits,
    q: Number((node.visits ? node.valueSum / node.visits : node.seedValue).toFixed(6)),
  };
}

function policyVisitShare(child, totalVisits) {
  if (Number(totalVisits || 0) > 0) {
    return Number((Number(child.visits || 0) / Number(totalVisits || 1)).toFixed(6));
  }
  return Number(child.prior.toFixed(6));
}

function isChanceNode(node) {
  return node.children.length > 0 && node.children.every((child) => isChanceAction(child.action));
}

function isChanceOnlyRoot(root) {
  return root.children.length > 0 && root.children.every((child) => isChanceAction(child.action));
}

function isChanceAction(action) {
  return action?.type === ENV_ACTION.RESOLVE_COMBAT && Number.isInteger(Number(action.dieRoll));
}

function normalizePriors(scores) {
  if (!scores.length) return [];
  const max = Math.max(...scores);
  const raw = scores.map((score) => Math.exp(Math.max(-8, Math.min(8, score - max))));
  const total = raw.reduce((sum, value) => sum + value, 0) || 1;
  return raw.map((value) => value / total);
}

function rootAdjustedPriors(priors, node, options) {
  if (node.depth !== 0 || !(Number(options.rootNoiseWeight || 0) > 0)) return priors;
  const noise = Array.isArray(options.rootNoise)
    ? options.rootNoise
    : generateAlphaRootNoise(priors.length, options.rootDirichletAlpha, options.rootNoiseRandom);
  return mixAlphaRootPriors(priors, noise, options.rootNoiseWeight);
}

function normalizeProbabilityVector(values) {
  const clean = Array.isArray(values)
    ? values.map((value) => Math.max(0, Number(value || 0)))
    : [];
  const total = clean.reduce((sum, value) => sum + value, 0);
  if (!clean.length) return [];
  if (!(total > 0)) return clean.map(() => 1 / clean.length);
  return clean.map((value) => value / total);
}

function padVector(values, length) {
  if (!Array.isArray(values) || !length) return [];
  return Array.from({ length }, (_, index) => values[index] ?? 0);
}

function sampleGamma(shape, random) {
  if (shape < 1) {
    const boost = sampleGamma(shape + 1, random);
    return boost * (clampOpenUnit(random()) ** (1 / shape));
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (let attempt = 0; attempt < 32; attempt += 1) {
    const x = sampleStandardNormal(random);
    const base = 1 + c * x;
    if (!(base > 0)) continue;
    const candidate = base ** 3;
    const accept = clampOpenUnit(random());
    if (accept < 1 - 0.0331 * (x ** 4)) return d * candidate;
    if (Math.log(accept) < 0.5 * x * x + d * (1 - candidate + Math.log(candidate))) {
      return d * candidate;
    }
  }
  return -Math.log(clampOpenUnit(random()));
}

function sampleStandardNormal(random) {
  const u1 = clampOpenUnit(random());
  const u2 = clampOpenUnit(random());
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clampOpenUnit(value) {
  return clamp(Number(value || 0), 1e-12, 0.999999999999);
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function quietApplyOptions() {
  return {
    enrichEvents: false,
    previousState: false,
    cloneResultState: false,
  };
}

function finiteProgress(currentDistance, nextDistance) {
  if (!Number.isFinite(currentDistance) || !Number.isFinite(nextDistance)) return 0;
  return currentDistance - nextDistance;
}

function outcomeValue(outcomeSide, side) {
  if (outcomeSide === side) return 1;
  if (outcomeSide === OPPOSITE_SIDE[side]) return -1;
  return 0;
}

function stableValue(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableValue).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableValue(value[key])}`).join(",")}}`;
}

function cloneJsonLike(value) {
  return value ? JSON.parse(JSON.stringify(value)) : value;
}

function now() {
  return Date.now();
}

function defaultPreApplyLimit(actionLimit) {
  return Math.min(96, Math.max(DEFAULT_PRE_APPLY_LIMIT, Number(actionLimit || DEFAULT_ACTION_LIMIT) * 3));
}
