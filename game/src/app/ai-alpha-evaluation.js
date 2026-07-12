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
} from "./ai-alpha-search.js";
import { makeSeededDieRoller } from "./ai-self-play.js";

const DEFAULT_MAX_PLIES = 160;
const DEFAULT_PROMOTION_THRESHOLD = 0.55;
const OPPOSITE_SIDE = Object.freeze({ axis: "allied", allied: "axis" });

export function runAlphaModelMatchGame({
  scenario,
  rules,
  board = null,
  initialState = null,
  candidateModel = null,
  baselineModel = null,
  candidateSide = "axis",
  seed = 1942,
  maxPlies = DEFAULT_MAX_PLIES,
  searchOptions = {},
  rollDie = null,
  suiteIndex = null,
  suiteLabel = null,
} = {}) {
  const dieRoll = rollDie || makeSeededDieRoller(seed);
  let environment = createEnvironment({ scenario, rules, board, state: initialState });
  const initialStateHash = stateHash(environment);
  const actions = [];
  const errors = [];
  let plies = 0;

  while (!environment.state.winner && plies < maxPlies) {
    const legalActions = generateLegalActions(environment, { includeChanceActions: true });
    if (!legalActions.length) {
      errors.push({ reason: "no_legal_actions", stateHash: stateHash(environment) });
      break;
    }

    const side = activeSide(environment);
    const role = side === candidateSide ? "candidate" : "baseline";
    let analysis = null;
    let action = null;
    let decision = null;
    if (legalActions.every(isChanceAction)) {
      action = chooseChanceAction(legalActions, dieRoll());
      decision = makeMatchDecisionEvidence(null, action, "chance");
    } else {
      analysis = analyzePosition(environment, {
        side,
        model: role === "candidate" ? candidateModel : baselineModel,
        ...searchOptionsForSide(searchOptions, side),
      });
      action = analysis.bestAction && !analysis.requiresChance
        ? analysis.bestAction
        : fallbackAction(legalActions);
      decision = makeMatchDecisionEvidence(
        analysis,
        action,
        action && analysis.bestAction && actionKey(action) === actionKey(analysis.bestAction)
          ? "search_best"
          : "fallback",
      );
    }

    if (!action) {
      errors.push({ reason: "no_action_selected", stateHash: stateHash(environment), side, role });
      break;
    }

    const stateHashBefore = stateHash(environment);
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
        stateHash: stateHashBefore,
        side,
        role,
      });
      break;
    }

    actions.push({
      ply: plies,
      side,
      role,
      stateHashBefore,
      action: applied.action,
      stateHashAfter: stateHash(applied.environment),
      rootValue: analysis?.rootValue ?? null,
      decision,
    });
    environment = applied.environment;
    plies += 1;
  }

  const winner = environment.state.winner || null;
  const candidateResult = matchResultForCandidate(winner, candidateSide, errors);
  return {
    schema: "zizi-el-alamein-alpha-match-game-v1",
    seed,
    suiteIndex,
    suiteLabel,
    candidateSide,
    baselineSide: OPPOSITE_SIDE[candidateSide] || null,
    initialStateHash,
    plies,
    maxPlies,
    guardHit: !winner && plies >= maxPlies,
    winner,
    candidateResult,
    finalStateHash: stateHash(environment),
    actions,
    errors,
  };
}

export function buildAlphaEvaluationSuite({
  games = 2,
  seed = 1942,
  candidateSide = "axis",
  alternateSides = true,
  seeds = null,
  candidateSides = null,
  maxPlies = null,
  labels = null,
  initialStates = null,
} = {}) {
  const seedList = cleanNumberList(seeds);
  const sideList = cleanSideList(candidateSides);
  const labelList = Array.isArray(labels) ? labels : [];
  const stateList = cleanInitialStateList(initialStates);
  const explicit = seedList.length > 0 || sideList.length > 0 || labelList.length > 0 || stateList.length > 0;
  const count = Math.max(
    1,
    Math.floor(Number(games || 0)),
    seedList.length,
    sideList.length,
    labelList.length,
    stateList.length,
  );
  const entries = [];
  for (let index = 0; index < count; index += 1) {
    const side = sideList.length
      ? sideList[index % sideList.length]
      : alternateSides && index % 2 === 1
        ? OPPOSITE_SIDE[candidateSide] || candidateSide
        : candidateSide;
    entries.push({
      index,
      label: labelList[index] || `suite-${index + 1}`,
      seed: seedList.length ? seedList[index % seedList.length] : Number(seed || 0) + index,
      candidateSide: normalizeSide(side, candidateSide),
      maxPlies: finiteOrNull(maxPlies),
      initialState: stateList.length ? cloneJsonLike(stateList[index % stateList.length]) : null,
    });
  }
  return {
    schema: "zizi-el-alamein-alpha-evaluation-suite-v1",
    seed,
    games: entries.length,
    candidateSide: normalizeSide(candidateSide, "axis"),
    alternateSides: Boolean(alternateSides),
    explicit,
    entries,
  };
}

export function runAlphaModelMatchBatch({
  games = 2,
  seed = 1942,
  candidateSide = "axis",
  alternateSides = true,
  promotionThreshold = DEFAULT_PROMOTION_THRESHOLD,
  minSideScore = null,
  minScoreLowerBound = null,
  suite = null,
  ...options
} = {}) {
  const suitePlan = normalizeAlphaEvaluationSuite(suite, {
    games,
    seed,
    candidateSide,
    alternateSides,
    maxPlies: options.maxPlies,
  });
  const results = [];
  for (const entry of suitePlan.entries) {
    results.push(runAlphaModelMatchGame({
      ...options,
      initialState: entry.initialState || options.initialState || null,
      candidateSide: entry.candidateSide,
      seed: entry.seed,
      maxPlies: entry.maxPlies ?? options.maxPlies,
      suiteIndex: entry.index,
      suiteLabel: entry.label,
    }));
  }
  return {
    schema: "zizi-el-alamein-alpha-match-batch-v1",
    generatedAt: new Date().toISOString(),
    seed: suitePlan.seed,
    games: suitePlan.entries.length,
    candidateSide,
    alternateSides: Boolean(alternateSides),
    promotionThreshold: Number(promotionThreshold || DEFAULT_PROMOTION_THRESHOLD),
    evaluationSuite: suitePlan,
    ...summarizeAlphaModelMatchResults(results, promotionThreshold, { minSideScore, minScoreLowerBound }),
    results,
  };
}

export function summarizeAlphaModelMatchResults(
  results = [],
  promotionThreshold = DEFAULT_PROMOTION_THRESHOLD,
  options = {},
) {
  const summary = summarizeResultGroup(results);
  const sideScores = summarizeAlphaModelSideScores(results);
  const requiredSideScore = optionalFiniteOrNull(options.minSideScore);
  const requiredScoreLowerBound = optionalFiniteOrNull(options.minScoreLowerBound);
  const sideScorePass = requiredSideScore === null
    ? true
    : sideScores.every((sideScore) => (
      sideScore.scoredGames > 0 && sideScore.candidateScore >= requiredSideScore
    ));
  const scoreLowerBoundPass = requiredScoreLowerBound === null
    ? true
    : Boolean(summary.arena?.scoreInterval95 && summary.arena.scoreInterval95.low >= requiredScoreLowerBound);
  return {
    ...summary,
    sideScores,
    decisionEvidence: summarizeAlphaMatchDecisionEvidence(results),
    minSideScore: requiredSideScore,
    sideScorePass,
    minScoreLowerBound: requiredScoreLowerBound,
    scoreLowerBoundPass,
    promote: summary.errors === 0
      && summary.scoredGames > 0
      && summary.candidateScore >= Number(promotionThreshold || DEFAULT_PROMOTION_THRESHOLD)
      && sideScorePass
      && scoreLowerBoundPass,
  };
}

export function summarizeAlphaMatchDecisionEvidence(results = []) {
  const actions = (results || []).flatMap((result) => result?.actions || []);
  const decisions = actions.map((action) => action.decision).filter((decision) => decision && typeof decision === "object");
  const analyzed = decisions.filter((decision) => Number(decision.searchIterations || 0) > 0);
  return {
    schema: "zizi-el-alamein-alpha-decision-evidence-v1",
    games: (results || []).length,
    actionCount: actions.length,
    decisionCount: decisions.length,
    analyzedActions: analyzed.length,
    candidateActions: actions.filter((action) => action.role === "candidate").length,
    baselineActions: actions.filter((action) => action.role === "baseline").length,
    chanceActions: decisions.filter((decision) => decision.selectionMode === "chance").length,
    fallbackActions: decisions.filter((decision) => decision.selectionMode === "fallback").length,
    averagePolicyEntropy: averageFinite(analyzed.map((decision) => decision.policyEntropy)) ?? 0,
    averageSearchIterations: averageFinite(analyzed.map((decision) => decision.searchIterations)) ?? 0,
    averageRootVisits: averageFinite(analyzed.map((decision) => decision.rootVisits)) ?? 0,
    selectedActionShare: averageFinite(analyzed.map((decision) => decision.selectedVisitShare)) ?? 0,
    actionTypes: countBy(actions, (action) => action.action?.type || "unknown"),
    selectionModes: countBy(decisions, (decision) => decision.selectionMode || "unknown"),
    roles: {
      candidate: summarizeRoleDecisionEvidence(actions, "candidate"),
      baseline: summarizeRoleDecisionEvidence(actions, "baseline"),
    },
  };
}

export function summarizeAlphaModelSideScores(results = []) {
  const groups = new Map();
  for (const result of results || []) {
    const side = normalizeSide(result?.candidateSide, null) || "unknown";
    if (!groups.has(side)) groups.set(side, []);
    groups.get(side).push(result);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => sideSortKey(left) - sideSortKey(right) || left.localeCompare(right))
    .map(([candidateSide, groupResults]) => ({
      candidateSide,
      games: groupResults.length,
      ...summarizeResultGroup(groupResults),
    }));
}

function searchOptionsForSide(searchOptions, side) {
  const base = { ...(searchOptions || {}) };
  delete base.axis;
  delete base.allied;
  return {
    ...base,
    ...(searchOptions?.[side] || {}),
  };
}

function normalizeAlphaEvaluationSuite(suite, fallback) {
  if (!suite) return buildAlphaEvaluationSuite(fallback);
  const entries = Array.isArray(suite) ? suite : suite.entries;
  if (!Array.isArray(entries) || !entries.length) return buildAlphaEvaluationSuite(fallback);
  const normalizedEntries = entries.map((entry, index) => ({
    index: Number.isInteger(Number(entry.index)) ? Number(entry.index) : index,
    label: entry.label || `suite-${index + 1}`,
    seed: Number.isFinite(Number(entry.seed)) ? Number(entry.seed) : Number(fallback.seed || 0) + index,
    candidateSide: normalizeSide(entry.candidateSide || entry.side, fallback.candidateSide),
    maxPlies: finiteOrNull(entry.maxPlies ?? fallback.maxPlies),
    initialState: normalizeSuiteInitialState(entry.initialState ?? entry.state),
    sourceStateHash: typeof entry.sourceStateHash === "string" ? entry.sourceStateHash : null,
    sourceSide: normalizeSide(entry.sourceSide, null),
    sourcePhaseId: typeof entry.sourcePhaseId === "string" ? entry.sourcePhaseId : null,
    sourceTurn: finiteOrNull(entry.sourceTurn),
    sourceOutcome: finiteOrNull(entry.sourceOutcome),
    sourceRootValue: finiteOrNull(entry.sourceRootValue),
    sourceOutcomeSource: typeof entry.sourceOutcomeSource === "string" ? entry.sourceOutcomeSource : null,
    sourceOutcomeWeight: finiteOrNull(entry.sourceOutcomeWeight),
    sourcePriority: finiteOrNull(entry.sourcePriority),
    sourcePolicyEntropy: finiteOrNull(entry.sourcePolicyEntropy),
    sourceSelectionMode: typeof entry.sourceSelectionMode === "string" ? entry.sourceSelectionMode : null,
    sourceTemperature: finiteOrNull(entry.sourceTemperature),
    sourceRootNoiseWeight: finiteOrNull(entry.sourceRootNoiseWeight),
    sourceSearchIterations: finiteOrNull(entry.sourceSearchIterations),
    sourceRootVisits: finiteOrNull(entry.sourceRootVisits),
  }));
  return {
    schema: "zizi-el-alamein-alpha-evaluation-suite-v1",
    seed: suite.seed ?? fallback.seed,
    games: normalizedEntries.length,
    candidateSide: normalizeSide(suite.candidateSide, fallback.candidateSide),
    alternateSides: Boolean(suite.alternateSides ?? fallback.alternateSides),
    explicit: Boolean(suite.explicit),
    challengeSelection: normalizeChallengeSelection(suite.challengeSelection),
    entries: normalizedEntries,
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

function matchResultForCandidate(winner, candidateSide, errors) {
  if (errors?.length) return "error";
  if (!winner?.side) return "draw";
  return winner.side === candidateSide ? "win" : "loss";
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

function makeMatchDecisionEvidence(analysis, action, selectionMode) {
  const selectedPolicy = selectedPolicyEntry(analysis, action);
  return {
    schema: "zizi-el-alamein-alpha-match-decision-v1",
    selectionMode,
    selectedAction: action || null,
    selectedVisitShare: finiteNumber(selectedPolicy?.visitShare, 0),
    selectedVisits: finiteNumber(selectedPolicy?.visits, 0),
    selectedQ: finiteOrNull(selectedPolicy?.q),
    selectedPrior: finiteOrNull(selectedPolicy?.prior),
    policyEntropy: policyEntropy(analysis?.policy || []),
    policySize: Array.isArray(analysis?.policy) ? analysis.policy.length : 0,
    searchIterations: finiteNumber(analysis?.search?.iterations, 0),
    rootVisits: finiteNumber(analysis?.search?.rootVisits, 0),
    rootChildren: finiteNumber(analysis?.search?.rootChildren, 0),
    rootValue: finiteOrNull(analysis?.rootValue),
    requiresChance: Boolean(analysis?.requiresChance),
  };
}

function selectedPolicyEntry(analysis, action) {
  if (!analysis || !action || !Array.isArray(analysis.policy)) return null;
  const key = actionKey(action);
  return analysis.policy.find((entry) => actionKey(entry.action) === key) || null;
}

function summarizeResultGroup(results = []) {
  const candidateWins = results.filter((result) => result.candidateResult === "win").length;
  const baselineWins = results.filter((result) => result.candidateResult === "loss").length;
  const draws = results.filter((result) => result.candidateResult === "draw").length;
  const errors = results.filter((result) => result.candidateResult === "error").length;
  const scoredGames = Math.max(0, results.length - errors);
  const score = scoredGames ? round((candidateWins + draws * 0.5) / scoredGames, 4) : 0;
  return {
    candidateWins,
    baselineWins,
    draws,
    errors,
    scoredGames,
    candidateScore: score,
    arena: summarizeArenaMetrics({
      candidateWins,
      baselineWins,
      draws,
      errors,
      scoredGames,
      candidateScore: score,
    }),
  };
}

function summarizeRoleDecisionEvidence(actions, role) {
  const roleActions = (actions || []).filter((action) => action.role === role);
  const decisions = roleActions.map((action) => action.decision).filter((decision) => decision && typeof decision === "object");
  const analyzed = decisions.filter((decision) => Number(decision.searchIterations || 0) > 0);
  return {
    actionCount: roleActions.length,
    decisionCount: decisions.length,
    analyzedActions: analyzed.length,
    averagePolicyEntropy: averageFinite(analyzed.map((decision) => decision.policyEntropy)) ?? 0,
    averageSearchIterations: averageFinite(analyzed.map((decision) => decision.searchIterations)) ?? 0,
    averageRootVisits: averageFinite(analyzed.map((decision) => decision.rootVisits)) ?? 0,
    selectedActionShare: averageFinite(analyzed.map((decision) => decision.selectedVisitShare)) ?? 0,
    actionTypes: countBy(roleActions, (action) => action.action?.type || "unknown"),
  };
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

function averageFinite(values) {
  const clean = (values || []).map(Number).filter(Number.isFinite);
  if (!clean.length) return null;
  return round(clean.reduce((sum, value) => sum + value, 0) / clean.length, 6);
}

function countBy(items, selector) {
  return (items || []).reduce((counts, item) => {
    const key = selector(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function summarizeArenaMetrics({
  candidateWins = 0,
  baselineWins = 0,
  draws = 0,
  errors = 0,
  scoredGames = 0,
  candidateScore = 0,
} = {}) {
  const games = candidateWins + baselineWins + draws + errors;
  const score = Number(candidateScore || 0);
  const interval = scoredGames > 0 ? wilsonScoreInterval(score, scoredGames) : null;
  return {
    schema: "zizi-el-alamein-alpha-arena-metrics-v1",
    games,
    scoredGames,
    errors,
    score,
    scoreInterval95: interval,
    eloDiff: scoredGames > 0 ? eloDiff(score) : null,
    eloDiffInterval95: interval ? {
      low: eloDiff(interval.low),
      high: eloDiff(interval.high),
    } : null,
    decisiveRate: scoredGames > 0 ? round((candidateWins + baselineWins) / scoredGames, 4) : 0,
    drawRate: scoredGames > 0 ? round(draws / scoredGames, 4) : 0,
  };
}

function wilsonScoreInterval(score, games, z = 1.96) {
  const n = Number(games || 0);
  if (!(n > 0)) return null;
  const p = clamp(Number(score || 0), 0, 1);
  const z2 = z * z;
  const denominator = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denominator;
  const margin = (z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denominator;
  return {
    low: round(clamp(center - margin, 0, 1), 4),
    high: round(clamp(center + margin, 0, 1), 4),
  };
}

function eloDiff(score) {
  const p = clamp(Number(score || 0), 0.001, 0.999);
  return Math.round(400 * Math.log10(p / (1 - p)));
}

function sideSortKey(side) {
  if (side === "axis") return 0;
  if (side === "allied") return 1;
  return 2;
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  const next = Number(value);
  if (!Number.isFinite(next)) return min;
  return Math.min(max, Math.max(min, next));
}

function cleanNumberList(values) {
  if (!Array.isArray(values)) return [];
  return values.map((value) => Number(value)).filter(Number.isFinite);
}

function cleanSideList(values) {
  if (!Array.isArray(values)) return [];
  return values.map((value) => normalizeSide(value, null)).filter(Boolean);
}

function cleanInitialStateList(values) {
  if (!Array.isArray(values)) return [];
  return values.map(normalizeSuiteInitialState).filter(Boolean);
}

function normalizeSuiteInitialState(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return cloneJsonLike(value);
}

function normalizeSide(side, fallback = "axis") {
  if (side === "axis" || side === "allied") return side;
  return fallback === "axis" || fallback === "allied" ? fallback : null;
}

function finiteOrNull(value) {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

function finiteNumber(value, fallback) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function optionalFiniteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  return finiteOrNull(value);
}

function cloneJsonLike(value) {
  return value ? JSON.parse(JSON.stringify(value)) : value;
}
