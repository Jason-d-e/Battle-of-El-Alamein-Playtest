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
  matchingAlphaPhasePlanHint,
  nextAlphaReusablePhasePlan,
} from "./ai-alpha-search.js";
import { createElAlameinAlphaEnvironmentAdapter } from "./ai-alpha-environment-adapter.js";
import {
  ALPHA_RUNTIME_ENGINE,
  analyzeGenericAlphaPosition,
  genericAlphaSearchOptionsFromModel,
} from "./ai-alpha-runtime.js";
import { makeSeededDieRoller } from "./ai-self-play.js";
import { evaluateSituation } from "./ai-situation.js";

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
  guardAdjudication = "situation",
  guardAdjudicationMargin = 0.05,
  requireRulesWinner = true,
} = {}) {
  const dieRoll = rollDie || makeSeededDieRoller(seed);
  let environment = createEnvironment({ scenario, rules, board, state: initialState });
  const searchEngine = normalizeEvaluationSearchEngine(searchOptions.engine);
  const genericAdapter = searchEngine === ALPHA_RUNTIME_ENGINE.GENERIC
    ? createElAlameinAlphaEnvironmentAdapter({ scenario, rules, board: environment.board })
    : null;
  const initialStateHash = stateHash(environment);
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

    const chanceOnly = legalActions.every(isChanceAction);
    const side = genericAdapter && !chanceOnly
      ? genericAdapter.currentPlayer(environment.state)
      : activeSide(environment);
    const role = side === candidateSide ? "candidate" : "baseline";
    let analysis = null;
    let action = null;
    let decision = null;
    if (chanceOnly) {
      action = chooseChanceAction(legalActions, dieRoll());
      decision = makeMatchDecisionEvidence(null, action, "chance");
    } else {
      const sideSearchOptions = searchOptionsForSide(searchOptions, side);
      const selectedModel = role === "candidate" ? candidateModel : baselineModel;
      if (searchEngine === ALPHA_RUNTIME_ENGINE.GENERIC) {
        analysis = analyzeGenericAlphaPosition(environment, selectedModel, sideSearchOptions);
      } else {
        const phasePlanHint = matchingAlphaPhasePlanHint(reusablePhasePlan, environment, side, legalActions);
        analysis = analyzePosition(environment, {
          side,
          model: selectedModel,
          ...sideSearchOptions,
          ...(phasePlanHint ? { phasePlanHint } : {}),
        });
      }
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
      reusablePhasePlan = searchEngine === ALPHA_RUNTIME_ENGINE.GENERIC
        ? null
        : nextAlphaReusablePhasePlan(analysis, action, environment, side);
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
  const guardHit = !winner && plies >= maxPlies;
  const adjudication = guardHit
    ? adjudicateGuardPosition(environment, {
      mode: guardAdjudication,
      margin: guardAdjudicationMargin,
    })
    : null;
  const scoredWinner = winner || (requireRulesWinner ? null : adjudication?.winner) || null;
  if (guardHit && !scoredWinner) {
    errors.push({
      reason: requireRulesWinner ? "rules_winner_required" : "guard_position_unadjudicated",
      stateHash: stateHash(environment),
      mode: adjudication?.mode || normalizeGuardAdjudicationMode(guardAdjudication),
    });
  }
  const candidateResult = matchResultForCandidate(scoredWinner, candidateSide, errors);
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
    guardHit,
    guardAdjudication: normalizeGuardAdjudicationMode(guardAdjudication),
    guardAdjudicationMargin: normalizedAdjudicationMargin(guardAdjudicationMargin),
    requireRulesWinner: Boolean(requireRulesWinner),
    searchEngine,
    adjudication,
    winner,
    candidateResult,
    finalStateHash: stateHash(environment),
    actions,
    errors,
  };
}

export function buildAlphaPairedEvaluationSuite({
  seed = 1942,
  seeds = null,
  maxPlies = null,
  labels = null,
  initialStates = null,
} = {}) {
  const stateList = cleanInitialStateList(initialStates);
  if (!stateList.length) throw new Error("Paired Alpha evaluation suite requires initial states");
  const seedList = cleanNumberList(seeds);
  const labelList = Array.isArray(labels) ? labels : [];
  const entries = [];
  for (let positionIndex = 0; positionIndex < stateList.length; positionIndex += 1) {
    const positionSeed = seedList.length ? seedList[positionIndex % seedList.length] : Number(seed || 0) + positionIndex;
    const label = labelList[positionIndex] || `paired-${positionIndex + 1}`;
    for (const candidateSide of ["axis", "allied"]) {
      entries.push({
        index: entries.length,
        label: `${label}-${candidateSide}`,
        seed: positionSeed,
        candidateSide,
        maxPlies: finiteOrNull(maxPlies),
        initialState: cloneJsonLike(stateList[positionIndex]),
      });
    }
  }
  return {
    schema: "zizi-el-alamein-alpha-evaluation-suite-v1",
    seed,
    games: entries.length,
    candidateSide: "axis",
    alternateSides: true,
    explicit: true,
    paired: true,
    entries,
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
  sideScoreBaselines = null,
  minSideScoreDelta = null,
  sideScoreDeltaThresholds = null,
  minScoreLowerBound = null,
  minEloLowerBound = null,
  maxArenaDrawRate = null,
  minArenaDecisiveRate = null,
  suite = null,
  overrideSuiteMaxPlies = false,
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
      maxPlies: overrideSuiteMaxPlies ? options.maxPlies : entry.maxPlies ?? options.maxPlies,
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
    overrideSuiteMaxPlies: Boolean(overrideSuiteMaxPlies),
    promotionThreshold: Number(promotionThreshold || DEFAULT_PROMOTION_THRESHOLD),
    searchConfiguration: alphaEvaluationSearchConfiguration(options.searchOptions),
    evaluationSuite: suitePlan,
    ...summarizeAlphaModelMatchResults(results, promotionThreshold, {
      minSideScore,
      sideScoreBaselines,
      minSideScoreDelta,
      sideScoreDeltaThresholds,
      minScoreLowerBound,
      minEloLowerBound,
      maxArenaDrawRate,
      minArenaDecisiveRate,
    }),
    results,
  };
}

export function alphaEvaluationSearchConfiguration(searchOptions = {}) {
  const engine = normalizeEvaluationSearchEngine(searchOptions.engine);
  if (engine === ALPHA_RUNTIME_ENGINE.GENERIC) {
    const normalized = genericAlphaSearchOptionsFromModel(null, searchOptions);
    return {
      engine,
      simulations: normalized.simulations,
      maxDepth: normalized.maxDepth,
      actionLimit: normalized.actionLimit,
      exploration: normalized.exploration,
      policyTemperature: normalized.policyTemperature,
      phasePlanWeight: 0,
    };
  }
  return {
    engine,
    iterations: Math.max(1, Number(searchOptions.iterations || 8)),
    maxDepth: Math.max(1, Number(searchOptions.maxDepth || 1)),
    actionLimit: Number(searchOptions.actionLimit || 8),
    preApplyLimit: Number(searchOptions.preApplyLimit || 24),
    phasePlanWeight: Math.max(0, Number(searchOptions.phasePlanWeight || 0)),
  };
}

export function runAlphaReferenceArena({
  scenario,
  rules,
  board = null,
  initialState = null,
  candidateModel = null,
  references = [],
  seed = 1942,
  games = 2,
  maxPlies = DEFAULT_MAX_PLIES,
  candidateSide = "axis",
  alternateSides = true,
  suite = null,
  searchOptions = {},
  promotionThreshold = DEFAULT_PROMOTION_THRESHOLD,
  guardAdjudication = "situation",
  guardAdjudicationMargin = 0.05,
  requireRulesWinner = true,
} = {}) {
  const normalizedReferences = normalizeReferenceModels(references);
  const entries = normalizedReferences.map((reference, index) => {
    const batch = runAlphaModelMatchBatch({
      scenario,
      rules,
      board,
      initialState,
      candidateModel,
      baselineModel: reference.model,
      seed: reference.seed === null ? Number(seed || 0) + index * 1000 : reference.seed,
      games: reference.games === null ? games : reference.games,
      maxPlies: reference.maxPlies === null ? maxPlies : reference.maxPlies,
      candidateSide: normalizeSide(reference.candidateSide, candidateSide),
      alternateSides: reference.alternateSides ?? alternateSides,
      promotionThreshold: reference.promotionThreshold === null ? promotionThreshold : reference.promotionThreshold,
      suite: reference.suite || suite,
      searchOptions: reference.searchOptions || searchOptions,
      guardAdjudication,
      guardAdjudicationMargin,
      requireRulesWinner,
    });
    return {
      schema: "zizi-el-alamein-alpha-reference-arena-entry-v1",
      index,
      label: reference.label,
      source: reference.source,
      modelPresent: Boolean(reference.model),
      evaluation: batch,
    };
  });
  return {
    schema: "zizi-el-alamein-alpha-reference-arena-v1",
    generatedAt: new Date().toISOString(),
    seed,
    references: entries.length,
    ...summarizeAlphaReferenceArena(entries),
    entries,
  };
}

export function summarizeAlphaReferenceArena(entries = []) {
  const scored = (entries || [])
    .map((entry) => ({
      label: entry?.label || "reference",
      candidateScore: finiteOrNull(entry?.evaluation?.candidateScore),
      scoredGames: finiteNumber(entry?.evaluation?.scoredGames, 0),
      errors: finiteNumber(entry?.evaluation?.errors, 0),
    }))
    .filter((entry) => entry.candidateScore !== null && entry.scoredGames > 0);
  const errors = (entries || []).reduce((sum, entry) => sum + finiteNumber(entry?.evaluation?.errors, 0), 0);
  return {
    scoredReferences: scored.length,
    minCandidateScore: scored.length ? round(Math.min(...scored.map((entry) => entry.candidateScore)), 4) : null,
    averageCandidateScore: scored.length
      ? round(scored.reduce((sum, entry) => sum + entry.candidateScore, 0) / scored.length, 4)
      : null,
    errors,
    scores: scored,
  };
}

export function summarizeAlphaModelMatchResults(
  results = [],
  promotionThreshold = DEFAULT_PROMOTION_THRESHOLD,
  options = {},
) {
  const summary = summarizeResultGroup(results);
  const normalizedSideBaselines = normalizeSideScoreBaselines(options.sideScoreBaselines);
  const sideScores = summarizeAlphaModelSideScores(results).map((sideScore) => {
    const baselineScore = optionalFiniteOrNull(normalizedSideBaselines?.[sideScore.candidateSide]);
    return {
      ...sideScore,
      baselineScore,
      scoreDelta: baselineScore === null ? null : round(sideScore.candidateScore - baselineScore, 4),
    };
  });
  const requiredSideScore = optionalFiniteOrNull(options.minSideScore);
  const requiredSideScoreDelta = optionalFiniteOrNull(options.minSideScoreDelta);
  const explicitSideScoreDeltas = normalizeSideScoreDeltaThresholds(options.sideScoreDeltaThresholds, null);
  const requiredSideScoreDeltas = Object.keys(explicitSideScoreDeltas).length
    ? explicitSideScoreDeltas
    : requiredSideScoreDelta === null
      ? {}
      : Object.fromEntries(sideScores.map((sideScore) => [sideScore.candidateSide, requiredSideScoreDelta]));
  const requiredScoreLowerBound = optionalFiniteOrNull(options.minScoreLowerBound);
  const requiredEloLowerBound = optionalFiniteOrNull(options.minEloLowerBound);
  const maxArenaDrawRate = optionalFiniteOrNull(options.maxArenaDrawRate);
  const minArenaDecisiveRate = optionalFiniteOrNull(options.minArenaDecisiveRate);
  const sideScorePass = requiredSideScore === null
    ? true
    : sideScores.every((sideScore) => (
      sideScore.scoredGames > 0 && sideScore.candidateScore >= requiredSideScore
    ));
  const sideScoreDeltaPass = !Object.keys(requiredSideScoreDeltas).length
    ? true
    : Object.entries(requiredSideScoreDeltas).every(([side, threshold]) => {
      const sideScore = sideScores.find((entry) => entry.candidateSide === side);
      return Boolean(
        sideScore
        && sideScore.scoredGames > 0
        && sideScore.scoreDelta !== null
        && sideScore.scoreDelta >= threshold,
      );
    });
  const scoreLowerBoundPass = requiredScoreLowerBound === null
    ? true
    : Boolean(summary.arena?.scoreInterval95 && summary.arena.scoreInterval95.low >= requiredScoreLowerBound);
  const eloLowerBoundPass = requiredEloLowerBound === null
    ? true
    : Boolean(summary.arena?.eloDiffInterval95 && summary.arena.eloDiffInterval95.low >= requiredEloLowerBound);
  const drawRatePass = maxArenaDrawRate === null
    ? true
    : Boolean(summary.scoredGames > 0 && summary.arena?.unresolvedRate <= maxArenaDrawRate);
  const decisiveRatePass = minArenaDecisiveRate === null
    ? true
    : Boolean(summary.scoredGames > 0 && summary.arena?.decisiveRate >= minArenaDecisiveRate);
  const promotionVerdict = summarizeAlphaPromotionVerdict({
    ...summary,
    sideScores,
    promotionThreshold: Number(promotionThreshold || DEFAULT_PROMOTION_THRESHOLD),
    minSideScore: requiredSideScore,
    sideScorePass,
    minSideScoreDelta: requiredSideScoreDelta,
    sideScoreDeltaThresholds: requiredSideScoreDeltas,
    sideScoreDeltaPass,
    minScoreLowerBound: requiredScoreLowerBound,
    scoreLowerBoundPass,
    minEloLowerBound: requiredEloLowerBound,
    eloLowerBoundPass,
    maxArenaDrawRate,
    drawRatePass,
    minArenaDecisiveRate,
    decisiveRatePass,
  });
  return {
    ...summary,
    rulesWinnerGames: results.filter((result) => Boolean(result?.winner?.side)).length,
    sideScores,
    decisionEvidence: summarizeAlphaMatchDecisionEvidence(results),
    minSideScore: requiredSideScore,
    sideScorePass,
    sideScoreBaselines: normalizedSideBaselines,
    minSideScoreDelta: requiredSideScoreDelta,
    sideScoreDeltaThresholds: requiredSideScoreDeltas,
    sideScoreDeltaPass,
    minScoreLowerBound: requiredScoreLowerBound,
    scoreLowerBoundPass,
    minEloLowerBound: requiredEloLowerBound,
    eloLowerBoundPass,
    maxArenaDrawRate,
    drawRatePass,
    minArenaDecisiveRate,
    decisiveRatePass,
    promotionVerdict,
    promote: promotionVerdict.ok,
  };
}

export function summarizeAlphaPromotionVerdict(evaluation = {}) {
  const promotionThreshold = Number(evaluation.promotionThreshold || DEFAULT_PROMOTION_THRESHOLD);
  const minSideScore = optionalFiniteOrNull(evaluation.minSideScore);
  const minSideScoreDelta = optionalFiniteOrNull(evaluation.minSideScoreDelta);
  const sideScoreDeltaThresholds = normalizeSideScoreDeltaThresholds(
    evaluation.sideScoreDeltaThresholds,
    minSideScoreDelta,
  );
  const sideScoreDeltaRequired = Object.keys(sideScoreDeltaThresholds).length > 0;
  const minScoreLowerBound = optionalFiniteOrNull(evaluation.minScoreLowerBound);
  const minEloLowerBound = optionalFiniteOrNull(evaluation.minEloLowerBound);
  const maxArenaDrawRate = optionalFiniteOrNull(evaluation.maxArenaDrawRate);
  const minArenaDecisiveRate = optionalFiniteOrNull(evaluation.minArenaDecisiveRate);
  const gates = [
    promotionGate("errors", Number(evaluation.errors || 0) <= 0, {
      actual: Number(evaluation.errors || 0),
      threshold: 0,
      comparison: "max",
      reason: "evaluation_errors_present",
    }),
    promotionGate("scoredGames", Number(evaluation.scoredGames || 0) > 0, {
      actual: Number(evaluation.scoredGames || 0),
      threshold: 1,
      comparison: "min",
      reason: "evaluation_games_too_few",
    }),
    promotionGate("candidateScore", Number(evaluation.candidateScore || 0) >= promotionThreshold, {
      actual: finiteOrNull(evaluation.candidateScore),
      threshold: promotionThreshold,
      comparison: "min",
      reason: "candidate_score_below_threshold",
    }),
    promotionGate("sideScore", minSideScore === null || evaluation.sideScorePass !== false, {
      actual: minSideScore === null ? null : lowestSideScore(evaluation.sideScores),
      threshold: minSideScore,
      comparison: "min",
      required: minSideScore !== null,
      reason: "candidate_side_score_below_threshold",
    }),
    promotionGate("sideScoreDelta", !sideScoreDeltaRequired || evaluation.sideScoreDeltaPass !== false, {
      actual: sideScoreDeltaRequired ? sideScoreDeltaMap(evaluation.sideScores) : null,
      threshold: sideScoreDeltaRequired ? sideScoreDeltaThresholds : null,
      comparison: "per_side_min",
      required: sideScoreDeltaRequired,
      reason: "candidate_side_score_delta_below_threshold",
    }),
    promotionGate("scoreLowerBound", minScoreLowerBound === null || evaluation.scoreLowerBoundPass !== false, {
      actual: evaluation.arena?.scoreInterval95?.low ?? null,
      threshold: minScoreLowerBound,
      comparison: "min",
      required: minScoreLowerBound !== null,
      reason: "candidate_score_lower_bound_below_threshold",
    }),
    promotionGate("eloLowerBound", minEloLowerBound === null || evaluation.eloLowerBoundPass !== false, {
      actual: evaluation.arena?.eloDiffInterval95?.low ?? null,
      threshold: minEloLowerBound,
      comparison: "min",
      required: minEloLowerBound !== null,
      reason: "candidate_elo_lower_bound_below_threshold",
    }),
    promotionGate("unresolvedRate", maxArenaDrawRate === null || evaluation.drawRatePass !== false, {
      actual: evaluation.arena?.unresolvedRate ?? evaluation.arena?.drawRate ?? null,
      threshold: maxArenaDrawRate,
      comparison: "max",
      required: maxArenaDrawRate !== null,
      reason: "arena_draw_rate_too_high",
    }),
    promotionGate("decisiveRate", minArenaDecisiveRate === null || evaluation.decisiveRatePass !== false, {
      actual: evaluation.arena?.decisiveRate ?? null,
      threshold: minArenaDecisiveRate,
      comparison: "min",
      required: minArenaDecisiveRate !== null,
      reason: "arena_decisive_rate_too_low",
    }),
  ];
  const failed = gates.find((gate) => !gate.ok);
  return {
    schema: "zizi-el-alamein-alpha-promotion-verdict-v1",
    ok: !failed,
    reason: failed?.reason || null,
    gates,
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
    averageRecommendationConfidence: averageFinite(analyzed.map((decision) => decision.recommendationConfidence)) ?? null,
    averageRecommendationVisitMargin: averageFinite(analyzed.map((decision) => decision.recommendationVisitMargin)) ?? null,
    averageRecommendationQMargin: averageFinite(analyzed.map((decision) => decision.recommendationQMargin)) ?? null,
    averageRecommendationUncertainty: averageFinite(analyzed.map((decision) => decision.recommendationUncertainty)) ?? null,
    averageSearchIterations: averageFinite(analyzed.map((decision) => decision.searchIterations)) ?? 0,
    averageRootVisits: averageFinite(analyzed.map((decision) => decision.rootVisits)) ?? 0,
    phasePlanActions: analyzed.filter((decision) => Number(decision.phasePlanActions || 0) > 0).length,
    phasePlanReusedActions: analyzed.filter((decision) => decision.phasePlanReused === true).length,
    phasePlanSearchActions: analyzed.filter((decision) => (
      Number(decision.phasePlanActions || 0) > 0 && decision.phasePlanReused !== true
    )).length,
    selectedActionShare: averageFinite(analyzed.map((decision) => decision.selectedVisitShare)) ?? 0,
    actionTypes: countBy(actions, (action) => action.action?.type || "unknown"),
    selectionModes: countBy(decisions, (decision) => decision.selectionMode || "unknown"),
    recommendationLabels: countBy(analyzed, (decision) => decision.recommendationLabel || "unknown"),
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

function promotionGate(key, ok, {
  actual = null,
  threshold = null,
  comparison = "min",
  required = true,
  reason = null,
} = {}) {
  return {
    key,
    ok: Boolean(ok),
    required: Boolean(required),
    actual: finiteOrNull(actual),
    threshold: finiteOrNull(threshold),
    comparison,
    reason: ok ? null : reason,
  };
}

function lowestSideScore(sideScores = []) {
  const scores = (sideScores || [])
    .map((score) => finiteOrNull(score?.candidateScore))
    .filter((score) => score !== null);
  return scores.length ? round(Math.min(...scores), 4) : null;
}

function lowestSideScoreDelta(sideScores = []) {
  const scores = (sideScores || [])
    .map((score) => optionalFiniteOrNull(score?.scoreDelta))
    .filter((score) => score !== null);
  return scores.length ? round(Math.min(...scores), 4) : null;
}

function sideScoreDeltaMap(sideScores = []) {
  return Object.fromEntries((sideScores || [])
    .filter((entry) => entry?.candidateSide && Number.isFinite(Number(entry?.scoreDelta)))
    .map((entry) => [entry.candidateSide, Number(entry.scoreDelta)]));
}

function normalizeSideScoreDeltaThresholds(value, fallback = null) {
  const thresholds = {};
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const side of ["axis", "allied"]) {
      const threshold = optionalFiniteOrNull(value[side]);
      if (threshold !== null) thresholds[side] = threshold;
    }
  }
  if (!Object.keys(thresholds).length && fallback !== null) {
    thresholds.axis = fallback;
    thresholds.allied = fallback;
  }
  return thresholds;
}

function normalizeSideScoreBaselines(value) {
  const source = Array.isArray(value)
    ? Object.fromEntries(value.map((entry) => [entry?.candidateSide, entry?.candidateScore]))
    : value?.sideScores
      ? Object.fromEntries(value.sideScores.map((entry) => [entry?.candidateSide, entry?.candidateScore]))
      : value;
  if (!source || typeof source !== "object" || Array.isArray(source)) return null;
  const output = {};
  for (const side of ["axis", "allied"]) {
    const score = optionalFiniteOrNull(source[side]);
    if (score !== null) output[side] = score;
  }
  return Object.keys(output).length ? output : null;
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

function normalizeEvaluationSearchEngine(value) {
  return String(value || "").trim().toLowerCase() === ALPHA_RUNTIME_ENGINE.GENERIC
    ? ALPHA_RUNTIME_ENGINE.GENERIC
    : ALPHA_RUNTIME_ENGINE.LEGACY;
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
    sourceRecommendationConfidence: finiteOrNull(entry.sourceRecommendationConfidence),
    sourceRecommendationVisitMargin: finiteOrNull(entry.sourceRecommendationVisitMargin),
    sourceRecommendationQMargin: finiteOrNull(entry.sourceRecommendationQMargin),
    sourceRecommendationLabel: typeof entry.sourceRecommendationLabel === "string" ? entry.sourceRecommendationLabel : null,
    sourceSelectionMode: typeof entry.sourceSelectionMode === "string" ? entry.sourceSelectionMode : null,
    sourceTemperature: finiteOrNull(entry.sourceTemperature),
    sourceRootNoiseWeight: finiteOrNull(entry.sourceRootNoiseWeight),
    sourceSearchIterations: finiteOrNull(entry.sourceSearchIterations),
    sourceRootVisits: finiteOrNull(entry.sourceRootVisits),
    sourceAlphaRecommendation: normalizeSourceAlphaRecommendation(entry.sourceAlphaRecommendation),
  }));
  return {
    schema: "zizi-el-alamein-alpha-evaluation-suite-v1",
    seed: suite.seed ?? fallback.seed,
    games: normalizedEntries.length,
    candidateSide: normalizeSide(suite.candidateSide, fallback.candidateSide),
    alternateSides: Boolean(suite.alternateSides ?? fallback.alternateSides),
    explicit: Boolean(suite.explicit),
    paired: Boolean(suite.paired),
    challengeSelection: normalizeChallengeSelection(suite.challengeSelection),
    entries: normalizedEntries,
  };
}

function adjudicateGuardPosition(environment, options = {}) {
  const mode = normalizeGuardAdjudicationMode(options.mode);
  const margin = normalizedAdjudicationMargin(options.margin);
  if (mode !== "situation") {
    return {
      schema: "zizi-el-alamein-alpha-guard-adjudication-v1",
      mode,
      margin,
      axisValue: null,
      alliedValue: null,
      zeroSumValue: null,
      winner: null,
    };
  }
  const axisValue = evaluateSituation(environment, { side: "axis" });
  const alliedValue = evaluateSituation(environment, { side: "allied" });
  const zeroSumValue = round((axisValue - alliedValue) / 2, 6);
  // The scenario has no draw result: if neither side has a positive edge,
  // Allied receives the close-call adjudication, matching the turn-limit rule.
  const winningSide = zeroSumValue > 0 ? "axis" : "allied";
  return {
    schema: "zizi-el-alamein-alpha-guard-adjudication-v1",
    mode,
    margin,
    axisValue: round(axisValue, 6),
    alliedValue: round(alliedValue, 6),
    zeroSumValue,
    withinMargin: Math.abs(zeroSumValue) <= margin,
    winner: {
      side: winningSide,
      reason: "guard-adjudication",
      type: "situation",
      turn: Number(environment.state?.turn || 1),
    },
  };
}

function normalizeGuardAdjudicationMode(value) {
  return value === "unscored" || value === "draw" ? "unscored" : "situation";
}

function normalizedAdjudicationMargin(value) {
  const next = Number(value);
  return Number.isFinite(next) ? Math.min(1, Math.max(0, next)) : 0.05;
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

function normalizeSourceAlphaRecommendation(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    ok: Boolean(value.ok),
    reason: typeof value.reason === "string" ? value.reason : null,
    selectedSource: typeof value.selectedSource === "string" ? value.selectedSource : null,
    illegalCandidateCount: finiteOrNull(value.illegalCandidateCount),
    legalActionCount: finiteOrNull(value.legalActionCount),
    runtimeActionMode: typeof value.runtimeActionMode === "string" ? value.runtimeActionMode : null,
    runtimeAnalysisMode: typeof value.runtimeAnalysisMode === "string" ? value.runtimeAnalysisMode : null,
  };
}

function normalizeReferenceModels(references) {
  if (!Array.isArray(references)) return [];
  return references.map((reference, index) => {
    const item = reference && typeof reference === "object" && !Array.isArray(reference)
      ? reference
      : { model: reference };
    return {
      label: typeof item.label === "string" && item.label ? item.label : `reference-${index + 1}`,
      source: typeof item.source === "string" ? item.source : null,
      model: item.model || null,
      seed: finiteOrNull(item.seed),
      games: finiteOrNull(item.games),
      maxPlies: finiteOrNull(item.maxPlies),
      candidateSide: normalizeSide(item.candidateSide, null),
      alternateSides: typeof item.alternateSides === "boolean" ? item.alternateSides : null,
      promotionThreshold: finiteOrNull(item.promotionThreshold),
      suite: item.suite && typeof item.suite === "object" ? item.suite : null,
      searchOptions: item.searchOptions && typeof item.searchOptions === "object" ? item.searchOptions : null,
    };
  });
}

function matchResultForCandidate(winner, candidateSide, errors) {
  if (errors?.length) return "error";
  if (!winner?.side) return "error";
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
    recommendationConfidence: finiteOrNull(analysis?.recommendation?.confidence),
    recommendationVisitMargin: finiteOrNull(analysis?.recommendation?.visitMargin),
    recommendationQMargin: finiteOrNull(analysis?.recommendation?.qMargin),
    recommendationUncertainty: recommendationUncertainty(analysis?.recommendation, analysis?.policy),
    recommendationLabel: typeof analysis?.recommendation?.label === "string" ? analysis.recommendation.label : null,
    policySize: Array.isArray(analysis?.policy) ? analysis.policy.length : 0,
    searchIterations: finiteNumber(analysis?.search?.iterations, 0),
    rootVisits: finiteNumber(analysis?.search?.rootVisits, 0),
    rootChildren: finiteNumber(analysis?.search?.rootChildren, 0),
    phasePlanActions: finiteNumber(analysis?.search?.phasePlan?.actions, 0),
    phasePlanReused: Boolean(analysis?.search?.phasePlan?.reused),
    phasePlanSearchedNodes: finiteNumber(analysis?.search?.phasePlan?.searchedNodes, 0),
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
  const unresolved = results.filter((result) => (
    result.candidateResult !== "win"
    && result.candidateResult !== "loss"
    && result.candidateResult !== "error"
  )).length;
  const errors = results.filter((result) => result.candidateResult === "error").length + unresolved;
  const scoredGames = candidateWins + baselineWins;
  const score = scoredGames ? round(candidateWins / scoredGames, 4) : 0;
  return {
    candidateWins,
    baselineWins,
    unresolved,
    errors,
    scoredGames,
    adjudicatedGames: results.filter((result) => Boolean(result?.adjudication?.winner)).length,
    candidateScore: score,
    arena: summarizeArenaMetrics({
      candidateWins,
      baselineWins,
      unresolved,
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
    averageRecommendationConfidence: averageFinite(analyzed.map((decision) => decision.recommendationConfidence)) ?? null,
    averageRecommendationVisitMargin: averageFinite(analyzed.map((decision) => decision.recommendationVisitMargin)) ?? null,
    averageRecommendationQMargin: averageFinite(analyzed.map((decision) => decision.recommendationQMargin)) ?? null,
    averageRecommendationUncertainty: averageFinite(analyzed.map((decision) => decision.recommendationUncertainty)) ?? null,
    averageSearchIterations: averageFinite(analyzed.map((decision) => decision.searchIterations)) ?? 0,
    averageRootVisits: averageFinite(analyzed.map((decision) => decision.rootVisits)) ?? 0,
    phasePlanActions: analyzed.filter((decision) => Number(decision.phasePlanActions || 0) > 0).length,
    phasePlanReusedActions: analyzed.filter((decision) => decision.phasePlanReused === true).length,
    phasePlanSearchActions: analyzed.filter((decision) => (
      Number(decision.phasePlanActions || 0) > 0 && decision.phasePlanReused !== true
    )).length,
    selectedActionShare: averageFinite(analyzed.map((decision) => decision.selectedVisitShare)) ?? 0,
    recommendationLabels: countBy(analyzed, (decision) => decision.recommendationLabel || "unknown"),
    actionTypes: countBy(roleActions, (action) => action.action?.type || "unknown"),
  };
}

function recommendationUncertainty(recommendation, policy) {
  const confidence = Number(recommendation?.confidence);
  if (Number.isFinite(confidence)) return round(1 - clamp(confidence, 0, 1), 6);
  const visitMargin = Number(recommendation?.visitMargin);
  const entropy = policyEntropy(policy || []);
  if (Number.isFinite(visitMargin)) return round(((1 - clamp(Math.abs(visitMargin), 0, 1)) + entropy) / 2, 6);
  return Number.isFinite(entropy) ? entropy : null;
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
  unresolved = 0,
  errors = 0,
  scoredGames = 0,
  candidateScore = 0,
} = {}) {
  const games = candidateWins + baselineWins + errors;
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
    decisiveRate: games > 0 ? round((candidateWins + baselineWins) / games, 4) : 0,
    unresolvedRate: games > 0 ? round(unresolved / games, 4) : 0,
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
