export function flattenTrainingLogs(logs) {
  return logs.flatMap((log, fileIndex) => {
    const source = log?.source || log?.path || `input-${fileIndex + 1}`;
    const entries = Array.isArray(log?.entries) ? log.entries : Array.isArray(log) ? log : [];
    return entries.map((entry, entryIndex) => ({
      ...entry,
      __source: source,
      __entryIndex: entryIndex,
    }));
  });
}

export function flattenTrainingEvents(logs) {
  return logs.flatMap((log, fileIndex) => {
    const source = log?.source || log?.path || `input-${fileIndex + 1}`;
    const events = Array.isArray(log?.events) ? log.events : [];
    return events.map((event, eventIndex) => ({
      ...event,
      __source: source,
      __eventIndex: eventIndex,
    }));
  });
}

export function summarizeTrainingEntries(entries) {
  const normalized = Array.isArray(entries) ? entries : [];
  const movementEntries = normalized.filter((entry) => entry.action === "move");
  const combatEntries = normalized.filter((entry) => entry.action === "combat-declaration");
  const retreatEntries = normalized.filter((entry) => entry.action === "retreat");
  const advanceEntries = normalized.filter((entry) => entry.action === "advance");

  return {
    entries: normalized.length,
    sources: countBy(normalized, (entry) => entry.__source || "unknown"),
    actions: countBy(normalized, (entry) => entry.action || "unknown"),
    sides: countBy(normalized, (entry) => entry.side || entry.controllerSide || "unknown"),
    phases: countBy(normalized, (entry) => entry.phaseId || "unknown"),
    turns: countBy(normalized, (entry) => `T${entry.turn ?? "?"}`),
    movement: summarizeScoredChoices(movementEntries),
    combat: summarizeScoredChoices(combatEntries),
    retreat: summarizeChoiceMatches(retreatEntries),
    advance: summarizeChoiceMatches(advanceEntries),
    worstMovementDivergences: worstScoredDivergences(movementEntries, 12),
    worstCombatDivergences: worstScoredDivergences(combatEntries, 12),
    retreatMismatches: choiceMismatches(retreatEntries, 12),
    advanceMismatches: choiceMismatches(advanceEntries, 12),
    schemaGaps: trainingSchemaGaps(normalized),
  };
}

export function summarizeTrainingEvents(events) {
  const normalized = Array.isArray(events) ? events : [];
  const alphaRecommendationEvents = normalized.filter((event) => event.type === "ALPHA_ACTION_RECOMMENDED");
  return {
    events: normalized.length,
    types: countBy(normalized, (event) => event.type || "unknown"),
    phases: countBy(normalized, (event) => event.phaseId || "unknown"),
    turns: countBy(normalized, (event) => `T${event.turn ?? "?"}`),
    stateHashCoverage: coverage(normalized, (event) => event.stateHashAfter),
    legalActionCoverage: coverage(normalized, (event) => Array.isArray(event.legalActionsBefore) && Array.isArray(event.legalActionsAfter)),
    metricCoverage: coverage(normalized, (event) => event.metricsAfter),
    diceEvents: normalized.filter((event) => event.dieRoll !== undefined || event.roll !== undefined).length,
    alphaRecommendations: summarizeAlphaRecommendationEvents(alphaRecommendationEvents),
    missingReplayFields: replaySchemaGaps(normalized),
  };
}

export function summarizeAlphaRecommendationEvents(events) {
  const normalized = Array.isArray(events) ? events : [];
  const payloads = normalized.map(alphaRecommendationPayload);
  const okEvents = normalized.filter((event, index) => Boolean(event.recommendationOk ?? payloads[index]?.ok));
  const illegalCandidateCounts = payloads
    .map((payload) => Number(payload?.legalSelection?.illegalCandidateCount ?? 0))
    .filter(Number.isFinite);
  const confidences = payloads
    .map((payload) => Number(payload?.snapshot?.action?.confidence ?? payload?.snapshot?.analysis?.recommendation?.confidence))
    .filter(Number.isFinite);
  const rootValues = payloads
    .map((payload) => Number(payload?.snapshot?.analysis?.rootValue))
    .filter(Number.isFinite);
  return {
    events: normalized.length,
    ok: okEvents.length,
    rejected: normalized.length - okEvents.length,
    okRate: normalized.length ? round(okEvents.length / normalized.length, 4) : null,
    reasons: countBy(normalized, (event, index) => event.recommendationReason || payloads[index]?.reason || (payloads[index]?.ok ? "ok" : "unknown")),
    actionTypes: countBy(payloads, (payload) => payload?.action?.type || "none"),
    phases: countBy(normalized, (event) => event.phaseId || "unknown"),
    sides: countBy(normalized, (event, index) => event.side || payloads[index]?.context?.side || "unknown"),
    snapshotStatuses: countBy(payloads, (payload) => payload?.snapshot?.status || "none"),
    selectedSources: countBy(payloads, (payload) => payload?.legalSelection?.selectedSource || "none"),
    candidateFallbacks: payloads.filter((payload) => {
      const source = payload?.legalSelection?.selectedSource;
      return source && source !== "direct";
    }).length,
    illegalCandidatesSkipped: illegalCandidateCounts.reduce((sum, count) => sum + count, 0),
    averageIllegalCandidatesSkipped: average(illegalCandidateCounts),
    averageConfidence: average(confidences),
    averageRootValue: average(rootValues),
    actionCoverage: coverage(payloads, (payload) => payload?.action?.type),
    legalSelectionCoverage: coverage(payloads, (payload) => payload?.legalSelection),
    snapshotCoverage: coverage(payloads, (payload) => payload?.snapshot),
    awarenessCoverage: coverage(payloads, (payload) => payload?.snapshot?.analysis?.awareness),
    searchCoverage: coverage(payloads, (payload) => payload?.snapshot?.analysis?.search),
    trustedModelCoverage: coverage(payloads, (payload) => payload?.snapshot?.model?.ok === true),
  };
}

export function summarizeScoredChoices(entries) {
  const scored = entries.filter((entry) => Number.isFinite(Number(entry.scoreDelta)));
  const deltas = scored.map((entry) => Number(entry.scoreDelta));
  const humanAboveAi = scored.filter((entry) => Number(entry.scoreDelta) > 0).length;
  const equal = scored.filter((entry) => Number(entry.scoreDelta) === 0).length;
  const aiAboveHuman = scored.filter((entry) => Number(entry.scoreDelta) < 0).length;
  return {
    entries: entries.length,
    scored: scored.length,
    averageScoreDelta: average(deltas),
    minScoreDelta: deltas.length ? Math.min(...deltas) : null,
    maxScoreDelta: deltas.length ? Math.max(...deltas) : null,
    humanAboveAi,
    equal,
    aiAboveHuman,
    top3Rate: movementTopRate(entries, 3),
    top5Rate: movementTopRate(entries, 5),
  };
}

export function summarizeChoiceMatches(entries) {
  const compared = entries.filter((entry) => choiceKey(entry.humanChoice) && choiceKey(entry.aiChoice));
  const matches = compared.filter((entry) => choiceKey(entry.humanChoice) === choiceKey(entry.aiChoice)).length;
  return {
    entries: entries.length,
    compared: compared.length,
    matches,
    mismatches: compared.length - matches,
    matchRate: compared.length ? round(matches / compared.length, 4) : null,
  };
}

export function worstScoredDivergences(entries, limit = 12) {
  return entries
    .filter((entry) => Number.isFinite(Number(entry.scoreDelta)))
    .sort((a, b) => Number(a.scoreDelta) - Number(b.scoreDelta))
    .slice(0, limit)
    .map((entry) => ({
      id: entry.id || null,
      source: entry.__source || null,
      turn: entry.turn ?? null,
      phaseId: entry.phaseId || null,
      side: entry.side || null,
      action: entry.action,
      scoreDelta: Number(entry.scoreDelta),
      unitId: entry.unit?.id || null,
      fromHexId: entry.fromHexId || entry.unit?.hexId || null,
      humanChoice: compactChoice(entry.humanChoice),
      aiChoice: compactChoice(entry.aiChoice),
      legalChoiceCount: entry.legalChoiceCount ?? null,
    }));
}

export function choiceMismatches(entries, limit = 12) {
  return entries
    .filter((entry) => choiceKey(entry.humanChoice) && choiceKey(entry.aiChoice))
    .filter((entry) => choiceKey(entry.humanChoice) !== choiceKey(entry.aiChoice))
    .slice(0, limit)
    .map((entry) => ({
      id: entry.id || null,
      source: entry.__source || null,
      turn: entry.turn ?? null,
      phaseId: entry.phaseId || null,
      side: entry.side || entry.controllerSide || null,
      action: entry.action,
      unitId: entry.unit?.id || null,
      humanChoice: compactChoice(entry.humanChoice),
      aiChoice: compactChoice(entry.aiChoice),
      legalChoiceCount: entry.legalChoiceCount ?? null,
    }));
}

export function trainingSchemaGaps(entries) {
  const gapCounts = {
    missingStateSnapshot: 0,
    missingLegalActions: 0,
    missingDiceOrCombatResult: 0,
    missingWinner: 0,
    missingStateHash: 0,
  };
  for (const entry of entries) {
    if (!entry.stateBefore && !entry.stateSnapshot) gapCounts.missingStateSnapshot += 1;
    if (!Array.isArray(entry.legalActionsBefore) && !Array.isArray(entry.legalActions)) gapCounts.missingLegalActions += 1;
    if (entry.action === "combat-declaration" && entry.roll === undefined && entry.result === undefined) gapCounts.missingDiceOrCombatResult += 1;
    if (!entry.winner && !entry.gameResult) gapCounts.missingWinner += 1;
    if (!entry.stateHash) gapCounts.missingStateHash += 1;
  }
  return {
    ...gapCounts,
    notes: [
      "current records are useful for heuristic preference tuning",
      "full replay training still needs state snapshots or event-sourced reconstruction",
      "combat outcome learning needs combat resolution events with injected dice and result codes",
    ],
  };
}

export function replaySchemaGaps(events) {
  const gapCounts = {
    missingStateHashAfter: 0,
    missingLegalActionsBefore: 0,
    missingLegalActionsAfter: 0,
    missingMetricsBefore: 0,
    missingMetricsAfter: 0,
    missingStateBefore: 0,
    missingStateAfter: 0,
    combatResolvedMissingDice: 0,
  };
  for (const event of events) {
    if (!event.stateHashAfter) gapCounts.missingStateHashAfter += 1;
    if (!Array.isArray(event.legalActionsBefore)) gapCounts.missingLegalActionsBefore += 1;
    if (!Array.isArray(event.legalActionsAfter)) gapCounts.missingLegalActionsAfter += 1;
    if (!event.metricsBefore && event.type !== "GAME_STARTED") gapCounts.missingMetricsBefore += 1;
    if (!event.metricsAfter) gapCounts.missingMetricsAfter += 1;
    if (!event.stateBefore && event.type !== "GAME_STARTED") gapCounts.missingStateBefore += 1;
    if (!event.stateAfter) gapCounts.missingStateAfter += 1;
    if (event.type === "COMBAT_RESOLVED" && !event.skipped && event.dieRoll === undefined && event.roll === undefined) {
      gapCounts.combatResolvedMissingDice += 1;
    }
  }
  return gapCounts;
}

export function choiceKey(choice) {
  if (!choice) return "";
  if (choice.hexId) return `hex:${choice.hexId}`;
  if (choice.id) return `unit:${choice.id}`;
  if (choice.defenderId || choice.attackerIds) {
    return `combat:${choice.defenderId || ""}:${(choice.attackerIds || []).slice().sort().join("+")}`;
  }
  return JSON.stringify(choice);
}

function alphaRecommendationPayload(event) {
  if (event?.alpha && typeof event.alpha === "object" && !Array.isArray(event.alpha)) return event.alpha;
  return event && typeof event === "object" && !Array.isArray(event) ? event : {};
}

function compactChoice(choice) {
  if (!choice) return null;
  const compact = {};
  for (const key of ["id", "hexId", "hex", "defenderId", "defenderHexId", "score", "odds"]) {
    if (choice[key] !== undefined) compact[key] = choice[key];
  }
  if (choice.attackerIds) compact.attackerIds = choice.attackerIds.slice();
  if (choice.attackerHexIds) compact.attackerHexIds = choice.attackerHexIds.slice();
  if (choice.route?.path) compact.path = choice.route.path.slice();
  return compact;
}

function movementTopRate(entries, limit) {
  const moveEntries = entries.filter((entry) => Array.isArray(entry.aiTopChoices) && entry.humanChoice?.hexId);
  if (!moveEntries.length) return null;
  const hits = moveEntries.filter((entry) => (
    entry.aiTopChoices.slice(0, limit).some((choice) => choice.hexId === entry.humanChoice.hexId)
  )).length;
  return round(hits / moveEntries.length, 4);
}

function countBy(entries, selector) {
  return entries.reduce((counts, entry, index) => {
    const key = selector(entry, index);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function coverage(entries, selector) {
  if (!entries.length) return null;
  const covered = entries.filter((entry) => Boolean(selector(entry))).length;
  return round(covered / entries.length, 4);
}

function average(values) {
  if (!values.length) return null;
  return round(values.reduce((sum, value) => sum + value, 0) / values.length, 4);
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
