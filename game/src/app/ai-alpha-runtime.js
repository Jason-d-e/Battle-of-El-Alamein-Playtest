import { createBoard, createEnvironment } from "../core/index.js";
import { analyzePosition } from "./ai-alpha-search.js";
import { normalizeAlphaModelArtifact } from "./ai-alpha-model.js";

export const ALPHA_RUNTIME_MESSAGE = Object.freeze({
  ANALYZE: "ANALYZE_ALPHA_POSITION",
  CHOOSE_ACTION: "CHOOSE_ALPHA_ACTION",
});

export function alphaSearchOptionsFromModel(model = null, options = {}) {
  return {
    ...options,
    model: normalizeAlphaModel(model),
  };
}

export function analyzeAlphaPosition({
  scenario,
  rules,
  board = null,
  state,
  model = null,
  searchOptions = {},
} = {}) {
  const environment = createEnvironment({
    scenario,
    rules,
    board: board || createBoard(scenario),
    state,
  });
  return analyzePosition(environment, alphaSearchOptionsFromModel(model, searchOptions));
}

export function chooseAlphaRuntimeAction({
  scenario,
  rules,
  board = null,
  state,
  model = null,
  searchOptions = {},
} = {}) {
  return chooseAlphaRuntimeDecision({
    scenario,
    rules,
    board,
    state,
    model,
    searchOptions,
  }).action;
}

export function chooseAlphaRuntimeDecision({
  scenario,
  rules,
  board = null,
  state,
  model = null,
  searchOptions = {},
} = {}) {
  const environment = createEnvironment({
    scenario,
    rules,
    board: board || createBoard(scenario),
    state,
  });
  const analysis = analyzePosition(environment, alphaSearchOptionsFromModel(model, searchOptions));
  return {
    action: analysis.requiresChance ? null : analysis.bestAction,
    analysis,
  };
}

export function handleAlphaRuntimeMessage(message) {
  try {
    if (!message || typeof message !== "object") return errorResponse(message, "invalid_message");
    if (message.type === ALPHA_RUNTIME_MESSAGE.ANALYZE) {
      return {
        id: message.id || null,
        type: `${message.type}_RESULT`,
        ok: true,
        analysis: analyzeAlphaPosition(message.payload || {}),
      };
    }
    if (message.type === ALPHA_RUNTIME_MESSAGE.CHOOSE_ACTION) {
      const decision = chooseAlphaRuntimeDecision(message.payload || {});
      return {
        id: message.id || null,
        type: `${message.type}_RESULT`,
        ok: true,
        action: decision.action,
        analysis: decision.analysis,
      };
    }
    return errorResponse(message, "unknown_message_type");
  } catch (error) {
    return errorResponse(message, "alpha_runtime_failed", error);
  }
}

function normalizeAlphaModel(model) {
  return normalizeAlphaModelArtifact(model);
}

function errorResponse(message, reason, error = null) {
  return {
    id: message?.id || null,
    type: `${message?.type || "ALPHA_RUNTIME"}_RESULT`,
    ok: false,
    reason,
    message: error ? String(error?.message || error) : undefined,
  };
}
