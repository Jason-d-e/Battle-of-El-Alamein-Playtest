import {
  OnlineMultiplayerProtocolError,
  cloneOnlineJson,
  computeOnlineStateHash,
  validateStructuredGameAction,
} from "./online-multiplayer-protocol.js";

export const ONLINE_GAME_STATE_SCHEMA = "el-alamein-online-state/v1";
export const ONLINE_RULESET_SCHEMA = "el-alamein-online-ruleset/v1";

function requireFunction(value, label) {
  if (typeof value !== "function") throw new TypeError(`${label} must be a function`);
  return value;
}

function requireCore(core) {
  if (!core || typeof core !== "object") throw new TypeError("core must be an object");
  for (const method of ["createEnvironment", "applyEnvironmentAction", "activeSide"]) {
    requireFunction(core[method], `core.${method}`);
  }
  return core;
}

function cloneField(state, key, fallback) {
  return cloneOnlineJson(state[key] ?? fallback, `state.${key}`);
}

export function projectOnlineAuthoritativeState(state) {
  if (!state || typeof state !== "object" || Array.isArray(state) || !Array.isArray(state.units)) {
    throw new OnlineMultiplayerProtocolError(
      "INVALID_GAME_STATE",
      "online authoritative state must contain a units array",
    );
  }
  return {
    schema: ONLINE_GAME_STATE_SCHEMA,
    version: Number(state.version || 2),
    turn: Number(state.turn),
    phaseIndex: Number(state.phaseIndex),
    combatMode: typeof state.combatMode === "string" ? state.combatMode : "declare",
    declaredCombats: cloneField(state, "declaredCombats", []),
    combatCompleteNotified: Boolean(state.combatCompleteNotified),
    movedUnits: cloneField(state, "movedUnits", []),
    usedAttackers: cloneField(state, "usedAttackers", []),
    usedDefenders: cloneField(state, "usedDefenders", []),
    lastMove: cloneField(state, "lastMove", null),
    retreatTask: cloneField(state, "retreatTask", null),
    advanceTask: cloneField(state, "advanceTask", null),
    lastCombatResult: cloneField(state, "lastCombatResult", null),
    battleReports: cloneField(state, "battleReports", []),
    eliminatedUnitIds: cloneField(state, "eliminatedUnitIds", []),
    losses: cloneField(state, "losses", {}),
    initialStrength: cloneField(state, "initialStrength", {}),
    winner: cloneField(state, "winner", null),
    units: cloneField(state, "units", []),
  };
}

export async function computeOnlineRulesetHash({ scenario, rules }) {
  if (!scenario || typeof scenario !== "object" || !rules || typeof rules !== "object") {
    throw new TypeError("scenario and rules must be objects");
  }
  return computeOnlineStateHash({
    schema: ONLINE_RULESET_SCHEMA,
    scenario,
    rules,
  });
}

export function createOnlineGameCommandExecutor({ core: coreInput, scenario, rules, board = null } = {}) {
  const core = requireCore(coreInput);
  if (!scenario || typeof scenario !== "object" || !rules || typeof rules !== "object") {
    throw new TypeError("scenario and rules must be objects");
  }

  return async function executeOnlineGameCommand({ action, authoritativeState } = {}) {
    const structuredAction = validateStructuredGameAction(action);
    const sourceState = projectOnlineAuthoritativeState(authoritativeState);
    const environment = core.createEnvironment({ scenario, rules, board, state: sourceState });
    const result = core.applyEnvironmentAction(environment, structuredAction, {
      enrichEvents: false,
    });
    if (!result?.ok) {
      throw new OnlineMultiplayerProtocolError(
        "ILLEGAL_GAME_ACTION",
        `core rejected online action: ${result?.reason || "illegal_action"}`,
        { action: structuredAction, reason: result?.reason || "illegal_action" },
      );
    }

    const candidateState = projectOnlineAuthoritativeState(result.state);
    return {
      action: result.action || structuredAction,
      candidateState,
      candidateActiveSide: candidateState.winner ? null : core.activeSide(result.environment),
      events: cloneOnlineJson(result.events || [], "events"),
    };
  };
}

export function rollOnlineFriendDie({ cryptoProvider = globalThis.crypto } = {}) {
  const getRandomValues = cryptoProvider?.getRandomValues?.bind(cryptoProvider);
  if (typeof getRandomValues !== "function") {
    throw new OnlineMultiplayerProtocolError(
      "RNG_UNAVAILABLE",
      "Web Crypto is required to roll an online friend-match die",
    );
  }
  const byte = new Uint8Array(1);
  do {
    getRandomValues(byte);
  } while (byte[0] >= 252);
  return (byte[0] % 6) + 1;
}
