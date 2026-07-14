export const ONLINE_PROTOCOL_VERSION = "el-alamein-online/v1";

export const ONLINE_PLAYER_SIDE = Object.freeze({
  AXIS: "axis",
  ALLIED: "allied",
});

export const ONLINE_ROOM_STATUS = Object.freeze({
  LOBBY: "lobby",
  ACTIVE: "active",
  FINISHED: "finished",
  ABANDONED: "abandoned",
  EXPIRED: "expired",
});

export const ONLINE_COMMAND_TYPE = Object.freeze({
  CREATE_ROOM: "CREATE_ROOM",
  JOIN_ROOM: "JOIN_ROOM",
  SELECT_SIDE: "SELECT_SIDE",
  SET_READY: "SET_READY",
  LEAVE_ROOM: "LEAVE_ROOM",
  GAME_ACTION: "GAME_ACTION",
});

export const ONLINE_EVENT_TYPE = Object.freeze({
  ROOM_CREATED: "ROOM_CREATED",
  PLAYER_JOINED: "PLAYER_JOINED",
  SIDE_SELECTED: "SIDE_SELECTED",
  PLAYER_READY: "PLAYER_READY",
  ROOM_ACTIVATED: "ROOM_ACTIVATED",
  GAME_ACTION_ACCEPTED: "GAME_ACTION_ACCEPTED",
  PLAYER_LEFT: "PLAYER_LEFT",
  ROOM_RECONNECTED: "ROOM_RECONNECTED",
  ROOM_ERROR: "ROOM_ERROR",
});

const ROOM_CODE_PATTERN = /^[A-Z0-9]{4,8}$/;
const STATE_HASH_PATTERN = /^sha256:[0-9a-f]{64}$/;
const PLAYER_SIDES = new Set(Object.values(ONLINE_PLAYER_SIDE));
const ROOM_STATUSES = new Set(Object.values(ONLINE_ROOM_STATUS));
const COMMAND_TYPES = new Set(Object.values(ONLINE_COMMAND_TYPE));
const EVENT_TYPES = new Set(Object.values(ONLINE_EVENT_TYPE));

export class OnlineMultiplayerProtocolError extends Error {
  constructor(code, message, details = null) {
    super(`[${code}] ${message}`);
    this.name = "OnlineMultiplayerProtocolError";
    this.code = code;
    this.details = details;
  }
}

function fail(code, message, details = null) {
  throw new OnlineMultiplayerProtocolError(code, message, details);
}

function isPlainObject(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function assertPlainObject(value, label) {
  if (!isPlainObject(value)) fail("INVALID_OBJECT", `${label} must be a plain object`);
  return value;
}

function assertOpaqueString(value, label, code = "INVALID_STRING") {
  if (typeof value !== "string" || value.trim() === "") {
    fail(code, `${label} must be a non-empty string`);
  }
  return value.trim();
}

function assertJsonValue(value, label = "value", active = new Set()) {
  if (value === null) return;
  const type = typeof value;
  if (type === "string" || type === "boolean") return;
  if (type === "number") {
    if (!Number.isFinite(value)) fail("NOT_JSON_SAFE", `${label} must contain only finite numbers`);
    return;
  }
  if (type !== "object") fail("NOT_JSON_SAFE", `${label} contains unsupported ${type} data`);
  if (active.has(value)) fail("NOT_JSON_SAFE", `${label} contains a cycle`);

  active.add(value);
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      if (!Object.prototype.hasOwnProperty.call(value, index)) {
        fail("NOT_JSON_SAFE", `${label} must not contain sparse array entries`);
      }
      assertJsonValue(value[index], `${label}[${index}]`, active);
    }
  } else {
    assertPlainObject(value, label);
    for (const [key, entry] of Object.entries(value)) {
      assertJsonValue(entry, `${label}.${key}`, active);
    }
  }
  active.delete(value);
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const entry of Object.values(value)) deepFreeze(entry);
  return value;
}

export function cloneOnlineJson(value, label = "value") {
  assertJsonValue(value, label);
  return JSON.parse(JSON.stringify(value));
}

function canonicalizeJsonValue(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalizeJsonValue(entry)).join(",")}]`;
  }
  const entries = Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalizeJsonValue(value[key])}`);
  return `{${entries.join(",")}}`;
}

export function canonicalizeOnlineJson(value, label = "value") {
  assertJsonValue(value, label);
  return canonicalizeJsonValue(value);
}

export async function computeOnlineStateHash(value, {
  cryptoProvider = globalThis.crypto,
} = {}) {
  if (typeof TextEncoder !== "function" || typeof cryptoProvider?.subtle?.digest !== "function") {
    fail(
      "STATE_HASH_UNAVAILABLE",
      "Web Crypto SHA-256 is required to compute online state hashes",
    );
  }
  const canonicalState = canonicalizeOnlineJson(value, "authoritativeState");
  const digest = await cryptoProvider.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(canonicalState),
  );
  const bytes = new Uint8Array(digest);
  if (bytes.length !== 32) {
    fail("INVALID_STATE_HASH_DIGEST", "SHA-256 state hash digest must contain 32 bytes");
  }
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `sha256:${hex}`;
}

export function normalizeOnlineRoomCode(value) {
  const code = String(value ?? "").trim().toUpperCase();
  if (!ROOM_CODE_PATTERN.test(code)) {
    fail("INVALID_ROOM_CODE", "roomCode must contain 4-8 uppercase letters or digits", { roomCode: code });
  }
  return code;
}

export function validateOnlinePlayerSide(value, label = "playerSide") {
  if (!PLAYER_SIDES.has(value)) fail("ILLEGAL_SIDE", `${label} must be axis or allied`, { side: value });
  return value;
}

export function validateOnlineRevision(value, label = "expectedRevision") {
  if (!Number.isSafeInteger(value) || value < 0) {
    fail("INVALID_REVISION", `${label} must be a nonnegative safe integer`, { revision: value });
  }
  return value;
}

export function validateOnlineStateHash(value, label = "expectedStateHash") {
  const hash = assertOpaqueString(value, label, "INVALID_STATE_HASH");
  if (!STATE_HASH_PATTERN.test(hash)) {
    fail("INVALID_STATE_HASH", `${label} must be a lowercase SHA-256 hash`, { stateHash: hash });
  }
  return hash;
}

export function validateStructuredGameAction(action, label = "action") {
  assertPlainObject(action, label);
  assertOpaqueString(action.type, `${label}.type`, "INVALID_GAME_ACTION");
  return deepFreeze(cloneOnlineJson(action, label));
}

export function validateStructuredEvents(events, label = "events") {
  if (!Array.isArray(events)) fail("INVALID_EVENTS", `${label} must be an array`);
  const normalized = events.map((event, index) => {
    assertPlainObject(event, `${label}[${index}]`);
    assertOpaqueString(event.type, `${label}[${index}].type`, "INVALID_EVENT");
    return cloneOnlineJson(event, `${label}[${index}]`);
  });
  return deepFreeze(normalized);
}

function normalizeCommandPayload(type, payload) {
  assertPlainObject(payload, "command.payload");
  const normalized = cloneOnlineJson(payload, "command.payload");

  if (type === ONLINE_COMMAND_TYPE.CREATE_ROOM) {
    validateOnlinePlayerSide(normalized.requestedSide, "command.payload.requestedSide");
    assertPlainObject(normalized.initialState, "command.payload.initialState");
  } else if (type === ONLINE_COMMAND_TYPE.JOIN_ROOM) {
    validateOnlinePlayerSide(normalized.requestedSide, "command.payload.requestedSide");
  } else if (type === ONLINE_COMMAND_TYPE.SELECT_SIDE) {
    validateOnlinePlayerSide(normalized.side, "command.payload.side");
  } else if (type === ONLINE_COMMAND_TYPE.SET_READY) {
    if (typeof normalized.ready !== "boolean") {
      fail("INVALID_READY", "command.payload.ready must be boolean");
    }
  } else if (type === ONLINE_COMMAND_TYPE.LEAVE_ROOM) {
    if (normalized.reason !== undefined) {
      assertOpaqueString(normalized.reason, "command.payload.reason", "INVALID_LEAVE_REASON");
    }
  } else if (type === ONLINE_COMMAND_TYPE.GAME_ACTION) {
    normalized.action = cloneOnlineJson(validateStructuredGameAction(normalized.action), "command.payload.action");
    assertPlainObject(normalized.candidateState, "command.payload.candidateState");
    normalized.candidateStateHash = validateOnlineStateHash(
      normalized.candidateStateHash,
      "command.payload.candidateStateHash",
    );
    normalized.events = cloneOnlineJson(validateStructuredEvents(normalized.events), "command.payload.events");
    if (normalized.candidateActiveSide !== null) {
      validateOnlinePlayerSide(normalized.candidateActiveSide, "command.payload.candidateActiveSide");
    }
  }

  return deepFreeze(normalized);
}

export function validateOnlineCommand(command) {
  assertPlainObject(command, "command");
  if (command.protocolVersion !== ONLINE_PROTOCOL_VERSION) {
    fail("PROTOCOL_MISMATCH", `protocolVersion must equal ${ONLINE_PROTOCOL_VERSION}`);
  }
  if (!COMMAND_TYPES.has(command.type)) fail("UNKNOWN_COMMAND", "command.type is not supported", { type: command.type });

  const normalized = {
    protocolVersion: ONLINE_PROTOCOL_VERSION,
    rulesetHash: assertOpaqueString(command.rulesetHash, "command.rulesetHash", "INVALID_RULESET_HASH"),
    commandId: assertOpaqueString(command.commandId, "command.commandId", "INVALID_COMMAND_ID"),
    type: command.type,
    roomCode: normalizeOnlineRoomCode(command.roomCode),
    playerId: assertOpaqueString(command.playerId, "command.playerId", "INVALID_PLAYER_ID"),
    playerSide: validateOnlinePlayerSide(command.playerSide),
    expectedRevision: validateOnlineRevision(command.expectedRevision),
    expectedStateHash: validateOnlineStateHash(command.expectedStateHash),
    payload: normalizeCommandPayload(command.type, command.payload || {}),
  };

  if (normalized.type === ONLINE_COMMAND_TYPE.CREATE_ROOM && normalized.expectedRevision !== 0) {
    fail("INVALID_REVISION", "CREATE_ROOM must use expectedRevision 0");
  }
  if (
    (normalized.type === ONLINE_COMMAND_TYPE.CREATE_ROOM || normalized.type === ONLINE_COMMAND_TYPE.JOIN_ROOM)
    && normalized.payload.requestedSide !== normalized.playerSide
  ) {
    fail("ILLEGAL_SIDE", "requestedSide must match command.playerSide");
  }
  return deepFreeze(normalized);
}

export function createOnlineCommand(fields) {
  return validateOnlineCommand({ protocolVersion: ONLINE_PROTOCOL_VERSION, ...fields });
}

function normalizePlayers(players) {
  assertPlainObject(players, "room.players");
  const normalized = {};
  for (const side of PLAYER_SIDES) {
    const playerId = players[side] ?? null;
    normalized[side] = playerId === null
      ? null
      : assertOpaqueString(playerId, `room.players.${side}`, "INVALID_PLAYER_ID");
  }
  return normalized;
}

function normalizeReady(ready, players) {
  assertPlainObject(ready, "room.ready");
  const normalized = {};
  for (const side of PLAYER_SIDES) {
    normalized[side] = Boolean(ready[side]);
    if (normalized[side] && players[side] === null) {
      fail("INVALID_READY", `room.ready.${side} cannot be true without a player`);
    }
  }
  return normalized;
}

export function validateOnlineRoomSnapshot(snapshot) {
  assertPlainObject(snapshot, "room");
  if (snapshot.protocolVersion !== ONLINE_PROTOCOL_VERSION) {
    fail("PROTOCOL_MISMATCH", `room.protocolVersion must equal ${ONLINE_PROTOCOL_VERSION}`);
  }
  if (!ROOM_STATUSES.has(snapshot.status)) {
    fail("INVALID_ROOM_STATUS", "room.status is not supported", { status: snapshot.status });
  }

  const players = normalizePlayers(snapshot.players || {});
  const ready = normalizeReady(snapshot.ready || {}, players);
  if (players.axis && players.axis === players.allied) {
    fail("DUPLICATE_PLAYER", "one playerId cannot occupy both sides");
  }
  const activeSide = snapshot.activeSide === null || snapshot.activeSide === undefined
    ? null
    : validateOnlinePlayerSide(snapshot.activeSide, "room.activeSide");
  assertPlainObject(snapshot.authoritativeState, "room.authoritativeState");

  if (snapshot.status === ONLINE_ROOM_STATUS.ACTIVE) {
    if (!players.axis || !players.allied || !ready.axis || !ready.allied) {
      fail("INVALID_ACTIVE_ROOM", "an active room requires two assigned and ready players");
    }
    if (!activeSide) fail("INVALID_ACTIVE_ROOM", "an active room must identify activeSide");
  }

  const normalized = {
    protocolVersion: ONLINE_PROTOCOL_VERSION,
    rulesetHash: assertOpaqueString(snapshot.rulesetHash, "room.rulesetHash", "INVALID_RULESET_HASH"),
    roomCode: normalizeOnlineRoomCode(snapshot.roomCode),
    status: snapshot.status,
    revision: validateOnlineRevision(snapshot.revision, "room.revision"),
    stateHash: validateOnlineStateHash(snapshot.stateHash, "room.stateHash"),
    players,
    ready,
    activeSide,
    authoritativeState: cloneOnlineJson(snapshot.authoritativeState, "room.authoritativeState"),
    lastCommandId: snapshot.lastCommandId === null || snapshot.lastCommandId === undefined
      ? null
      : assertOpaqueString(snapshot.lastCommandId, "room.lastCommandId", "INVALID_COMMAND_ID"),
  };
  return deepFreeze(normalized);
}

export function onlinePlayerSideForId(roomSnapshot, playerId) {
  const room = validateOnlineRoomSnapshot(roomSnapshot);
  const normalizedPlayerId = assertOpaqueString(playerId, "playerId", "INVALID_PLAYER_ID");
  return Object.values(ONLINE_PLAYER_SIDE).find((side) => room.players[side] === normalizedPlayerId) || null;
}

export function assertOnlineCommandAgainstRoom(commandInput, roomInput) {
  const command = validateOnlineCommand(commandInput);
  const room = validateOnlineRoomSnapshot(roomInput);
  if (command.type === ONLINE_COMMAND_TYPE.CREATE_ROOM) {
    fail("ROOM_ALREADY_EXISTS", "CREATE_ROOM cannot be checked against an existing room");
  }
  if (command.roomCode !== room.roomCode) fail("ROOM_MISMATCH", "command roomCode does not match room snapshot");
  if (command.rulesetHash !== room.rulesetHash) fail("RULESET_MISMATCH", "command rulesetHash does not match room");
  if (command.expectedRevision !== room.revision) {
    fail("STALE_REVISION", "command expectedRevision does not match room revision", {
      expectedRevision: command.expectedRevision,
      actualRevision: room.revision,
    });
  }
  if (command.expectedStateHash !== room.stateHash) {
    fail("STATE_HASH_MISMATCH", "command expectedStateHash does not match room stateHash");
  }

  if (command.type === ONLINE_COMMAND_TYPE.JOIN_ROOM) {
    const occupant = room.players[command.playerSide];
    if (occupant && occupant !== command.playerId) {
      fail("SIDE_UNAVAILABLE", `${command.playerSide} is already assigned`);
    }
    const otherSide = command.playerSide === ONLINE_PLAYER_SIDE.AXIS
      ? ONLINE_PLAYER_SIDE.ALLIED
      : ONLINE_PLAYER_SIDE.AXIS;
    if (room.players[otherSide] === command.playerId) {
      fail("PLAYER_ALREADY_ASSIGNED", "playerId is already assigned to the other side");
    }
    if (room.status !== ONLINE_ROOM_STATUS.LOBBY) fail("ROOM_NOT_JOINABLE", "only lobby rooms accept new players");
    return true;
  }

  if (room.players[command.playerSide] !== command.playerId) {
    fail("ILLEGAL_SIDE", "playerId is not assigned to command.playerSide");
  }

  if (command.type === ONLINE_COMMAND_TYPE.SELECT_SIDE) {
    if (room.status !== ONLINE_ROOM_STATUS.LOBBY) fail("ROOM_NOT_LOBBY", "side selection is limited to lobby rooms");
    const occupant = room.players[command.payload.side];
    if (occupant && occupant !== command.playerId) {
      fail("SIDE_UNAVAILABLE", `${command.payload.side} is already assigned`);
    }
  }
  if (command.type === ONLINE_COMMAND_TYPE.SET_READY && room.status !== ONLINE_ROOM_STATUS.LOBBY) {
    fail("ROOM_NOT_LOBBY", "ready changes are limited to lobby rooms");
  }
  if (command.type === ONLINE_COMMAND_TYPE.GAME_ACTION) {
    if (room.status !== ONLINE_ROOM_STATUS.ACTIVE) fail("ROOM_NOT_ACTIVE", "game actions require an active room");
    if (room.activeSide !== command.playerSide) {
      fail("ILLEGAL_SIDE", "only the active side may submit a game action", {
        activeSide: room.activeSide,
        playerSide: command.playerSide,
      });
    }
  }
  return true;
}

export function validateOnlineProtocolEvent(event) {
  assertPlainObject(event, "event");
  if (event.protocolVersion !== ONLINE_PROTOCOL_VERSION) {
    fail("PROTOCOL_MISMATCH", `event.protocolVersion must equal ${ONLINE_PROTOCOL_VERSION}`);
  }
  if (!EVENT_TYPES.has(event.type)) fail("UNKNOWN_EVENT", "event.type is not supported", { type: event.type });
  const normalized = {
    protocolVersion: ONLINE_PROTOCOL_VERSION,
    rulesetHash: assertOpaqueString(event.rulesetHash, "event.rulesetHash", "INVALID_RULESET_HASH"),
    eventId: assertOpaqueString(event.eventId, "event.eventId", "INVALID_EVENT_ID"),
    commandId: event.commandId === null || event.commandId === undefined
      ? null
      : assertOpaqueString(event.commandId, "event.commandId", "INVALID_COMMAND_ID"),
    type: event.type,
    roomCode: normalizeOnlineRoomCode(event.roomCode),
    playerSide: event.playerSide === null || event.playerSide === undefined
      ? null
      : validateOnlinePlayerSide(event.playerSide),
    revision: validateOnlineRevision(event.revision, "event.revision"),
    stateHash: validateOnlineStateHash(event.stateHash, "event.stateHash"),
    payload: cloneOnlineJson(event.payload || {}, "event.payload"),
  };
  return deepFreeze(normalized);
}

export function createOnlineProtocolEvent(fields) {
  return validateOnlineProtocolEvent({ protocolVersion: ONLINE_PROTOCOL_VERSION, ...fields });
}
