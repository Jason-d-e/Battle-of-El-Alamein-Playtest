import {
  ONLINE_ROOM_STATUS,
  OnlineMultiplayerProtocolError,
  onlinePlayerSideForId,
  validateOnlineRoomSnapshot,
} from "./online-multiplayer-protocol.js";

export const ONLINE_CONNECTION_STATE = Object.freeze({
  OFFLINE: "offline",
  CONNECTING: "connecting",
  CONNECTED: "connected",
  DISCONNECTED: "disconnected",
  RECONNECTING: "reconnecting",
  LEAVING: "leaving",
  ERROR: "error",
});

export const ONLINE_CLIENT_EVENT = Object.freeze({
  OPERATION_STARTED: "OPERATION_STARTED",
  ROOM_SYNCED: "ROOM_SYNCED",
  CONNECTION_LOST: "CONNECTION_LOST",
  ROOM_LEFT: "ROOM_LEFT",
  ERROR_RAISED: "ERROR_RAISED",
  ERROR_CLEARED: "ERROR_CLEARED",
});

export const ONLINE_OPERATION = Object.freeze({
  CREATE: "create",
  JOIN: "join",
  SELECT_SIDE: "select-side",
  READY: "ready",
  GAME_ACTION: "game-action",
  RECONNECT: "reconnect",
  LEAVE: "leave",
});

const OPERATIONS = new Set(Object.values(ONLINE_OPERATION));

function normalizeClientError(error, operation = null) {
  const code = typeof error?.code === "string" && error.code ? error.code : "ONLINE_ERROR";
  const message = typeof error?.message === "string" && error.message ? error.message : code;
  return Object.freeze({
    code,
    message,
    operation,
    recoverable: error?.recoverable !== false,
  });
}

function freezeState(state) {
  if (state.error && !Object.isFrozen(state.error)) Object.freeze(state.error);
  return Object.freeze(state);
}

export function createOnlineMultiplayerState({ playerId = null } = {}) {
  return freezeState({
    mode: "offline",
    connection: ONLINE_CONNECTION_STATE.OFFLINE,
    operation: null,
    playerId,
    playerSide: null,
    room: null,
    error: null,
    lastEvent: null,
  });
}

export function reduceOnlineMultiplayerState(currentState, event) {
  const state = currentState || createOnlineMultiplayerState();
  if (!event || typeof event !== "object" || typeof event.type !== "string") {
    throw new OnlineMultiplayerProtocolError("INVALID_CLIENT_EVENT", "client state events must be structured objects");
  }

  if (event.type === ONLINE_CLIENT_EVENT.OPERATION_STARTED) {
    if (!OPERATIONS.has(event.operation)) {
      throw new OnlineMultiplayerProtocolError("INVALID_OPERATION", "unknown online client operation");
    }
    const connection = event.operation === ONLINE_OPERATION.RECONNECT
      ? ONLINE_CONNECTION_STATE.RECONNECTING
      : event.operation === ONLINE_OPERATION.LEAVE
        ? ONLINE_CONNECTION_STATE.LEAVING
        : state.room
          ? ONLINE_CONNECTION_STATE.CONNECTED
          : ONLINE_CONNECTION_STATE.CONNECTING;
    return freezeState({
      ...state,
      mode: "online",
      connection,
      operation: event.operation,
      error: null,
      lastEvent: event.type,
    });
  }

  if (event.type === ONLINE_CLIENT_EVENT.ROOM_SYNCED) {
    const room = validateOnlineRoomSnapshot(event.room);
    const playerId = event.playerId || state.playerId;
    const playerSide = onlinePlayerSideForId(room, playerId);
    if (!playerSide && event.requireMembership !== false) {
      throw new OnlineMultiplayerProtocolError("PLAYER_NOT_MEMBER", "playerId is not assigned in the room snapshot");
    }
    return freezeState({
      ...state,
      mode: "online",
      connection: ONLINE_CONNECTION_STATE.CONNECTED,
      operation: null,
      playerId,
      playerSide,
      room,
      error: null,
      lastEvent: event.type,
    });
  }

  if (event.type === ONLINE_CLIENT_EVENT.CONNECTION_LOST) {
    return freezeState({
      ...state,
      mode: state.room ? "online" : "offline",
      connection: state.room ? ONLINE_CONNECTION_STATE.DISCONNECTED : ONLINE_CONNECTION_STATE.OFFLINE,
      operation: null,
      error: normalizeClientError(event.error || { code: "TRANSPORT_DISCONNECTED" }),
      lastEvent: event.type,
    });
  }

  if (event.type === ONLINE_CLIENT_EVENT.ERROR_RAISED) {
    return freezeState({
      ...state,
      mode: state.room || state.mode === "online" ? "online" : "offline",
      connection: ONLINE_CONNECTION_STATE.ERROR,
      operation: null,
      error: normalizeClientError(event.error, event.operation || state.operation),
      lastEvent: event.type,
    });
  }

  if (event.type === ONLINE_CLIENT_EVENT.ERROR_CLEARED) {
    return freezeState({
      ...state,
      connection: state.room ? ONLINE_CONNECTION_STATE.DISCONNECTED : ONLINE_CONNECTION_STATE.OFFLINE,
      operation: null,
      error: null,
      lastEvent: event.type,
    });
  }

  if (event.type === ONLINE_CLIENT_EVENT.ROOM_LEFT) {
    return freezeState({
      ...createOnlineMultiplayerState({ playerId: state.playerId }),
      lastEvent: event.type,
    });
  }

  throw new OnlineMultiplayerProtocolError("UNKNOWN_CLIENT_EVENT", `unsupported client state event ${event.type}`);
}

export function isOnlineRoomConnected(state) {
  return Boolean(state?.room && state.connection === ONLINE_CONNECTION_STATE.CONNECTED);
}

export function canIssueOnlineGameAction(state) {
  return Boolean(
    isOnlineRoomConnected(state)
    && state.room.status === ONLINE_ROOM_STATUS.ACTIVE
    && state.playerSide
    && state.room.activeSide === state.playerSide,
  );
}

export function onlineRoomPlayerReady(state) {
  return Boolean(state?.playerSide && state?.room?.ready?.[state.playerSide]);
}
