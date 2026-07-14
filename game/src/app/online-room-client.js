import {
  ONLINE_COMMAND_TYPE,
  ONLINE_PLAYER_SIDE,
  ONLINE_PROTOCOL_VERSION,
  OnlineMultiplayerProtocolError,
  assertOnlineCommandAgainstRoom,
  cloneOnlineJson,
  computeOnlineStateHash,
  createOnlineCommand,
  normalizeOnlineRoomCode,
  onlinePlayerSideForId,
  validateOnlinePlayerSide,
  validateOnlineRoomSnapshot,
  validateOnlineStateHash,
  validateStructuredEvents,
  validateStructuredGameAction,
} from "./online-multiplayer-protocol.js";
import {
  ONLINE_CLIENT_EVENT,
  ONLINE_CONNECTION_STATE,
  ONLINE_OPERATION,
  canIssueOnlineGameAction,
  createOnlineMultiplayerState,
  reduceOnlineMultiplayerState,
} from "./online-multiplayer-state.js";

const ROOM_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

function requiredString(value, label, code) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new OnlineMultiplayerProtocolError(code, `${label} must be a non-empty string`);
  }
  return value.trim();
}

function defaultCommandIdFactory() {
  if (typeof globalThis.crypto?.randomUUID !== "function") {
    throw new OnlineMultiplayerProtocolError(
      "COMMAND_ID_FACTORY_REQUIRED",
      "createCommandId must be injected when crypto.randomUUID is unavailable",
    );
  }
  return globalThis.crypto.randomUUID();
}

function defaultRoomCodeFactory() {
  if (typeof globalThis.crypto?.getRandomValues !== "function") {
    throw new OnlineMultiplayerProtocolError(
      "ROOM_CODE_FACTORY_REQUIRED",
      "createRoomCode must be injected when crypto.getRandomValues is unavailable",
    );
  }
  const bytes = new Uint8Array(6);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => ROOM_CODE_ALPHABET[value % ROOM_CODE_ALPHABET.length]).join("");
}

function validateTransport(transport) {
  if (!transport || typeof transport !== "object") {
    throw new TypeError("online room transport must be an object");
  }
  for (const method of ["createRoom", "readRoom", "sendCommand"]) {
    if (typeof transport[method] !== "function") {
      throw new TypeError(`online room transport.${method} must be a function`);
    }
  }
  return transport;
}

function asClientError(error, operation) {
  if (error instanceof OnlineMultiplayerProtocolError) return error;
  if (error && typeof error.code === "string" && error.code) {
    const normalized = new OnlineMultiplayerProtocolError(
      error.code,
      typeof error.message === "string" && error.message ? error.message : error.code,
      error.details || null,
    );
    normalized.recoverable = error.recoverable !== false;
    return normalized;
  }
  const normalized = new OnlineMultiplayerProtocolError(
    "TRANSPORT_ERROR",
    typeof error?.message === "string" && error.message ? error.message : `online ${operation} failed`,
  );
  normalized.recoverable = true;
  return normalized;
}

function responseRoom(response) {
  const room = response && typeof response === "object" && Object.prototype.hasOwnProperty.call(response, "room")
    ? response.room
    : response;
  return validateOnlineRoomSnapshot(room);
}

function sameRoomRevisionIdentity(left, right) {
  return left.stateHash === right.stateHash
    && left.status === right.status
    && left.activeSide === right.activeSide
    && left.lastCommandId === right.lastCommandId
    && left.players.axis === right.players.axis
    && left.players.allied === right.players.allied
    && left.ready.axis === right.ready.axis
    && left.ready.allied === right.ready.allied;
}

export function createOnlineRoomClient({
  transport: transportInput,
  playerId: playerIdInput,
  rulesetHash: rulesetHashInput,
  commandExecutor = null,
  hashState = computeOnlineStateHash,
  createCommandId = defaultCommandIdFactory,
  createRoomCode = defaultRoomCodeFactory,
  onListenerError = () => {},
} = {}) {
  const transport = validateTransport(transportInput);
  const playerId = requiredString(playerIdInput, "playerId", "INVALID_PLAYER_ID");
  const rulesetHash = requiredString(rulesetHashInput, "rulesetHash", "INVALID_RULESET_HASH");
  if (commandExecutor !== null && typeof commandExecutor !== "function") {
    throw new TypeError("commandExecutor must be a function when provided");
  }
  if (typeof hashState !== "function") {
    throw new TypeError("hashState must be a function");
  }
  if (typeof createCommandId !== "function" || typeof createRoomCode !== "function") {
    throw new TypeError("createCommandId and createRoomCode must be functions");
  }

  let state = createOnlineMultiplayerState({ playerId });
  let operationInFlight = false;
  let pollInFlight = null;
  let closed = false;
  let commandSequence = 0;
  const listeners = new Set();

  function notify(event) {
    for (const listener of listeners) {
      try {
        listener(state, event);
      } catch (error) {
        onListenerError(error);
      }
    }
  }

  function dispatch(event) {
    state = reduceOnlineMultiplayerState(state, event);
    notify(event);
    return state;
  }

  function ensureOpen() {
    if (closed) throw new OnlineMultiplayerProtocolError("CLIENT_CLOSED", "online room client is closed");
  }

  function nextCommandId(type) {
    commandSequence += 1;
    return requiredString(
      createCommandId({ type, sequence: commandSequence, playerId }),
      "commandId",
      "INVALID_COMMAND_ID",
    );
  }

  function ensureRoomIdentity(room, roomCode) {
    if (room.protocolVersion !== ONLINE_PROTOCOL_VERSION) {
      throw new OnlineMultiplayerProtocolError("PROTOCOL_MISMATCH", "room protocol does not match client");
    }
    if (room.rulesetHash !== rulesetHash) {
      throw new OnlineMultiplayerProtocolError("RULESET_MISMATCH", "room ruleset does not match client");
    }
    if (room.roomCode !== normalizeOnlineRoomCode(roomCode)) {
      throw new OnlineMultiplayerProtocolError("ROOM_MISMATCH", "transport returned a different room");
    }
    return room;
  }

  async function computeValidatedStateHash(authoritativeState, label = "stateHash") {
    const computed = await hashState(cloneOnlineJson(authoritativeState, "authoritativeState"));
    return validateOnlineStateHash(computed, label);
  }

  async function verifyRoomStateHash(room) {
    const computed = await computeValidatedStateHash(room.authoritativeState, "computedStateHash");
    if (computed !== room.stateHash) {
      throw new OnlineMultiplayerProtocolError(
        "STATE_HASH_MISMATCH",
        "room stateHash does not match its authoritativeState",
        { expectedStateHash: room.stateHash, computedStateHash: computed },
      );
    }
    return room;
  }

  function syncRoom(room, operation, requireMembership = true) {
    dispatch({
      type: ONLINE_CLIENT_EVENT.ROOM_SYNCED,
      room,
      playerId,
      operation,
      requireMembership,
    });
    return state.room;
  }

  async function runOperation(operation, task) {
    ensureOpen();
    if (operationInFlight) {
      throw new OnlineMultiplayerProtocolError("OPERATION_IN_PROGRESS", "another online room operation is in progress");
    }
    operationInFlight = true;
    try {
      if (pollInFlight) await pollInFlight.catch(() => null);
      dispatch({ type: ONLINE_CLIENT_EVENT.OPERATION_STARTED, operation });
      return await task();
    } catch (error) {
      const normalized = asClientError(error, operation);
      dispatch({ type: ONLINE_CLIENT_EVENT.ERROR_RAISED, operation, error: normalized });
      throw normalized;
    } finally {
      operationInFlight = false;
    }
  }

  function currentRoomCommand(type, payload) {
    if (!state.room || !state.playerSide) {
      throw new OnlineMultiplayerProtocolError("NOT_IN_ROOM", "player must be assigned in a room");
    }
    const command = createOnlineCommand({
      rulesetHash,
      commandId: nextCommandId(type),
      type,
      roomCode: state.room.roomCode,
      playerId,
      playerSide: state.playerSide,
      expectedRevision: state.room.revision,
      expectedStateHash: state.room.stateHash,
      payload,
    });
    assertOnlineCommandAgainstRoom(command, state.room);
    return command;
  }

  function validateAdvancedRoom(room, command) {
    ensureRoomIdentity(room, command.roomCode);
    if (room.revision <= command.expectedRevision) {
      throw new OnlineMultiplayerProtocolError("STALE_RESPONSE", "transport response did not advance room revision", {
        expectedRevision: command.expectedRevision,
        actualRevision: room.revision,
      });
    }
    if (room.lastCommandId !== command.commandId) {
      throw new OnlineMultiplayerProtocolError(
        "COMMAND_NOT_ACCEPTED",
        "transport response does not identify the submitted command as accepted",
      );
    }
    return room;
  }

  async function sendCurrentRoomCommand(operation, type, payload, { expectedSide = null } = {}) {
    const command = currentRoomCommand(type, payload);
    const response = responseRoom(await transport.sendCommand(command));
    const room = validateAdvancedRoom(await verifyRoomStateHash(response), command);
    if (expectedSide && onlinePlayerSideForId(room, playerId) !== expectedSide) {
      throw new OnlineMultiplayerProtocolError("SIDE_SELECTION_REJECTED", "transport did not assign the requested side");
    }
    return syncRoom(room, operation);
  }

  async function createRoom({
    roomCode = null,
    playerSide = ONLINE_PLAYER_SIDE.AXIS,
    initialState,
    stateHash = null,
  } = {}) {
    return runOperation(ONLINE_OPERATION.CREATE, async () => {
      const normalizedSide = validateOnlinePlayerSide(playerSide);
      const normalizedCode = normalizeOnlineRoomCode(roomCode || createRoomCode({ playerId }));
      const normalizedInitialState = cloneOnlineJson(initialState, "initialState");
      const computedHash = await computeValidatedStateHash(normalizedInitialState);
      const normalizedHash = stateHash === null || stateHash === undefined
        ? computedHash
        : validateOnlineStateHash(stateHash, "stateHash");
      if (normalizedHash !== computedHash) {
        throw new OnlineMultiplayerProtocolError(
          "STATE_HASH_MISMATCH",
          "provided stateHash does not match initialState",
          { expectedStateHash: normalizedHash, computedStateHash: computedHash },
        );
      }
      const command = createOnlineCommand({
        rulesetHash,
        commandId: nextCommandId(ONLINE_COMMAND_TYPE.CREATE_ROOM),
        type: ONLINE_COMMAND_TYPE.CREATE_ROOM,
        roomCode: normalizedCode,
        playerId,
        playerSide: normalizedSide,
        expectedRevision: 0,
        expectedStateHash: normalizedHash,
        payload: {
          requestedSide: normalizedSide,
          initialState: normalizedInitialState,
        },
      });
      const response = responseRoom(await transport.createRoom(command));
      const room = ensureRoomIdentity(await verifyRoomStateHash(response), normalizedCode);
      if (room.revision < 1 || room.stateHash !== normalizedHash) {
        throw new OnlineMultiplayerProtocolError("INVALID_CREATE_RESPONSE", "created room did not preserve initial state identity");
      }
      if (room.lastCommandId !== command.commandId) {
        throw new OnlineMultiplayerProtocolError("COMMAND_NOT_ACCEPTED", "created room did not accept the submitted commandId");
      }
      if (onlinePlayerSideForId(room, playerId) !== normalizedSide) {
        throw new OnlineMultiplayerProtocolError("SIDE_SELECTION_REJECTED", "created room did not assign requested side");
      }
      return syncRoom(room, ONLINE_OPERATION.CREATE);
    });
  }

  async function joinRoom({ roomCode, playerSide = ONLINE_PLAYER_SIDE.ALLIED } = {}) {
    return runOperation(ONLINE_OPERATION.JOIN, async () => {
      const normalizedCode = normalizeOnlineRoomCode(roomCode);
      const normalizedSide = validateOnlinePlayerSide(playerSide);
      const currentResponse = responseRoom(await transport.readRoom(normalizedCode));
      const current = ensureRoomIdentity(await verifyRoomStateHash(currentResponse), normalizedCode);
      const command = createOnlineCommand({
        rulesetHash,
        commandId: nextCommandId(ONLINE_COMMAND_TYPE.JOIN_ROOM),
        type: ONLINE_COMMAND_TYPE.JOIN_ROOM,
        roomCode: normalizedCode,
        playerId,
        playerSide: normalizedSide,
        expectedRevision: current.revision,
        expectedStateHash: current.stateHash,
        payload: { requestedSide: normalizedSide },
      });
      assertOnlineCommandAgainstRoom(command, current);
      const response = responseRoom(await transport.sendCommand(command));
      const room = validateAdvancedRoom(await verifyRoomStateHash(response), command);
      if (onlinePlayerSideForId(room, playerId) !== normalizedSide) {
        throw new OnlineMultiplayerProtocolError("SIDE_SELECTION_REJECTED", "joined room did not assign requested side");
      }
      return syncRoom(room, ONLINE_OPERATION.JOIN);
    });
  }

  async function selectSide(playerSide) {
    return runOperation(ONLINE_OPERATION.SELECT_SIDE, async () => {
      const normalizedSide = validateOnlinePlayerSide(playerSide);
      return sendCurrentRoomCommand(
        ONLINE_OPERATION.SELECT_SIDE,
        ONLINE_COMMAND_TYPE.SELECT_SIDE,
        { side: normalizedSide },
        { expectedSide: normalizedSide },
      );
    });
  }

  async function setReady(ready = true) {
    return runOperation(ONLINE_OPERATION.READY, async () => sendCurrentRoomCommand(
      ONLINE_OPERATION.READY,
      ONLINE_COMMAND_TYPE.SET_READY,
      { ready: Boolean(ready) },
    ));
  }

  async function submitGameAction(action, { executor = commandExecutor } = {}) {
    return runOperation(ONLINE_OPERATION.GAME_ACTION, async () => {
      if (!canIssueOnlineGameAction(state)) {
        throw new OnlineMultiplayerProtocolError(
          "ILLEGAL_SIDE",
          "the connected player is not the active side for this room",
        );
      }
      if (typeof executor !== "function") {
        throw new OnlineMultiplayerProtocolError(
          "COMMAND_EXECUTOR_REQUIRED",
          "a core-backed commandExecutor must be injected for game actions",
        );
      }
      const structuredAction = validateStructuredGameAction(action);
      const sourceRoom = state.room;
      const prepared = await executor({
        action: cloneOnlineJson(structuredAction),
        authoritativeState: cloneOnlineJson(sourceRoom.authoritativeState),
        room: sourceRoom,
        playerId,
        playerSide: state.playerSide,
      });
      if (!prepared || typeof prepared !== "object") {
        throw new OnlineMultiplayerProtocolError("INVALID_COMMAND_RESULT", "commandExecutor must return a structured result");
      }
      if (!Object.prototype.hasOwnProperty.call(prepared, "candidateActiveSide")) {
        throw new OnlineMultiplayerProtocolError(
          "INVALID_COMMAND_RESULT",
          "commandExecutor must return candidateActiveSide, using null only for a terminal state",
        );
      }
      const candidateState = cloneOnlineJson(prepared.candidateState, "candidateState");
      const computedCandidateStateHash = await computeValidatedStateHash(
        candidateState,
        "candidateStateHash",
      );
      if (prepared.candidateStateHash !== undefined && prepared.candidateStateHash !== null) {
        const reportedHash = validateOnlineStateHash(prepared.candidateStateHash, "candidateStateHash");
        if (reportedHash !== computedCandidateStateHash) {
          throw new OnlineMultiplayerProtocolError(
            "STATE_HASH_MISMATCH",
            "commandExecutor candidateStateHash does not match candidateState",
            { expectedStateHash: reportedHash, computedStateHash: computedCandidateStateHash },
          );
        }
      }
      const payload = {
        action: cloneOnlineJson(prepared.action || structuredAction),
        candidateState,
        candidateStateHash: computedCandidateStateHash,
        candidateActiveSide: prepared.candidateActiveSide ?? null,
        events: cloneOnlineJson(validateStructuredEvents(prepared.events || [])),
      };
      return sendCurrentRoomCommand(
        ONLINE_OPERATION.GAME_ACTION,
        ONLINE_COMMAND_TYPE.GAME_ACTION,
        payload,
      );
    });
  }

  async function reconnect({ roomCode = null } = {}) {
    return runOperation(ONLINE_OPERATION.RECONNECT, async () => {
      const targetRoomCode = state.room?.roomCode
        || (roomCode === null || roomCode === undefined ? null : normalizeOnlineRoomCode(roomCode));
      if (!targetRoomCode) {
        throw new OnlineMultiplayerProtocolError("NOT_IN_ROOM", "no room is available to reconnect");
      }
      const response = responseRoom(await transport.readRoom(targetRoomCode));
      const room = ensureRoomIdentity(
        await verifyRoomStateHash(response),
        targetRoomCode,
      );
      if (!onlinePlayerSideForId(room, playerId)) {
        throw new OnlineMultiplayerProtocolError("PLAYER_NOT_MEMBER", "player is no longer assigned in the room");
      }
      return syncRoom(room, ONLINE_OPERATION.RECONNECT);
    });
  }

  async function pollRoom() {
    ensureOpen();
    if (!state.room) throw new OnlineMultiplayerProtocolError("NOT_IN_ROOM", "no room is available to poll");
    if (operationInFlight) return state.room;
    if (pollInFlight) return pollInFlight;

    const sourceRoom = state.room;
    pollInFlight = (async () => {
      try {
        const response = responseRoom(await transport.readRoom(sourceRoom.roomCode));
        const room = ensureRoomIdentity(
          await verifyRoomStateHash(response),
          sourceRoom.roomCode,
        );
        if (!onlinePlayerSideForId(room, playerId)) {
          throw new OnlineMultiplayerProtocolError("PLAYER_NOT_MEMBER", "player is no longer assigned in the room");
        }
        if (room.revision < sourceRoom.revision) {
          throw new OnlineMultiplayerProtocolError(
            "STALE_RESPONSE",
            "polled room revision is older than the local room revision",
            { localRevision: sourceRoom.revision, remoteRevision: room.revision },
          );
        }
        if (room.revision === sourceRoom.revision) {
          if (!sameRoomRevisionIdentity(room, sourceRoom)) {
            throw new OnlineMultiplayerProtocolError(
              "REVISION_CONFLICT",
              "room data changed without advancing revision",
            );
          }
          if (state.connection === ONLINE_CONNECTION_STATE.CONNECTED) return state.room;
        }
        if (closed) return room;
        return syncRoom(room, ONLINE_OPERATION.RECONNECT);
      } catch (error) {
        const normalized = asClientError(error, ONLINE_OPERATION.RECONNECT);
        if (!closed) dispatch({ type: ONLINE_CLIENT_EVENT.CONNECTION_LOST, error: normalized });
        throw normalized;
      }
    })();
    try {
      return await pollInFlight;
    } finally {
      pollInFlight = null;
    }
  }

  async function leaveRoom() {
    return runOperation(ONLINE_OPERATION.LEAVE, async () => {
      const command = currentRoomCommand(ONLINE_COMMAND_TYPE.LEAVE_ROOM, { reason: "user_left" });
      const response = responseRoom(await transport.sendCommand(command));
      validateAdvancedRoom(await verifyRoomStateHash(response), command);
      dispatch({ type: ONLINE_CLIENT_EVENT.ROOM_LEFT });
      return state;
    });
  }

  function markDisconnected(error = { code: "TRANSPORT_DISCONNECTED", message: "transport disconnected" }) {
    ensureOpen();
    dispatch({ type: ONLINE_CLIENT_EVENT.CONNECTION_LOST, error });
    return state;
  }

  function clearError() {
    ensureOpen();
    dispatch({ type: ONLINE_CLIENT_EVENT.ERROR_CLEARED });
    return state;
  }

  function subscribe(listener, { emitCurrent = true } = {}) {
    ensureOpen();
    if (typeof listener !== "function") throw new TypeError("online room listener must be a function");
    listeners.add(listener);
    if (emitCurrent) listener(state, { type: "CURRENT_STATE" });
    return () => listeners.delete(listener);
  }

  function destroy() {
    closed = true;
    listeners.clear();
  }

  return Object.freeze({
    createRoom,
    joinRoom,
    selectSide,
    setReady,
    submitGameAction,
    reconnect,
    pollRoom,
    leaveRoom,
    markDisconnected,
    clearError,
    subscribe,
    destroy,
    getState: () => state,
  });
}

export function createOnlineRoomPoller({
  client,
  intervalMs = 2_000,
  setIntervalImpl = globalThis.setInterval?.bind(globalThis),
  clearIntervalImpl = globalThis.clearInterval?.bind(globalThis),
  onError = () => {},
} = {}) {
  if (!client || typeof client.pollRoom !== "function") {
    throw new TypeError("client.pollRoom must be a function");
  }
  if (!Number.isFinite(intervalMs) || intervalMs < 250) {
    throw new TypeError("intervalMs must be at least 250 milliseconds");
  }
  if (typeof setIntervalImpl !== "function" || typeof clearIntervalImpl !== "function") {
    throw new TypeError("poller timer functions must be provided");
  }
  if (typeof onError !== "function") throw new TypeError("onError must be a function");

  let timerId = null;
  let pollInFlight = null;

  async function pollNow() {
    if (pollInFlight) return pollInFlight;
    pollInFlight = Promise.resolve().then(() => client.pollRoom());
    try {
      return await pollInFlight;
    } finally {
      pollInFlight = null;
    }
  }

  function scheduledPoll() {
    return pollNow().catch((error) => {
      onError(error);
      return null;
    });
  }

  function start({ immediate = true } = {}) {
    if (timerId !== null) return false;
    timerId = setIntervalImpl(scheduledPoll, intervalMs);
    if (immediate) void scheduledPoll();
    return true;
  }

  function stop() {
    if (timerId === null) return false;
    clearIntervalImpl(timerId);
    timerId = null;
    return true;
  }

  return Object.freeze({
    start,
    stop,
    pollNow,
    isRunning: () => timerId !== null,
  });
}
