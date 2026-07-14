import { ONLINE_PLAYER_SIDE, ONLINE_ROOM_STATUS } from "../app/online-multiplayer-protocol.js";
import {
  ONLINE_CONNECTION_STATE,
  onlineRoomPlayerReady,
} from "../app/online-multiplayer-state.js";

const DEFAULT_LABELS = Object.freeze({
  title: "Online friend match",
  roomCode: "Room code",
  axis: "Axis",
  allied: "Allied",
  create: "Create room",
  join: "Join room",
  ready: "Ready",
  notReady: "Not ready",
  reconnect: "Reconnect",
  leave: "Leave room",
  offline: "Offline",
  connecting: "Connecting",
  connected: "Connected",
  disconnected: "Disconnected",
  error: "Online error",
  lobby: "Lobby",
  active: "Match active",
  finished: "Match finished",
  abandoned: "Room abandoned",
  expired: "Room expired",
  empty: "Open",
  occupied: "Occupied",
});

function makeElement(documentRef, tag, className, text = "") {
  const element = documentRef.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function roomStatusLabel(status, labels) {
  if (status === ONLINE_ROOM_STATUS.LOBBY) return labels.lobby;
  if (status === ONLINE_ROOM_STATUS.ACTIVE) return labels.active;
  if (status === ONLINE_ROOM_STATUS.FINISHED) return labels.finished;
  if (status === ONLINE_ROOM_STATUS.ABANDONED) return labels.abandoned;
  if (status === ONLINE_ROOM_STATUS.EXPIRED) return labels.expired;
  return status || labels.offline;
}

function connectionLabel(connection, labels) {
  if (connection === ONLINE_CONNECTION_STATE.CONNECTING || connection === ONLINE_CONNECTION_STATE.RECONNECTING) {
    return labels.connecting;
  }
  if (connection === ONLINE_CONNECTION_STATE.CONNECTED) return labels.connected;
  if (connection === ONLINE_CONNECTION_STATE.DISCONNECTED) return labels.disconnected;
  if (connection === ONLINE_CONNECTION_STATE.ERROR) return labels.error;
  return labels.offline;
}

export function createOnlineMultiplayerPanel({
  root,
  client,
  getInitialRoomState = null,
  labels: labelOverrides = {},
  documentRef = globalThis.document,
  onError = () => {},
} = {}) {
  if (!root || typeof root.replaceChildren !== "function") {
    throw new TypeError("online multiplayer panel root must be a DOM element");
  }
  if (!client || typeof client.subscribe !== "function") {
    throw new TypeError("online multiplayer panel requires an online room client");
  }
  if (!documentRef || typeof documentRef.createElement !== "function") {
    throw new TypeError("online multiplayer panel requires a document");
  }
  if (getInitialRoomState !== null && typeof getInitialRoomState !== "function") {
    throw new TypeError("getInitialRoomState must be a function when provided");
  }

  const labels = Object.freeze({ ...DEFAULT_LABELS, ...labelOverrides });
  let preferredSide = ONLINE_PLAYER_SIDE.AXIS;
  let destroyed = false;

  const panel = makeElement(documentRef, "section", "online-multiplayer-panel");
  panel.dataset.onlineState = "offline";
  panel.setAttribute("aria-label", labels.title);

  const heading = makeElement(documentRef, "h2", "online-multiplayer-title", labels.title);
  const status = makeElement(documentRef, "p", "online-multiplayer-status");
  status.setAttribute("role", "status");
  const roomDetails = makeElement(documentRef, "p", "online-multiplayer-room-details");
  const error = makeElement(documentRef, "p", "online-multiplayer-error");
  error.setAttribute("role", "alert");

  const roomRow = makeElement(documentRef, "div", "online-multiplayer-room-row");
  const roomLabel = makeElement(documentRef, "label", "online-multiplayer-room-label", labels.roomCode);
  const roomInput = makeElement(documentRef, "input", "online-multiplayer-room-input");
  roomInput.type = "text";
  roomInput.autocomplete = "off";
  roomInput.maxLength = 8;
  roomInput.inputMode = "text";
  roomInput.setAttribute("aria-label", labels.roomCode);
  roomLabel.append(roomInput);
  roomRow.append(roomLabel);

  const sideRow = makeElement(documentRef, "div", "online-multiplayer-side-row");
  const axisButton = makeElement(documentRef, "button", "online-multiplayer-side", labels.axis);
  const alliedButton = makeElement(documentRef, "button", "online-multiplayer-side", labels.allied);
  axisButton.type = "button";
  alliedButton.type = "button";
  axisButton.dataset.side = ONLINE_PLAYER_SIDE.AXIS;
  alliedButton.dataset.side = ONLINE_PLAYER_SIDE.ALLIED;
  sideRow.append(axisButton, alliedButton);

  const actionRow = makeElement(documentRef, "div", "online-multiplayer-actions");
  const createButton = makeElement(documentRef, "button", "online-multiplayer-create", labels.create);
  const joinButton = makeElement(documentRef, "button", "online-multiplayer-join", labels.join);
  const readyButton = makeElement(documentRef, "button", "online-multiplayer-ready", labels.ready);
  const reconnectButton = makeElement(documentRef, "button", "online-multiplayer-reconnect", labels.reconnect);
  const leaveButton = makeElement(documentRef, "button", "online-multiplayer-leave", labels.leave);
  for (const button of [createButton, joinButton, readyButton, reconnectButton, leaveButton]) button.type = "button";
  actionRow.append(createButton, joinButton, readyButton, reconnectButton, leaveButton);

  const assignment = makeElement(documentRef, "div", "online-multiplayer-assignments");
  const axisAssignment = makeElement(documentRef, "span", "online-multiplayer-assignment");
  const alliedAssignment = makeElement(documentRef, "span", "online-multiplayer-assignment");
  assignment.append(axisAssignment, alliedAssignment);

  panel.append(heading, status, roomDetails, roomRow, sideRow, assignment, actionRow, error);
  root.replaceChildren(panel);

  async function perform(operation) {
    try {
      await operation();
    } catch (operationError) {
      onError(operationError);
    }
  }

  async function chooseSide(side) {
    preferredSide = side;
    const state = client.getState();
    if (state.room && state.connection === ONLINE_CONNECTION_STATE.CONNECTED && state.playerSide !== side) {
      await client.selectSide(side);
    }
    render(client.getState());
  }

  axisButton.addEventListener("click", () => perform(() => chooseSide(ONLINE_PLAYER_SIDE.AXIS)));
  alliedButton.addEventListener("click", () => perform(() => chooseSide(ONLINE_PLAYER_SIDE.ALLIED)));
  createButton.addEventListener("click", () => perform(async () => {
    if (!getInitialRoomState) throw new Error("getInitialRoomState is required to create an online room");
    const initial = await getInitialRoomState();
    await client.createRoom({
      roomCode: roomInput.value.trim() || null,
      playerSide: preferredSide,
      initialState: initial?.state,
      stateHash: initial?.stateHash,
    });
  }));
  joinButton.addEventListener("click", () => perform(() => client.joinRoom({
    roomCode: roomInput.value,
    playerSide: preferredSide,
  })));
  readyButton.addEventListener("click", () => perform(() => client.setReady(!onlineRoomPlayerReady(client.getState()))));
  reconnectButton.addEventListener("click", () => perform(() => client.reconnect()));
  leaveButton.addEventListener("click", () => perform(() => client.leaveRoom()));

  function render(state) {
    if (destroyed) return;
    const room = state.room;
    const busy = Boolean(state.operation);
    const connected = state.connection === ONLINE_CONNECTION_STATE.CONNECTED;
    const inLobby = room?.status === ONLINE_ROOM_STATUS.LOBBY;
    panel.dataset.onlineState = state.connection;
    panel.dataset.roomStatus = room?.status || "none";
    status.textContent = room
      ? `${connectionLabel(state.connection, labels)} · ${roomStatusLabel(room.status, labels)}`
      : connectionLabel(state.connection, labels);
    roomDetails.textContent = room
      ? `${room.roomCode} · revision ${room.revision} · ${room.stateHash}`
      : "";
    if (room && roomInput.value !== room.roomCode) roomInput.value = room.roomCode;
    roomInput.disabled = Boolean(room) || busy;

    preferredSide = state.playerSide || preferredSide;
    axisButton.setAttribute("aria-pressed", String(preferredSide === ONLINE_PLAYER_SIDE.AXIS));
    alliedButton.setAttribute("aria-pressed", String(preferredSide === ONLINE_PLAYER_SIDE.ALLIED));
    axisButton.disabled = busy || Boolean(room?.players.axis && room.players.axis !== state.playerId) || Boolean(room && !inLobby);
    alliedButton.disabled = busy || Boolean(room?.players.allied && room.players.allied !== state.playerId) || Boolean(room && !inLobby);

    axisAssignment.textContent = `${labels.axis}: ${room?.players.axis ? labels.occupied : labels.empty}`;
    alliedAssignment.textContent = `${labels.allied}: ${room?.players.allied ? labels.occupied : labels.empty}`;
    createButton.hidden = Boolean(room);
    joinButton.hidden = Boolean(room);
    createButton.disabled = busy || !getInitialRoomState;
    joinButton.disabled = busy;
    readyButton.hidden = !room || !inLobby;
    readyButton.disabled = busy || !connected || !state.playerSide;
    readyButton.textContent = onlineRoomPlayerReady(state) ? labels.notReady : labels.ready;
    reconnectButton.hidden = !room || ![
      ONLINE_CONNECTION_STATE.DISCONNECTED,
      ONLINE_CONNECTION_STATE.ERROR,
    ].includes(state.connection);
    reconnectButton.disabled = busy;
    leaveButton.hidden = !room;
    leaveButton.disabled = busy;
    error.textContent = state.error ? `${state.error.code}: ${state.error.message}` : "";
    error.hidden = !state.error;
  }

  const unsubscribe = client.subscribe((state) => render(state));

  return Object.freeze({
    element: panel,
    render: () => render(client.getState()),
    destroy() {
      if (destroyed) return;
      destroyed = true;
      unsubscribe();
      if (panel.parentNode === root) root.replaceChildren();
    },
  });
}
