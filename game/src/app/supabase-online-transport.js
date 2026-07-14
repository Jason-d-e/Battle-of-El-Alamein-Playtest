import {
  ONLINE_COMMAND_TYPE,
  OnlineMultiplayerProtocolError,
  cloneOnlineJson,
  normalizeOnlineRoomCode,
  validateOnlineCommand,
  validateOnlineRoomSnapshot,
} from "./online-multiplayer-protocol.js";
import { createOnlineRoomClient, createOnlineRoomPoller } from "./online-room-client.js";

const AUTH_REFRESH_MARGIN_MS = 60_000;
const BUSINESS_ERROR_CODE = /^[A-Z][A-Z0-9_]{2,63}$/;
const RECOVERABLE_PROTOCOL_ERRORS = new Set([
  "STALE_REVISION",
  "STATE_HASH_MISMATCH",
  "TRANSPORT_DISCONNECTED",
]);

function fail(code, message, details = null) {
  throw new OnlineMultiplayerProtocolError(code, message, details);
}

function requiredString(value, label, code) {
  if (typeof value !== "string" || value.trim() === "") fail(code, `${label} must be a non-empty string`);
  return value.trim();
}

function normalizeProjectUrl(value) {
  const raw = requiredString(value, "projectUrl", "INVALID_SUPABASE_URL");
  let url;
  try {
    url = new URL(raw);
  } catch {
    fail("INVALID_SUPABASE_URL", "projectUrl must be an absolute URL");
  }
  const localHost = url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "::1";
  if (url.protocol !== "https:" && !(localHost && url.protocol === "http:")) {
    fail("INVALID_SUPABASE_URL", "projectUrl must use HTTPS, except for localhost development");
  }
  if (url.username || url.password || url.search || url.hash || (url.pathname !== "/" && url.pathname !== "")) {
    fail("INVALID_SUPABASE_URL", "projectUrl must contain only the Supabase project origin");
  }
  return url.origin;
}

function decodeLegacyKeyRole(key) {
  if (typeof globalThis.atob !== "function") return null;
  const segments = key.split(".");
  if (segments.length !== 3) return null;
  try {
    const normalized = segments[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
    return JSON.parse(globalThis.atob(padded))?.role || null;
  } catch {
    return null;
  }
}

function validatePublishableKey(value) {
  const key = requiredString(value, "publishableKey", "INVALID_SUPABASE_KEY");
  if (key.startsWith("sb_secret_") || decodeLegacyKeyRole(key) === "service_role") {
    fail("SECRET_KEY_REJECTED", "Supabase secret and service-role keys must never be used in browser code");
  }
  return key;
}

function validateFetch(fetchImpl) {
  if (typeof fetchImpl !== "function") throw new TypeError("fetchImpl must be a function");
  return fetchImpl;
}

function validateSessionStore(sessionStore) {
  if (sessionStore === null || sessionStore === undefined) return null;
  if (typeof sessionStore !== "object") throw new TypeError("sessionStore must be an object");
  for (const method of ["load", "save", "clear"]) {
    if (typeof sessionStore[method] !== "function") {
      throw new TypeError(`sessionStore.${method} must be a function`);
    }
  }
  return sessionStore;
}

export function createSupabaseWebStorageSessionStore({
  storage,
  key = "el-alamein-supabase-session-v1",
} = {}) {
  if (!storage || typeof storage !== "object") throw new TypeError("storage must be a Web Storage object");
  for (const method of ["getItem", "setItem", "removeItem"]) {
    if (typeof storage[method] !== "function") throw new TypeError(`storage.${method} must be a function`);
  }
  const storageKey = requiredString(key, "key", "INVALID_SESSION_STORAGE_KEY");

  function storageFailure(operation, cause) {
    return new OnlineMultiplayerProtocolError(
      "SESSION_STORAGE_ERROR",
      `failed to ${operation} Supabase session storage`,
      { cause: typeof cause?.message === "string" ? cause.message : null },
    );
  }

  return Object.freeze({
    async load() {
      let raw;
      try {
        raw = storage.getItem(storageKey);
      } catch (error) {
        throw storageFailure("read", error);
      }
      if (raw === null) return null;
      try {
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("invalid session object");
        return parsed;
      } catch (error) {
        throw storageFailure("parse", error);
      }
    },
    async save(session) {
      try {
        storage.setItem(storageKey, JSON.stringify(cloneOnlineJson(session, "session")));
      } catch (error) {
        throw storageFailure("write", error);
      }
    },
    async clear() {
      try {
        storage.removeItem(storageKey);
      } catch (error) {
        throw storageFailure("clear", error);
      }
    },
  });
}

function normalizeSession(payload, nowMs) {
  const source = payload?.session && typeof payload.session === "object" ? payload.session : payload;
  const user = payload?.user || source?.user;
  const accessToken = requiredString(source?.access_token ?? source?.accessToken, "accessToken", "INVALID_AUTH_SESSION");
  const refreshToken = requiredString(
    source?.refresh_token ?? source?.refreshToken,
    "refreshToken",
    "INVALID_AUTH_SESSION",
  );
  const userId = requiredString(user?.id ?? source?.userId, "user.id", "INVALID_AUTH_SESSION");
  const rawExpiresAt = source?.expires_at ?? source?.expiresAt;
  const expiresIn = Number(source?.expires_in ?? source?.expiresIn);
  const expiresAt = Number.isFinite(Number(rawExpiresAt))
    ? Number(rawExpiresAt)
    : Number.isFinite(expiresIn) && expiresIn > 0
      ? Math.floor(nowMs / 1000) + expiresIn
      : NaN;
  if (!Number.isFinite(expiresAt) || expiresAt < 0) {
    fail("INVALID_AUTH_SESSION", "Auth session must contain a valid expiration time");
  }
  return Object.freeze({ accessToken, refreshToken, userId, expiresAt });
}

function publicSession(session) {
  return Object.freeze({ playerId: session.userId, expiresAt: session.expiresAt });
}

function parseResponseBody(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    fail("INVALID_SUPABASE_RESPONSE", "Supabase returned a non-JSON response");
  }
}

function responseError(response, payload) {
  const payloadMessage = typeof payload?.message === "string"
    ? payload.message
    : typeof payload?.msg === "string"
      ? payload.msg
      : null;
  const code = payloadMessage && BUSINESS_ERROR_CODE.test(payloadMessage)
    ? payloadMessage
    : typeof (payload?.error_code ?? payload?.code) === "string"
        && BUSINESS_ERROR_CODE.test(payload.error_code ?? payload.code)
      ? payload.error_code ?? payload.code
      : `SUPABASE_HTTP_${response.status}`;
  const message = typeof payload?.details === "string" && payload.details
    ? payload.details
    : typeof payload?.error_description === "string" && payload.error_description
      ? payload.error_description
      : payloadMessage || `Supabase request failed with HTTP ${response.status}`;
  const error = new OnlineMultiplayerProtocolError(code, message, {
    httpStatus: response.status,
    hint: typeof payload?.hint === "string" ? payload.hint : null,
  });
  error.recoverable = RECOVERABLE_PROTOCOL_ERRORS.has(code)
    || response.status >= 500
    || response.status === 408
    || response.status === 429;
  return error;
}

function normalizeRoomResponse(payload) {
  const room = payload && typeof payload === "object" && Object.prototype.hasOwnProperty.call(payload, "room")
    ? payload.room
    : payload;
  return Object.freeze({ room: validateOnlineRoomSnapshot(room) });
}

export function createSupabaseOnlineTransport({
  projectUrl: projectUrlInput,
  publishableKey: publishableKeyInput,
  fetchImpl: fetchInput = globalThis.fetch,
  now = () => Date.now(),
  captchaToken = null,
  initialSession = null,
  sessionStore: sessionStoreInput = null,
} = {}) {
  const projectUrl = normalizeProjectUrl(projectUrlInput);
  const publishableKey = validatePublishableKey(publishableKeyInput);
  const fetchImpl = validateFetch(fetchInput);
  const sessionStore = validateSessionStore(sessionStoreInput);
  if (typeof now !== "function") throw new TypeError("now must be a function");
  if (captchaToken !== null && captchaToken !== undefined && typeof captchaToken !== "string") {
    throw new TypeError("captchaToken must be a string when provided");
  }

  let session = initialSession ? normalizeSession(initialSession, now()) : null;
  let sessionLoaded = initialSession !== null;
  let authInFlight = null;

  async function request(path, { body, accessToken = null } = {}) {
    const headers = {
      apikey: publishableKey,
      Authorization: `Bearer ${accessToken || publishableKey}`,
      "Content-Type": "application/json",
    };
    let response;
    try {
      response = await fetchImpl(`${projectUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body || {}),
      });
    } catch (cause) {
      const error = new OnlineMultiplayerProtocolError(
        "TRANSPORT_DISCONNECTED",
        typeof cause?.message === "string" ? cause.message : "Supabase request failed",
      );
      error.recoverable = true;
      throw error;
    }
    const payload = parseResponseBody(await response.text());
    return { response, payload };
  }

  async function saveSession(nextSession) {
    session = nextSession;
    if (sessionStore) await sessionStore.save(cloneOnlineJson(nextSession, "session"));
    return session;
  }

  async function clearSession() {
    session = null;
    sessionLoaded = true;
    if (sessionStore) await sessionStore.clear();
  }

  async function loadSession() {
    if (sessionLoaded) return session;
    sessionLoaded = true;
    const stored = sessionStore ? await sessionStore.load() : null;
    session = stored ? normalizeSession(stored, now()) : null;
    return session;
  }

  async function signInAnonymously() {
    const security = {};
    if (captchaToken) security.captcha_token = captchaToken;
    const { response, payload } = await request("/auth/v1/signup", {
      body: {
        data: {},
        gotrue_meta_security: security,
      },
    });
    if (!response.ok) throw responseError(response, payload);
    return saveSession(normalizeSession(payload, now()));
  }

  async function refreshSession() {
    if (!session?.refreshToken) {
      await clearSession();
      return signInAnonymously();
    }
    const { response, payload } = await request("/auth/v1/token?grant_type=refresh_token", {
      body: { refresh_token: session.refreshToken },
    });
    if (!response.ok) {
      await clearSession();
      throw responseError(response, payload);
    }
    return saveSession(normalizeSession(payload, now()));
  }

  async function ensureSession({ forceRefresh = false } = {}) {
    if (authInFlight) return authInFlight;
    authInFlight = (async () => {
      await loadSession();
      if (!session) return signInAnonymously();
      if (forceRefresh || session.expiresAt * 1000 <= now() + AUTH_REFRESH_MARGIN_MS) {
        return refreshSession();
      }
      return session;
    })();
    try {
      return await authInFlight;
    } finally {
      authInFlight = null;
    }
  }

  async function invokeRpc(functionName, args, { retryAuth = true } = {}) {
    const activeSession = await ensureSession();
    const result = await request(`/rest/v1/rpc/${functionName}`, {
      accessToken: activeSession.accessToken,
      body: args,
    });
    if (result.response.status === 401 && retryAuth) {
      const refreshed = await ensureSession({ forceRefresh: true });
      return invokeRpcWithSession(functionName, args, refreshed);
    }
    if (!result.response.ok) throw responseError(result.response, result.payload);
    return result.payload;
  }

  async function invokeRpcWithSession(functionName, args, activeSession) {
    const result = await request(`/rest/v1/rpc/${functionName}`, {
      accessToken: activeSession.accessToken,
      body: args,
    });
    if (!result.response.ok) throw responseError(result.response, result.payload);
    return result.payload;
  }

  async function authenticate() {
    return publicSession(await ensureSession());
  }

  async function assertCommandIdentity(command) {
    const activeSession = await ensureSession();
    if (command.playerId !== activeSession.userId) {
      fail("PLAYER_ID_MISMATCH", "command.playerId must equal the authenticated Supabase user id");
    }
  }

  async function createRoom(commandInput) {
    const command = validateOnlineCommand(commandInput);
    if (command.type !== ONLINE_COMMAND_TYPE.CREATE_ROOM) {
      fail("INVALID_CREATE_COMMAND", "createRoom requires a CREATE_ROOM command");
    }
    await assertCommandIdentity(command);
    return normalizeRoomResponse(await invokeRpc("el_alamein_online_create_room", {
      p_command: cloneOnlineJson(command),
    }));
  }

  async function readRoom(roomCodeInput) {
    const roomCode = normalizeOnlineRoomCode(roomCodeInput);
    return normalizeRoomResponse(await invokeRpc("el_alamein_online_read_room", {
      p_room_code: roomCode,
    }));
  }

  async function sendCommand(commandInput) {
    const command = validateOnlineCommand(commandInput);
    if (command.type === ONLINE_COMMAND_TYPE.CREATE_ROOM) {
      fail("INVALID_COMMAND", "CREATE_ROOM must use transport.createRoom");
    }
    await assertCommandIdentity(command);
    return normalizeRoomResponse(await invokeRpc("el_alamein_online_apply_command", {
      p_command: cloneOnlineJson(command),
    }));
  }

  return Object.freeze({
    authenticate,
    createRoom,
    readRoom,
    sendCommand,
    clearSession,
  });
}

export async function createSupabaseOnlineRoomRuntime({
  enabled = false,
  projectUrl,
  publishableKey,
  fetchImpl = globalThis.fetch,
  now = () => Date.now(),
  captchaToken = null,
  initialSession = null,
  sessionStore = null,
  rulesetHash,
  commandExecutor = null,
  hashState,
  createCommandId,
  createRoomCode,
  onListenerError = () => {},
  pollIntervalMs = 2_000,
  setIntervalImpl = globalThis.setInterval?.bind(globalThis),
  clearIntervalImpl = globalThis.clearInterval?.bind(globalThis),
  onPollError = () => {},
} = {}) {
  if (typeof enabled !== "boolean") throw new TypeError("enabled must be a boolean");
  if (!enabled) {
    return Object.freeze({
      enabled: false,
      playerId: null,
      expiresAt: null,
      transport: null,
      client: null,
      poller: null,
      startPolling: () => false,
      stopPolling: () => false,
      destroy: () => false,
    });
  }

  const normalizedRulesetHash = requiredString(rulesetHash, "rulesetHash", "INVALID_RULESET_HASH");
  if (commandExecutor !== null && typeof commandExecutor !== "function") {
    throw new TypeError("commandExecutor must be a function when provided");
  }
  for (const [label, value] of [
    ["hashState", hashState],
    ["createCommandId", createCommandId],
    ["createRoomCode", createRoomCode],
  ]) {
    if (value !== undefined && typeof value !== "function") {
      throw new TypeError(`${label} must be a function when provided`);
    }
  }
  if (typeof onListenerError !== "function") throw new TypeError("onListenerError must be a function");
  if (!Number.isFinite(pollIntervalMs) || pollIntervalMs < 250) {
    throw new TypeError("pollIntervalMs must be at least 250 milliseconds");
  }
  if (typeof setIntervalImpl !== "function" || typeof clearIntervalImpl !== "function") {
    throw new TypeError("poller timer functions must be provided");
  }
  if (typeof onPollError !== "function") throw new TypeError("onPollError must be a function");

  const transport = createSupabaseOnlineTransport({
    projectUrl,
    publishableKey,
    fetchImpl,
    now,
    captchaToken,
    initialSession,
    sessionStore,
  });
  const identity = await transport.authenticate();
  const client = createOnlineRoomClient({
    transport,
    playerId: identity.playerId,
    rulesetHash: normalizedRulesetHash,
    commandExecutor,
    hashState,
    createCommandId,
    createRoomCode,
    onListenerError,
  });
  const poller = createOnlineRoomPoller({
    client,
    intervalMs: pollIntervalMs,
    setIntervalImpl,
    clearIntervalImpl,
    onError: onPollError,
  });
  let destroyed = false;

  function ensureActive() {
    if (destroyed) {
      fail("RUNTIME_DESTROYED", "Supabase online room runtime is destroyed");
    }
  }

  function startPolling(options) {
    ensureActive();
    return poller.start(options);
  }

  function stopPolling() {
    if (destroyed) return false;
    return poller.stop();
  }

  function destroy() {
    if (destroyed) return false;
    destroyed = true;
    poller.stop();
    client.destroy();
    return true;
  }

  return Object.freeze({
    enabled: true,
    playerId: identity.playerId,
    expiresAt: identity.expiresAt,
    transport,
    client,
    poller,
    startPolling,
    stopPolling,
    destroy,
  });
}
