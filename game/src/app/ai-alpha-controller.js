import {
  ALPHA_RUNTIME_MESSAGE,
  analyzeAlphaPosition,
  chooseAlphaRuntimeDecision,
  chooseAlphaRuntimeAction,
} from "./ai-alpha-runtime.js";

export function createAlphaAiClient({
  workerFactory = null,
  runtime = null,
  directFallback = true,
  timeoutMs = 220,
} = {}) {
  const directRuntime = runtime || {
    analyzeAlphaPosition,
    chooseAlphaRuntimeDecision,
    chooseAlphaRuntimeAction,
  };
  let worker = null;
  let workerFailed = false;
  let nextRequestId = 1;
  let lastMode = "direct";
  let lastAnalysis = null;
  const pendingRequests = new Set();

  async function chooseAction(payload = {}, options = {}) {
    const response = await requestWorker(ALPHA_RUNTIME_MESSAGE.CHOOSE_ACTION, payload, options);
    if (response?.ok) {
      lastMode = "worker";
      lastAnalysis = response.analysis || null;
      return response.action || null;
    }
    if (response?.cancelled) {
      lastMode = "cancelled";
      return null;
    }
    lastMode = "direct";
    if (!directFallback) return null;
    if (directRuntime.chooseAlphaRuntimeDecision) {
      const decision = directRuntime.chooseAlphaRuntimeDecision(payload);
      lastAnalysis = decision?.analysis || null;
      return decision?.action || null;
    }
    lastAnalysis = null;
    return directRuntime.chooseAlphaRuntimeAction?.(payload) || null;
  }

  async function analyze(payload = {}, options = {}) {
    const response = await requestWorker(ALPHA_RUNTIME_MESSAGE.ANALYZE, payload, options);
    if (response?.ok) {
      lastMode = "worker";
      lastAnalysis = response.analysis || null;
      return response.analysis || null;
    }
    if (response?.cancelled) {
      lastMode = "cancelled";
      return null;
    }
    lastMode = "direct";
    if (!directFallback) return null;
    lastAnalysis = directRuntime.analyzeAlphaPosition?.(payload) || null;
    return lastAnalysis;
  }

  function getMode() {
    if (worker && !workerFailed) return lastMode;
    if (workerFactory && !workerFailed) return "worker-ready";
    return directFallback ? "direct" : "disabled";
  }

  function getLastRequestStatus() {
    return lastMode;
  }

  function dispose() {
    cancelPending("disposed", { disable: true });
  }

  function cancelPending(reason = "cancelled", options = {}) {
    const target = worker;
    worker = null;
    workerFailed = Boolean(options.disable);
    target?.terminate?.();
    const pending = [...pendingRequests];
    for (const cancel of pending) cancel(reason);
    lastMode = reason;
    return pending.length;
  }

  function getLastAnalysis() {
    return lastAnalysis;
  }

  function requestWorker(type, payload, options) {
    const target = ensureWorker();
    if (!target) return Promise.resolve(null);
    const id = `alpha-${nextRequestId++}`;
    const message = {
      id,
      type,
      payload: workerPayload(payload),
    };
    const waitMs = Math.max(1, Number(options.timeoutMs ?? timeoutMs));
    return new Promise((resolve) => {
      let settled = false;
      let cancelRequest = null;
      const cleanup = () => {
        settled = true;
        clearTimeout(timer);
        target.removeEventListener?.("message", onMessage);
        target.removeEventListener?.("error", onError);
        target.removeEventListener?.("messageerror", onError);
        if (cancelRequest) pendingRequests.delete(cancelRequest);
      };
      const settle = (value) => {
        if (settled) return;
        cleanup();
        resolve(value);
      };
      const onMessage = (event) => {
        const data = event?.data || null;
        if (data?.id !== id) return;
        settle(data);
      };
      cancelRequest = (reason = "cancelled") => settle({
        id,
        type: `${type}_RESULT`,
        ok: false,
        cancelled: true,
        reason,
      });
      const failWorker = () => {
        workerFailed = true;
        if (worker === target) worker = null;
        target.terminate?.();
      };
      const onError = () => {
        failWorker();
        settle(null);
      };
      const timer = setTimeout(() => {
        failWorker();
        settle(null);
      }, waitMs);
      target.addEventListener?.("message", onMessage);
      target.addEventListener?.("error", onError);
      target.addEventListener?.("messageerror", onError);
      pendingRequests.add(cancelRequest);
      try {
        target.postMessage(message);
      } catch {
        failWorker();
        settle(null);
      }
    });
  }

  function ensureWorker() {
    if (workerFailed || !workerFactory) return null;
    if (worker) return worker;
    try {
      worker = workerFactory();
      if (!worker) {
        workerFailed = true;
        worker = null;
        return null;
      }
      return worker;
    } catch {
      workerFailed = true;
      worker = null;
      return null;
    }
  }

  return {
    analyze,
    cancelPending,
    chooseAction,
    dispose,
    getLastAnalysis,
    getLastRequestStatus,
    getMode,
  };
}

export function workerPayload(payload = {}) {
  const { board, ...rest } = payload || {};
  return rest;
}
