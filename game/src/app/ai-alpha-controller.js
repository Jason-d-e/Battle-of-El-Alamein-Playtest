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

  async function chooseAction(payload = {}, options = {}) {
    const response = await requestWorker(ALPHA_RUNTIME_MESSAGE.CHOOSE_ACTION, payload, options);
    if (response?.ok) {
      lastMode = "worker";
      lastAnalysis = response.analysis || null;
      return response.action || null;
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
    lastMode = "direct";
    if (!directFallback) return null;
    lastAnalysis = directRuntime.analyzeAlphaPosition?.(payload) || null;
    return lastAnalysis;
  }

  function getMode() {
    if (worker && !workerFailed) return lastMode;
    return directFallback ? "direct" : "disabled";
  }

  function dispose() {
    worker?.terminate?.();
    worker = null;
    workerFailed = true;
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
      const cleanup = () => {
        settled = true;
        clearTimeout(timer);
        target.removeEventListener?.("message", onMessage);
        target.removeEventListener?.("error", onError);
        target.removeEventListener?.("messageerror", onError);
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
      return worker;
    } catch {
      workerFailed = true;
      worker = null;
      return null;
    }
  }

  return {
    analyze,
    chooseAction,
    dispose,
    getLastAnalysis,
    getMode,
  };
}

export function workerPayload(payload = {}) {
  const { board, ...rest } = payload || {};
  return rest;
}
