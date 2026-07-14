import { handleAlphaRuntimeMessage } from "./ai-alpha-runtime.js?v=20260713-generic-runtime-1";

self.addEventListener("message", (event) => {
  self.postMessage(handleAlphaRuntimeMessage(event.data));
});
