import { handleAlphaRuntimeMessage } from "./ai-alpha-runtime.js";

self.addEventListener("message", (event) => {
  self.postMessage(handleAlphaRuntimeMessage(event.data));
});
