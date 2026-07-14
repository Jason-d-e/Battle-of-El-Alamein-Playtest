import {
  NODE_KIND,
  applyChanceOutcome,
  readEnvironmentNode,
} from "./environment-contract.js";

export function evaluateChanceExpectation(adapter, state, evaluate, options = {}) {
  if (typeof evaluate !== "function") {
    throw new TypeError("evaluate must be a function");
  }

  const node = readEnvironmentNode(adapter, state, options);
  if (node.nodeKind !== NODE_KIND.CHANCE) {
    throw new TypeError("evaluateChanceExpectation requires a chance node");
  }

  let expectedValue = 0;
  let compensation = 0;
  let evaluatedOutcomeCount = 0;
  for (const entry of node.chanceOutcomes) {
    if (entry.probability === 0) continue;
    const nextState = applyChanceOutcome(adapter, state, entry.key, options);
    const value = evaluate(nextState, entry);
    if (!Number.isFinite(value)) {
      throw new TypeError("chance evaluator must return a finite number");
    }
    const weightedValue = entry.probability * value;
    const compensatedValue = weightedValue - compensation;
    const nextExpectedValue = expectedValue + compensatedValue;
    compensation = (nextExpectedValue - expectedValue) - compensatedValue;
    expectedValue = nextExpectedValue;
    evaluatedOutcomeCount += 1;
  }

  return Object.freeze({ expectedValue, evaluatedOutcomeCount });
}
