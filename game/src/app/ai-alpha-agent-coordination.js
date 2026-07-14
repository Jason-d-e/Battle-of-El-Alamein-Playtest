export const COORDINATION_MANIFEST_SCHEMA = "zizi-el-alamein-alpha-agent-coordination-v1";
export const COORDINATION_SCOPE = "collaboration-boundary-check";

const TASK_FIELDS = ["id", "owner", "writeGlobs", "readOnlyGlobs", "status"];

/**
 * Validate and normalize a collaboration manifest without consulting Git or the filesystem.
 */
export function validateCoordinationManifest(input) {
  const errors = [];
  if (!isPlainObject(input)) {
    return invalidManifest([error("manifest", "manifest_must_be_object")]);
  }

  if (input.schema !== undefined && input.schema !== COORDINATION_MANIFEST_SCHEMA) {
    errors.push(error("schema", "unsupported_manifest_schema"));
  }
  if (!Array.isArray(input.tasks)) {
    errors.push(error("tasks", "tasks_must_be_array"));
    return invalidManifest(errors);
  }

  const tasks = [];
  const seenIds = new Set();
  input.tasks.forEach((task, taskIndex) => {
    const taskResult = normalizeTask(task, taskIndex);
    errors.push(...taskResult.errors);
    if (taskResult.task) {
      if (seenIds.has(taskResult.task.id)) {
        errors.push(error(`tasks[${taskIndex}].id`, "duplicate_task_id", taskResult.task.id));
      } else {
        seenIds.add(taskResult.task.id);
      }
      tasks.push(taskResult.task);
    }
  });

  if (errors.length) return invalidManifest(errors);
  return {
    ok: true,
    reason: null,
    errors: [],
    manifest: {
      schema: COORDINATION_MANIFEST_SCHEMA,
      tasks,
    },
  };
}

/**
 * Check every pair of tasks for an obvious write-glob overlap.
 */
export function checkCoordinationManifest(input) {
  const validation = validateCoordinationManifest(input);
  if (!validation.ok) {
    return {
      schema: COORDINATION_MANIFEST_SCHEMA,
      scope: COORDINATION_SCOPE,
      ok: false,
      valid: false,
      reason: "invalid_manifest",
      errors: validation.errors,
      conflicts: [],
      git: coordinationScopeNotice(),
    };
  }

  const tasks = [...validation.manifest.tasks].sort(compareTasks);
  const conflicts = [];
  for (let leftIndex = 0; leftIndex < tasks.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < tasks.length; rightIndex += 1) {
      conflicts.push(...checkWriteGlobConflicts(tasks[leftIndex], tasks[rightIndex]).conflicts);
    }
  }
  conflicts.sort(compareConflicts);

  return {
    schema: COORDINATION_MANIFEST_SCHEMA,
    scope: COORDINATION_SCOPE,
    ok: conflicts.length === 0,
    valid: true,
    reason: conflicts.length ? "write_glob_overlap" : null,
    errors: [],
    taskIds: tasks.map((task) => task.id),
    conflicts,
    git: coordinationScopeNotice(),
  };
}

/**
 * Compare two task objects. Only write globs participate in this check.
 */
export function checkWriteGlobConflicts(leftTask, rightTask) {
  const leftResult = normalizeTask(leftTask, 0);
  const rightResult = normalizeTask(rightTask, 1);
  const errors = [...leftResult.errors, ...rightResult.errors];
  if (errors.length) {
    return {
      ok: false,
      reason: "invalid_task",
      errors,
      conflicts: [],
    };
  }

  const [left, right] = [leftResult.task, rightResult.task].sort(compareTasks);
  const conflicts = [];
  for (const leftGlob of left.writeGlobs) {
    for (const rightGlob of right.writeGlobs) {
      const overlappingPath = findGlobOverlap(leftGlob, rightGlob);
      if (!overlappingPath) continue;
      conflicts.push({
        reason: "write_globs_overlap",
        taskAId: left.id,
        taskBId: right.id,
        ownerA: left.owner,
        ownerB: right.owner,
        writeGlobA: leftGlob,
        writeGlobB: rightGlob,
        overlappingPath,
      });
    }
  }
  conflicts.sort(compareConflicts);
  return {
    ok: conflicts.length === 0,
    reason: conflicts.length ? "write_glob_overlap" : null,
    errors: [],
    conflicts,
  };
}

/**
 * Check whether a concrete repository-relative path may be edited by a task.
 */
export function checkPathEditAllowed(task, filePath) {
  const taskResult = normalizeTask(task, 0);
  if (taskResult.errors.length) {
    return {
      allowed: false,
      reason: "invalid_task",
      errors: taskResult.errors,
      path: String(filePath ?? ""),
      matchedWriteGlobs: [],
      matchedReadOnlyGlobs: [],
    };
  }

  const normalizedPath = normalizeRepositoryPath(filePath);
  if (!normalizedPath.ok) {
    return {
      allowed: false,
      reason: normalizedPath.reason,
      errors: [error("path", normalizedPath.reason, filePath)],
      path: String(filePath ?? ""),
      matchedWriteGlobs: [],
      matchedReadOnlyGlobs: [],
    };
  }

  const { task: normalizedTask } = taskResult;
  const matchedWriteGlobs = normalizedTask.writeGlobs.filter((glob) => globMatchesPath(glob, normalizedPath.path));
  const matchedReadOnlyGlobs = normalizedTask.readOnlyGlobs.filter((glob) => globMatchesPath(glob, normalizedPath.path));
  if (matchedReadOnlyGlobs.length) {
    return {
      allowed: false,
      reason: "path_is_read_only",
      errors: [],
      taskId: normalizedTask.id,
      path: normalizedPath.path,
      matchedWriteGlobs,
      matchedReadOnlyGlobs,
    };
  }
  if (!matchedWriteGlobs.length) {
    return {
      allowed: false,
      reason: "path_outside_write_globs",
      errors: [],
      taskId: normalizedTask.id,
      path: normalizedPath.path,
      matchedWriteGlobs,
      matchedReadOnlyGlobs,
    };
  }
  return {
    allowed: true,
    reason: null,
    errors: [],
    taskId: normalizedTask.id,
    path: normalizedPath.path,
    matchedWriteGlobs,
    matchedReadOnlyGlobs,
  };
}

export function isPathEditAllowed(task, filePath) {
  return checkPathEditAllowed(task, filePath).allowed;
}

export function coordinationScopeNotice() {
  return {
    checked: false,
    mergeChecked: false,
    uncommittedStateIgnored: true,
    note: "This checks declared collaboration boundaries only; it is not a git merge or worktree-status check.",
  };
}

/**
 * Match a concrete repository-relative path against the supported glob syntax.
 */
export function globMatchesPath(glob, filePath) {
  const normalizedGlob = normalizeRepositoryGlob(glob);
  const normalizedPath = normalizeRepositoryPath(filePath);
  if (!normalizedGlob.ok || !normalizedPath.ok) return false;
  const tokens = tokenizeGlob(normalizedGlob.glob);
  let states = epsilonClosure(tokens, new Set([0]));
  for (const character of normalizedPath.path) {
    const nextStates = new Set();
    for (const state of states) {
      const nextState = consume(tokens, state, character);
      if (nextState !== null) nextStates.add(nextState);
    }
    states = epsilonClosure(tokens, nextStates);
    if (!states.size) return false;
  }
  return states.has(tokens.length);
}

function findGlobOverlap(leftGlob, rightGlob) {
  const leftTokens = tokenizeGlob(leftGlob);
  const rightTokens = tokenizeGlob(rightGlob);
  const alphabet = overlapAlphabet(leftTokens, rightTokens);
  const queue = [{ left: 0, right: 0, witness: "" }];
  const seen = new Set(["0:0"]);

  while (queue.length) {
    const state = queue.shift();
    if (
      state.witness
      && !state.witness.endsWith("/")
      && state.left === leftTokens.length
      && state.right === rightTokens.length
    ) {
      return state.witness;
    }

    const epsilonStates = [
      [epsilonNext(leftTokens, state.left), state.right],
      [state.left, epsilonNext(rightTokens, state.right)],
    ];
    for (const [nextLeft, nextRight] of epsilonStates) {
      if (nextLeft === null && nextRight === null) continue;
      const resolvedLeft = nextLeft === null ? state.left : nextLeft;
      const resolvedRight = nextRight === null ? state.right : nextRight;
      enqueue(resolvedLeft, resolvedRight, state.witness, queue, seen);
    }

    for (const character of alphabet) {
      const nextLeft = consume(leftTokens, state.left, character);
      const nextRight = consume(rightTokens, state.right, character);
      if (nextLeft === null || nextRight === null) continue;
      const witness = state.witness + character;
      if (!isValidWitnessPrefix(witness)) continue;
      enqueue(nextLeft, nextRight, witness, queue, seen);
    }
  }
  return null;
}

function normalizeTask(task, taskIndex) {
  const errors = [];
  if (!isPlainObject(task)) {
    return { task: null, errors: [error(`tasks[${taskIndex}]`, "task_must_be_object")] };
  }

  for (const field of TASK_FIELDS) {
    if (!(field in task)) errors.push(error(`tasks[${taskIndex}].${field}`, "missing_task_field"));
  }
  const id = normalizeText(task.id);
  const owner = normalizeText(task.owner);
  const status = normalizeText(task.status);
  if (!id) errors.push(error(`tasks[${taskIndex}].id`, "task_id_must_be_non_empty_string"));
  if (!owner) errors.push(error(`tasks[${taskIndex}].owner`, "task_owner_must_be_non_empty_string"));
  if (!status) errors.push(error(`tasks[${taskIndex}].status`, "task_status_must_be_non_empty_string"));

  const writeResult = normalizeGlobList(task.writeGlobs, `tasks[${taskIndex}].writeGlobs`);
  const readOnlyResult = normalizeGlobList(task.readOnlyGlobs, `tasks[${taskIndex}].readOnlyGlobs`);
  errors.push(...writeResult.errors, ...readOnlyResult.errors);
  if (errors.length) return { task: null, errors };
  return {
    task: {
      id,
      owner,
      writeGlobs: writeResult.globs,
      readOnlyGlobs: readOnlyResult.globs,
      status,
    },
    errors: [],
  };
}

function normalizeGlobList(value, fieldPath) {
  if (!Array.isArray(value)) {
    return { globs: [], errors: [error(fieldPath, "globs_must_be_array")] };
  }
  const errors = [];
  const globs = [];
  value.forEach((glob, index) => {
    const result = normalizeRepositoryGlob(glob);
    if (!result.ok) {
      errors.push(error(`${fieldPath}[${index}]`, result.reason, glob));
    } else if (!globs.includes(result.glob)) {
      globs.push(result.glob);
    }
  });
  globs.sort(compareStrings);
  return { globs, errors };
}

function normalizeRepositoryGlob(value) {
  if (typeof value !== "string" || !value.trim()) {
    return { ok: false, reason: "glob_must_be_non_empty_string" };
  }
  let glob = value.trim().replaceAll("\\", "/").replace(/\/+/g, "/");
  while (glob.startsWith("./")) glob = glob.slice(2);
  if (!glob || glob.startsWith("/") || /^[A-Za-z]:\//.test(glob)) {
    return { ok: false, reason: "glob_must_be_repository_relative" };
  }
  if (glob.split("/").some((segment) => segment === "..")) {
    return { ok: false, reason: "glob_must_not_escape_repository" };
  }
  while (glob.endsWith("/")) glob = glob.slice(0, -1);
  if (!glob) return { ok: false, reason: "glob_must_be_non_empty_string" };
  return { ok: true, glob };
}

function normalizeRepositoryPath(value) {
  if (typeof value !== "string" || !value.trim()) {
    return { ok: false, reason: "path_must_be_non_empty_string" };
  }
  let path = value.trim().replaceAll("\\", "/").replace(/\/+/g, "/");
  while (path.startsWith("./")) path = path.slice(2);
  if (!path || path.startsWith("/") || /^[A-Za-z]:\//.test(path)) {
    return { ok: false, reason: "path_must_be_repository_relative" };
  }
  if (path.split("/").some((segment) => segment === ".." || segment === ".")) {
    return { ok: false, reason: "path_must_be_normalized_repository_relative" };
  }
  return { ok: true, path };
}

function tokenizeGlob(glob) {
  const tokens = [];
  for (let index = 0; index < glob.length;) {
    if (glob.startsWith("**/", index)) {
      const startIndex = tokens.length;
      tokens.push({ type: "globstarStart" });
      tokens.push({ type: "globstarBody", startIndex });
      index += 3;
    } else if (glob.startsWith("**", index)) {
      tokens.push({ type: "doubleStar" });
      index += 2;
    } else if (glob[index] === "*") {
      tokens.push({ type: "star" });
      index += 1;
    } else if (glob[index] === "?") {
      tokens.push({ type: "question" });
      index += 1;
    } else {
      tokens.push({ type: "literal", character: glob[index] });
      index += 1;
    }
  }
  return tokens;
}

function epsilonClosure(tokens, initialStates) {
  const states = new Set(initialStates);
  const pending = [...states];
  while (pending.length) {
    const state = pending.pop();
    const next = epsilonNext(tokens, state);
    if (next !== null && !states.has(next)) {
      states.add(next);
      pending.push(next);
    }
  }
  return states;
}

function epsilonNext(tokens, state) {
  const token = tokens[state];
  if (!token) return null;
  if (token.type === "star" || token.type === "doubleStar") return state + 1;
  if (token.type === "globstarStart") return state + 2;
  return null;
}

function consume(tokens, state, character) {
  const token = tokens[state];
  if (!token) return null;
  if (token.type === "literal") return token.character === character ? state + 1 : null;
  if (token.type === "question" || token.type === "star") {
    return character === "/" ? null : token.type === "star" ? state : state + 1;
  }
  if (token.type === "doubleStar") return state;
  if (token.type === "globstarStart") return character === "/" ? null : state + 1;
  if (token.type === "globstarBody") {
    if (character === "/") return token.startIndex;
    return state;
  }
  return null;
}

function overlapAlphabet(leftTokens, rightTokens) {
  const characters = new Set(["a", "/"]);
  for (const token of [...leftTokens, ...rightTokens]) {
    if (token.type === "literal") characters.add(token.character);
  }
  return [...characters].sort(compareStrings);
}

function enqueue(left, right, witness, queue, seen) {
  const key = `${left}:${right}`;
  if (seen.has(key)) return;
  seen.add(key);
  queue.push({ left, right, witness });
}

function isValidWitnessPrefix(value) {
  if (!value || value.startsWith("/") || value.includes("//")) return false;
  const segments = value.split("/");
  return !segments.some((segment) => segment === "." || segment === "..");
}

function compareTasks(left, right) {
  return compareStrings(left.id, right.id);
}

function compareConflicts(left, right) {
  return compareStrings(left.taskAId, right.taskAId)
    || compareStrings(left.taskBId, right.taskBId)
    || compareStrings(left.writeGlobA, right.writeGlobA)
    || compareStrings(left.writeGlobB, right.writeGlobB)
    || compareStrings(left.overlappingPath, right.overlappingPath);
}

function compareStrings(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function normalizeText(value) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function error(path, reason, value = undefined) {
  const result = { path, reason };
  if (value !== undefined) result.value = value;
  return result;
}

function invalidManifest(errors) {
  return {
    ok: false,
    reason: "invalid_manifest",
    errors,
    manifest: null,
  };
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
