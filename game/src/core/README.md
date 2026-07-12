# El Alamein Core Rules Contract

This folder is the deterministic rules layer for the El Alamein game. It should stay small, pure, and easy to test directly from Node.

## Future Codex Session Rules

If you are a future Codex session editing this folder:

1. Read this file before changing rule code.
2. Change one rule behavior group at a time.
3. Add or update tests that import `src/core/index.js`.
4. Keep UI text, DOM work, animation, storage, and logging outside this folder.
5. Return structured facts or plans that `game.js`, app, UI, and replay code can execute.
6. Do not duplicate a rule in both `game.js` and `src/core/`; keep `game.js` as the caller while the transition is in progress.
7. Run `npm.cmd run check` before reporting completion.

When the worktree is dirty, assume unrelated changes may belong to the user or another agent. Read the relevant diff and preserve it.

## Boundary Rules

- No DOM, canvas, CSS, browser events, localStorage, fetch, timers, animation, Supabase, or Firebase.
- No direct randomness. Dice rolls must be injected by app or replay code.
- No localized UI text. Return structured data, IDs, booleans, numbers, paths, and reason codes.
- Do not mutate scenario or rules data. Treat them as read-only inputs.
- Prefer explicit inputs: `board`, `rules`, `units`, `state`, and options.
- Prefer explicit outputs: maps, sets, arrays, result objects, and event-ready facts.
- Keep functions narrow enough to test with small fixture units.

## Context Shape

Core functions should accept a small context object when they need shared game data:

```js
{
  board,
  rules,
  units,
  state
}
```

`board` should come from `createBoard(scenario)`. Callers outside core are responsible for loading JSON, saving games, drawing UI, logging messages, and translating reason codes into display text.

## Comment Policy

Use comments to explain rule intent, scenario exceptions, or non-obvious edge cases. Do not comment ordinary JavaScript mechanics.

Good:

```js
// El Alamein rule: a unit may not step directly from one enemy ZOC hex into another enemy ZOC hex.
```

Avoid:

```js
// Loop through neighbors.
```

Public core functions that encode rule behavior should have short JSDoc blocks describing their inputs, outputs, and important rule constraints.

## Result And Reason Codes

When a rule can reject an action, prefer a structured result over a UI sentence:

```js
{ ok: false, reason: "enemy_zoc_to_enemy_zoc" }
```

UI code can translate the reason. Replay and tests can compare the reason exactly.

## Extraction Checklist

When moving a behavior out of `game.js`:

1. Add or update tests around the current behavior.
2. Extract one behavior group into this folder.
3. Keep the extracted function pure and deterministic.
4. Make `game.js` call the extracted function as a compatibility layer.
5. Run `npm.cmd run check`.
6. Document any intentional behavior change.

## High-Risk Rule Areas

Changes in these areas need focused tests before or alongside implementation:

- ZOC entry and exit restrictions.
- Retreat path legality and retreat through friendly occupied hexes.
- Stacking and occupied-hex movement.
- First-turn Allied movement reduction.
- Combat odds and CRT column mapping.
- AR, DR, AE, and DE combat result effects.
- Advance after combat.
- Phase gates and cleanup.
- Victory conditions.
- Save/load roundtrip and replay state hash.

## Event Direction

Core rules should expose enough structured facts for app and replay code to produce events such as:

- `UNIT_MOVED`
- `MOVE_UNDONE`
- `COMBAT_DECLARED`
- `COMBAT_RESOLVED`
- `RETREAT_REQUIRED`
- `UNIT_RETREATED`
- `UNIT_ELIMINATED`
- `UNIT_ADVANCED`
- `PHASE_ENDED`
- `TURN_STARTED`
- `GAME_ENDED`

Combat-related events must include battle IDs and injected die rolls. Replay must never depend on localized log strings.
