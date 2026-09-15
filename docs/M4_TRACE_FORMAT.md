# M4 differential harness — trace & report formats (v1)

The acceptance gate for "validated-equivalent" is: the same fixed scenarios run
in **gpusim** and in the **Java reference** (crforge gym-bridge), and their
per-tick traces diff within tolerances. This document is the contract both
sides emit against.

Produced by the gpusim side with:

```
python3 fidelity/m4_scenarios.py run                      # traces + report
python3 fidelity/m4_scenarios.py compare A.jsonl B.jsonl [--tol T]
```

## 1. Files

| file | content |
|---|---|
| `fidelity/m4_traces/<name>.scenario.json` | manifest (format `m4-scenario-v1`): decks, action schedule, tick semantics, tower slot order |
| `fidelity/m4_traces/<name>.gpusim.jsonl` | gpusim trace (format `m4-trace-v1`), one JSON object per tick |
| `reports/m4_scenarios_report.md` / `.json` | summary report of the last run (format `m4-report-v1`) |

## 2. Trace format (`m4-trace-v1`)

One JSON object **per line, per tick**, in tick order. Record `i` is the state
after `i` ticks; game time `t = i * 0.05 s`. Record 0 is the pre-tick baseline.

| key | type | meaning |
|---|---|---|
| `i` | int | tick index (0..duration_ticks) |
| `t` | float (3dp) | game time in seconds |
| `el` | [b, r] (3dp) | elixir, blue / red |
| `th` | 6 floats (2dp) | tower HP, fixed slot order (below); floors at 0 (Java `Health.takeDamage` semantics — no negative overkill) |
| `ta` | 6 ints | tower alive flags (1/0) |
| `cr` | [b, r] ints | crowns |
| `go` | int | game over flag |
| `w` | int | winner: -1 undecided, 0 blue, 1 red, 2 draw |
| `u` | list, optional | active units (omitted when none) |

Tower slot order (identical both sides):
`[king_blue, princess_blue_left, princess_blue_right, king_red, princess_red_left, princess_red_right]`
— Java tower lists are `{type: crown|princess, x}`; map `x<9` → left,
`x>=9` → right, crown → king.

Unit record: `{"s": 0|1, "c": <card/unit name, lowercase>, "x": 3dp, "y": 3dp, "hp": 2dp}`.
Only active units; sorted by `(s, x, y)` — both sides must emit that order.

## 3. Action schedule & tick semantics (reproducibility)

Manifest `actions`: `[{"at_tick": k, "side": 0|1, "slot": 0..3, "x":, "y":, "card":}]`.
`slot` is the hand index (0..3), `(x, y)` in tiles.

**Semantics:** an action with `at_tick=k` is applied *between* record `k` and
record `k+1` (when game time equals `k*0.05 s`). On the Java side with
`ticks_per_step=1`: submit the action in **step call number `k+1`** (1-based):
`reset()` → record 0; step 1 with that action → record 1; etc. Record after
**every** step.

Decks: both sides use the manifest `decks` list; the **first 4 cards are the
opening hand** (the scenarios only ever play slots 0–2, so no rotation drift).
Scenario actions carry the expected `card` name — both sides should assert the
hand slot holds that card when the action is applied.

## 4. Report format (`m4-report-v1`)

`reports/m4_scenarios_report.json`: `{format, source, engine, generated,
data_dir, level, dt, duration_ticks, deterministic, scenarios: [...]}`; each
scenario: `{name, desc, ticks, sha256, checkpoints: [{tick, t, tower_hp_blue,
tower_hp_red}], first_damage: {blue, red}, unit_deaths, final: {tower_hp,
towers_alive, crowns, winner, units_alive}}`. The `.md` renders the same
numbers (checkpoints every 15 s). M4.3 will add the Java-side numbers and the
enforced thresholds/known deviations.

## 5. Determinism & workflow

- The sim core has no RNG; `run` executes the whole batch **twice** and reports
  `deterministic: true` only when both passes are byte-identical. Re-running
  the runner must be idempotent (byte-identical `.gpusim.jsonl`).
- After any behaviour-affecting change to `gpusim/`, regenerate and commit the
  traces in the same commit (`python3 fidelity/m4_scenarios.py run`); the diff
  history of those files *is* the record of behaviour changes.
- `gpusim/tests/test_m4.py` pins the tick-exact expected events (cadence,
  first-hit ticks, damage values) so behaviour changes fail loudly.

## 6. Known caveats (do not mistake for diff noise)

- **Spell flight time (M3.4):** spells currently apply instantly at the cast
  tick; when flight time is ported, `spell_hit` impact timing shifts a few
  ticks and the trace + test constants are updated in that change.
- **Same-tick resolution:** all attacks of a tick are applied, then deaths are
  resolved at end of tick (mirror duel `duel_knight` → mutual KO on the same
  tick). Java source inspection shows the same shape (`processDeaths()` runs
  after combat; no mid-tick alive re-check in `CombatSystem`) — the M4.2 trace
  diff is the confirming test.
- Units that died earlier in the same tick can still be "hit" (damage clamps
  at 0, invisible); the attack is consumed either way. Expected equivalent on
  the Java side (the shot at a dead target also consumes the cooldown) —
  verify in the same diff.
