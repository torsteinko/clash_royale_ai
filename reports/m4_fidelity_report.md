# M4.3 — Fidelity report: gpusim vs the Java reference

generated: 2026-09-15 · engine: gpusim @ `meidell/linux-state-pipeline`
reference: crforge gym-bridge on `clash-training`, run with the **patched-data
wrapper** (the M0-spec data used by both sides) — traces in
`fidelity/m4_traces/java_patched/`.

## Method

Eight fixed scenarios (`duel_knight`, `duel_musketeer`, `tower_press`,
`spell_hit`, `push_left`, `poison_zone`, `earthquake_zone`, `log_roll`) are
replayed 1:1 on both engines from the same manifest
(`fidelity/m4_traces/<name>.scenario.json`, trace format `m4-trace-v1`,
1500 ticks à 0.05 s, seed 7, `ticks_per_step=1`). The traces are compared with

    python3 fidelity/m4_scenarios.py compare <java>.jsonl <gpusim>.jsonl

## Result — the original five traces and log_roll are byte-identical to the reference

| scenario        | gpusim vs java_patched            |
|-----------------|-----------------------------------|
| duel_knight     | `cmp` identical                   |
| duel_musketeer  | `cmp` identical                   |
| tower_press     | `cmp` identical                   |
| spell_hit       | `cmp` identical                   |
| push_left       | `cmp` identical                   |
| log_roll        | `cmp` identical (M3.6, addendum below) |
| poison_zone     | zone damage exact; movement soft-differs (buff slow, #23) |
| earthquake_zone | zone damage exact; movement soft-differs (buff slow, #23) |

### M3.5 zone scenarios (poison_zone, earthquake_zone)

Added in this pass (7 scenarios total). While the two sims are in lockstep the
zone results are exact: poison ticks at ticks 85/90/… (32 ticks × 5 on the
tower, 23 per tick on units), earthquake at 82/84/… (30 ticks × 20 on the
tower, 7 per tick on units), ground/air filtering identical (poison hits the
flying minions; earthquake does not — `hitsAir false`). The first divergence
tick is 86 (poison) / 83 (quake) — the very tick AFTER the first zone damage
tick, caused by the **buff slow** that gpusim does not model
(DIVERGENCES #23): units inside a zone move at 0.85×/0.5× speed in the
reference, so trajectories drift after zone entry and later events shift.
Zone damage derivation/timing itself is java-exact.

### M3.6 The Log (log_roll)

Added in this pass (8 scenarios total). The Log is spellAsDeploy: the deploy
projectile spawns AT the cast point and travels 3 game units toward the enemy
side (the reference's preserved legacy `round(minDistance / 1000)` arithmetic),
then spawns the rolling piercing sub-projectile (raw speed 200 → float32
166.6666717529297 units/tick, range 10.1 tiles, hit radius 1.95 + collision
radius, minDistance 2.5 tiles, pushback 700, crown −85). The trace is
`cmp`-identical: the red knight takes exactly one 268 hit (scaleCard(105) at
level 11) on tick 115 — the min-distance gate tick; a gate-less model would
hit on tick 113 — is pushed back along the roll direction (35 game units/tick
for 10 ticks, `u_kb_time` 0.5 s), the flying MegaMinion crossing the roll's
path takes no damage (aoeToGround only), and the red left princess tower takes
the 15 % crown damage (40) at tick 147. The roll deactivates at range end
(61 motion ticks).

### M3.6 fixes made along the way

1. **Deploy formation offsets** (`TroopFactory.createTroop`): red mirrors the
   formation `(-dx, -dy)` in the java frame, and the sim frame is y-flipped —
   `deploy()` now applies `(dx, -dy)` for blue / `(-dx, +dy)` for red. Found
   because the reference spawned red minions at the mirrored offsets.
2. **Trace unit names**: gpusim wrote the card name ("minions"); the reference
   runner writes the unit name ("minion"). `_serialize_env` now emits unit
   names with the java-side normalization.
3. A first log_roll draft used 3 Minions for the air-immunity check; the java
   reference separates clustered AIR units every tick (air-air collisions) while
   gpusim does not (#3), so their trajectories drift after spawn. The scenario
   now uses the single-unit MegaMinion; multi-unit clusters stay outside the
   byte-verified set until collisions are ported (post-M5 per the checklist).

Determinism: two full gpusim passes are byte-identical (`reports/m4_scenarios_report.md`).
Compare tool: for the original five scenarios and log_roll `first_divergence_tick = None`,
all `max_abs_diff` fields 0.0; the two zone scenarios diverge only via the documented
buff-slow movement (#23).

## Thresholds

* Equality gates on the trace fields `th`/`hp`/`el`/`cr`/`ta`/unit `x`/`y`:
  **exact** (0.0) — the traces carry integer game-unit geometry.
* `t` is derived on both sides from the tick index (`round(i * 0.05, 3)`), so
  it is exact as well. (Before the M4.2b fix the gpusim side wrote the fp32
  clock, leaking a 0.001 s formatting diff from tick 1031 onward.)
* No open timing deviations remain inside the scenario set.

## Fixes made in this pass (all verified by the byte-identical traces)

1. **Crown tower collision radius** — gpusim used 1.5 tiles; the reference is
   `Tower.CROWN_COLLISION_RADIUS = 1400` (1.4). Symptom: units stopped ~0.1
   tile early at max attack range (push_left: the musketeer froze at d=7.96
   instead of 7.86 from the king, shifting the first king hit by ~2 ticks and
   producing a transient 263 HP difference before re-syncing). Fixed in
   `gpusim/env.py`; the whole push_left cascade disappeared.
2. **Multi-spawn stagger** — the delay is stored as float32, so `0.1` came
   back as `0.1000000015` and `ceil(0.1 / 0.05)` landed on 3 ticks instead of
   2 (Java: `0.1f = exactly 2 * 0.05f`). Fixed with a 4-decimal rounding in
   `_stagger_ticks` plus a half-tick snap against float32 decrement residue in
   the spawn-fire condition. Minions now spawn on ticks 21/23/25; skeletons
   (no `summonDeployDelay` in the data) spawn all three at once, as the
   reference `Card.java` specifies ("Zero = all at once").
3. **Trace `t` field** — derived from the tick index instead of the fp32 sim
   clock (matches the Java-side runner).

Earlier M4.2 work: the card-cycle FIFO fix (draw-queue `% 4`) and the
`fidelity/java_random.py` shuffle replica (both verified against the recorded
replays, `reports/m4_replay_findings.md`).

## Known deviations (outside the scenario set)

* `DIVERGENCES.md` stays the authoritative list. Open items at this revision:
  non-homing/arc projectiles (#14), scatter/pierce/returning projectiles (#16),
  unit collision/occupancy (#1/#3 — accepted until M5; observed as per-tick
  air-air separation in multi-unit deploys, which keeps those out of the
  byte-verified set), spawn overflow (#10), buff system (#23 — the reason the
  zone scenarios stay soft).
* #22 (recorded-replay diffs, real-client capture): the simulator-vs-simulator
  gate above is fully exact; the recording-based diff is re-run in this pass
  (see `reports/m4_replay_diff.*`). A recording is an *observation* of the real
  client (15-tick sampling, label artifacts), not the reference engine — any
  residual difference there is weighed against these exact traces before
  acting on it.

## Reproduce

```bash
# gpusim side (regenerates traces + report)
python3 fidelity/m4_scenarios.py run
for n in duel_knight duel_musketeer tower_press spell_hit push_left log_roll; do
  python3 fidelity/m4_scenarios.py compare \
    fidelity/m4_traces/$n.gpusim.jsonl fidelity/m4_traces/java_patched/$n.java.jsonl
done
# Java side (on clash-training; wrapper = patched M0 data shadow classpath)
~/venvs/clash/bin/python ~/m4run/m4_scenarios_java.py \
  --scenarios-dir ~/m4run/scenarios --out ~/m4run/out_patched --seed 7 --passes 2 \
  --bridge-bin ~/m4run/gym-bridge-patched            # --only log_roll for a single scenario
```
