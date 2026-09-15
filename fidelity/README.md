# Fidelity & verification ladder (P0)

Why this exists: a policy trained on a wrong simulator is a policy for a
different game. Every hour of training is only as valuable as the engine's
fidelity. This directory holds the tools and the acceptance criteria.

## Ground truth: what is actually knowable

Perfect parity with Clash Royale is **impossible** to guarantee: the battle
logic executes on Supercell's servers (closed source; nobody outside has it).
Even long-running private-server projects (e.g. Null's Royale) are
re-implementations tuned against *observed* behaviour, not decompiled servers.
So the goal is not bit-perfection. The goal is:

> **No strategic divergence, with every known divergence measured, bounded,
> and tracked.**

- Small numeric friction (a frame here, a rounding there) = acceptable.
- Strategic divergence = NOT acceptable:
  - placement rules (what/where can be deployed)
  - pathing / aggro / retargeting
  - targeting priorities (air/ground/buildings-only, king activation)
  - elixir economy & card-cycle mechanics
  - spell behaviour (radii, damage falloff, tower damage %)
  - timing windows that change trades (hit speeds, deploy times, first-hit)

## Truth sources, in priority order

1. **Live community card database (`noff.gg`)** — server-rendered, embeds the
   complete card DB (127 cards) as JSON in every card page. Fetchable from our
   boxes; re-fetchable after every balance patch. Tool: `fetch_noff.py`
   (verified 2026-09-15). Historical baseline: cr-api-data dump (Oct 2023) for
   diffing what changed since.
   - *The APK route is CLOSED (settled 2026-09-15):* the 2026 client is a 148 MB
     bootstrap — `csv_logic` is gone from the bundle; game data downloads from
     the CDN post-install and ships in Supercell's `ECC2` encrypted containers
     (magic verified in `libg.so.*.ecc`, no public tooling exists). Re-check
     only if the modding scene cracks `ECC2`.
2. **Recorded real matches** — replay real action sequences INSIDE our engine
   and diff the state evolution against what actually happened. This is the
   emulator-verification pattern (Dolphin/MAME/bsnes all do this) and our
   single most powerful tool.
3. **Probes in the real game** — scripted single interactions measured on
   recordings/through the emulator (we already have a small battery in
   `training/fidelity_probe*.py` + reports).
4. **Targeted RE of native code** — only if a divergence cannot be explained
   by 1-3. (Expensive: Ghidra/dynamic analysis; an assist is possible, a
   silver bullet it is not.)

## The ladder

- **L1 — Constant parity.** Every card's HP/damage/hit-speed/range/speed/cost
  in our engine == the reference tables. Exact, automated, must be 100%.
  Tool: `check_constants.py` (card-level diff vs either reference; alias map
  for crforge's internal names). First runs: 2023 baseline → 141 field
  mismatches; live 2026 dump → **152 stat mismatches + 4 elixir-cost errors
  across 110 matched cards** (reports in `report_l1_*.md`).
- **L2 — Micro-mechanic probes.** Scripted interactions with per-metric
  tolerances (e.g. knight-vs-knight time-to-kill, elixir rate, fireball on
  tower, single-hog damage to princess tower). Battery grows over time;
  runs on every engine change.
- **L3 — Replay differential.** Full recorded matches (action log + state
  trace) replayed in our engine; compare tower HP / elixir / unit survival /
  event timings with tolerances. Regressions block changes. (Harness = the
  next P0 build item after L1.)
- **L4 — Distributional checks.** Outcome/pace/elixir distributions of sim
  matches vs real match corpora; statistical tests for drift.
- **L5 — Closed loop (later).** With the emulator + test account: the bot
  plays real matches; systematic sim-vs-real gaps feed fixes back.

## Known candidates (to be settled by L1)

- Giant HP discrepancy (~+21% flagged earlier) — may actually be a
  level-scale mixup in OUR reference (level 9 -> 11 scale change), not an
  engine error. APK data decides.
- Missing cards (Minion Giant, Void, Cannon Cart in our id space).
- xbow bridge placement; overtime/tiebreak rules; log vs goblin barrel.

## Standing rule

No engine/mechanic changes merge without: (L1 full pass) + (L2 battery pass) +
(L3 subset). Fidelity regressions are release-blocking, same as crashes.
