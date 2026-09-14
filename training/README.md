# RL training in the CRForge simulator (POC)

Status: **moved to Olsen's Windows desktop (RTX 5070 Ti)** on Sept 14 2026 — the VM run was
stopped on request. Starter pack for the local run: `STARTER_PACK_WINDOWS.md`.

## Why

Self-play RL against a real-time client (or a private server) is capped at real-time
(3-minute matches), and private-server battles are simulated client-side (the server is
just a relay — verified against the Null's Royale v14.593.1 source). The fast RL loop
runs in `crforge` — see `../docs/CRFORGE.md` for the environment guide and benchmark
(0.74 s per match headless on this VM ≈ 327× realtime).

## Fidelity verification (measured Sept 14 2026, level 11)

Verified equal to real Clash Royale values:

- Knight: 1766 HP / 202 damage / 1.2 s hit speed / speed 60 / 1.0 s deploy — exact
- King Tower: 4824 HP — exact. Princess Tower: 3052 HP, damage 109, 0.8 s hit speed — exact
- Elixir: start 5.0, regen exactly 1 per 2.80 s; double elixir starts at t=120.0s — exact
- Overtime: starts at t=180s; x3 elixir in the final OT minute (t=240s) — exact
- Match resolution: king kill = instant 3-crown end; crown lead at 3:00 wins;
  0-0 goes to overtime, capped at 5:00 — matches the real rules

Gaps / quirks found:

- `--opponent rule_based` is broken in binary mode (which `train_ppo.py` hardcodes): the
  opponent silently no-ops because `env._rule_based_action` doesn't pass the flat
  observation. Verified Sept 14: a random policy "wins" 30/30 against the dead bot;
  after a one-line fix (`obs_flat=self._last_obs_flat` in `crforge_gym/env.py`), 2/2/26.
  Upstream issue candidate.
- Self-play (`--opponent self_play`) works in binary mode (mirrors the flat observation),
  but red cannot see its own hand (upstream code comment acknowledges this) — acceptable
  for the POC, worth improving later.
- Spells cannot hit towers through the 10-zone action space (fireball dealt 0 damage
  from zones 7/8/9; zones are ~3.6 tiles from princess towers vs fireball radius 2.5).
  An action-space extension would be needed for spell-finish strategies.
- `blueActionFailed` / `redActionFailed` flags are always true for attempted actions
  (the elixir check runs before `engine.tick()`); upstream issue candidate.
- Missing cards: Minion Giant, Void, Cannon Cart. 240 card entries exist
  (130 base + 21 evolutions + 89 heroes); missing ones can be added via the data JSONs.
  Evolutions/heroes are separate card IDs with explicit base links (`heroForm` /
  `baseCard`, `evolved` / `evolvedCard`) and can be placed in decks (verified).

Probe scripts + raw reports: `fidelity_probe.py` (scripted knight/elixir check —
note: probe 1's elixir slope analysis was flawed; see probe 2), `fidelity_probe2.py`,
`fidelity_probe3.py`, `fidelity_report_probe1.txt`, `fidelity_report_probe2.txt`.


## Local fixes applied to crforge (patch: `patches/crforge-poc-fixes.patch`)

1. `python/crforge_gym/env.py` — `_rule_based_action` now passes `obs_flat`, so the
   rule-based opponent actually plays in binary mode (it silently no-opped before).
2. `core/.../GameEngine.java` + `GameState.java` — time-limit decisions (crowns at
   3:00, overtime crowns, overtime tower-health tiebreaker) now propagate the winner
   to the GameState, so the reward calculator reports win/loss instead of always
   "draw". Verified: chip scenario +29.98 (was -0.02); random-vs-random 3/7/0
   (was 10/10/0 draws); self-play run at 6k steps: win/loss ~50/50, draws 0%
   (was 93% draws).

Apply on the Windows box (from the crforge root):

```bat
curl -L -o poc_fixes.patch "https://raw.githubusercontent.com/torsteinko/clash_royale_ai/meidell/linux-state-pipeline/training/patches/crforge-poc-fixes.patch"
git apply poc_fixes.patch
gradlew.bat build
```


Patch 3: `patches/crforge-windows-fix.patch` — Windows support for `train_ppo.py`
(uses `gradlew.bat`, resolves `gym-bridge.bat`, launches via `cmd /c`, JAVA_HOME only when
set). Without it, jpype / `--num-envs` modes crash on Windows with WinError 193
(the script runs the Unix `gradlew` shell script via CreateProcess).
Workaround without patching: run `gradlew.bat :gym-bridge:installDist` once — the
start-script check then passes and jpype mode works.

## Multi-process self-play (`multi_selfplay_train.py`)

One trainer + N worker processes, **each with its own bridge server and its own
self-play opponent**. Each worker's opponent reloads the trainer's latest snapshot
(atomic tmp+rename save, mtime check) so self-play keeps improving while N simulators
run in parallel — the combination `train_ppo.py` blocks in subprocess mode (it only
supports `self_play` on the single-env / jpype path).

Run from the crforge repo root (launches and cleans up its own servers):

```bat
curl -L -o multi_selfplay_train.py "https://raw.githubusercontent.com/torsteinko/clash_royale_ai/meidell/linux-state-pipeline/training/multi_selfplay_train.py"
python multi_selfplay_train.py --num-envs 5 --steps 2000000
```

- Blue plays a fixed deck (default **2.6 Hog cycle**); worker i's opponent plays
  `RED_POOL[i % 6]`: Hog / Giant beatdown / Log bait / X-Bow / LavaLoon / Splashyard
  (all card ids verified against cards.json) — i.e. "learn 2.6 Hog against the field".
- Ports default to 9890+ (does not touch the usual 9876 bridge). `--red-pool hog` for
  mirror-only self-play.
- Validated end to end on the 2-vCPU VM (2 envs: servers up → snapshots reloaded in both
  workers → training → eval → clean shutdown). Expect roughly 1/3 core per env: on the
  5600X, N=5 should land near **5 × single-env fps**.

**jpype does not scale** (measured on the 5600X, Sept 14): 5 envs via jpype = 350 fps
total vs 318 fps single-env ZMQ — i.e. ~70 fps per env (~0.2×). Use jpype only for
single-env convenience; use this script for throughput.

**Bridge startup robustness:** upstream `_wait_for_server` re-tries the ZMQ init every
second with a 2 s timeout *while the JVM is still booting*. If an attempt is abandoned
mid-handshake under load, the PAIR server can be left holding a dead peer and stop
accepting new handshakes (observed on the VM: the server log shows one "Game session
reset" and then nothing). This script instead TCP-polls the port first, does one
handshake with a 12 s timeout, and restarts that server (max 3×) if the handshake
fails. Upstream issue candidate.

## POC runs (started Sept 14 2026)

- Run A: default decks, self-play, 1M steps (`train_ppo.py --opponent self_play`) — running
  locally on Olsen's machine (Ryzen 5 5600X): first smoke green at **450–540 steps/s**
  single-env (vs ~120–170 on the 2-vCPU VM). Self-play with the draw fix: win=66% /
  loss=34% / draw=0% (was 93% draws before the fix).
- Run B: two different decks — Hog cycle vs Giant beatdown — self-play, 400k steps,
  then eval vs `rule_based` and `noop` (`poc_decks.py`)
- Run C (`multi_selfplay_train.py`): N-process self-play with the deck pool — the main
  throughput path; supersedes jpype for multi-env (see above).

Logs on the VM: `/tmp/crforge_poc/poc_chain.log`. Results summarized here when done.
