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

## POC runs (started Sept 14 2026)

- Run A: default decks, self-play, 1M steps (`train_ppo.py --opponent self_play`) — running
  locally on Olsen's machine (Ryzen 5 5600X): first smoke green at **450–540 steps/s**
  single-env (vs ~120–170 on the 2-vCPU VM).
- Run B: two different decks — Hog cycle vs Giant beatdown — self-play, 400k steps,
  then eval vs `rule_based` and `noop` (`poc_decks.py`)

Logs on the VM: `/tmp/crforge_poc/poc_chain.log`. Results summarized here when done.
