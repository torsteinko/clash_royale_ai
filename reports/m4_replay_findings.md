# M4.2 findings — recorded Java replays replayed in gpusim

Status 2026-09-15 (M4.2). Machine-generated numbers: `reports/m4_replay_diff.{md,json}`
(8 replays from the pool12 recording set, committed under `fidelity/replays/`).
Tool: `python3 fidelity/m4_replay_diff.py run --dir fidelity/replays --out reports`.

## What was replayed

For each replay: blue's recorded actions are applied at their recorded step at
their exact (x, y); red is reconstructed from new-entity evidence (spawn name +
count must match a card of red's recorded deck and fit red's elixir budget),
with unobservable plays (spells are invisible in frames; ability spawns such as
witch skeletons arrive 4 at a time vs the 3 of the skeletons card) *flagged*
rather than replayed. Then both elixir tracks and the 6-slot tower-HP track are
diffed per step against the recorded frames (tolerances: HP ±1, elixir ±1.5).

## Finding 1 — BUG FIXED: card-cycle draw walked off the 4-slot queue

`play()` advanced `cycle_pos` with `% 8` while the draw queue only holds 4
cards: after the 4th play in a match the next draw read an uninitialised slot
and a hand slot silently became `-1` (every later play in that slot failed,
card invisible, rotation lost). All recorded replays hit this after their 4th
blue play — that is what the first diff run exposed.

Fix: `cycle_pos % 4` with wraparound (exact FIFO equivalent of the Java
`Hand.playCard` queue, unchanged for the first 4 plays). Regression test:
`test_m3.py::test_full_cycle_draw_order_no_empty_slots` (fails on the old code
with `hand must never contain -1: [-1, ...]`). After the fix, blue actions are
applied in 20/20 … 75/86 of the recorded steps per replay and the blue elixir
track matches the recording within ~1–3 elixir across all 8 replays.

## Finding 2 — the Java hand SHUFFLES the deck at reset (documented)

`Hand.java` runs `Collections.shuffle(deckCards, random)`; `GameSession.reset`
derives `new Random(seed)` for blue and `new Random(seed + 1)` for red when the
reset carries a seed (the recorder does). gpusim takes the deck order as given
(the caller may shuffle). For replay work the shuffle is now replicated
bit-exactly (`fidelity/java_random.py`, java.util.Random + Collections.shuffle);
the model explains the recorded blue labels in 20/20 … 40/42 … 35/36 of the
steps per replay — strong evidence both the shuffle replication and the FIFO
hand model are right. The remaining labels are isolated or consecutive-duplicate
entries in the recorded files (e.g. two `Cannon` labels at steps 123+124 in
s7000, two `Skeletons` at 58+59 in s7002) — a *recording-side* artifact to
investigate later; the tool falls back to entity evidence and flags them.

## Finding 3 — tower-HP diff: remaining divergence has two classes

1. **Reconstruction gaps (expected):** red spells and ability spawns are
   invisible in frames, so a replayed match cannot be identical once red uses
   them. The flags per replay quantify this (e.g. 24–31 not-replayed red plays
   in the hog-vs-giant/xbow games; 0–1 in the cleanest one, s7011).
2. **A real unit-level difference to investigate:** in `giant-vs-hog s7011`
   (20/20 blue actions applied, only 1 red play not replayed) blue's Minions die
   ~4–5 steps earlier in gpusim than in the recording (sim: 3→2→1 by step 11;
   recording: 2 minions still alive at step 14), and the first tower-HP
   divergence (step 15: recorded red-left princess at 2945, sim still 3052)
   follows from exactly that. Air pathing itself matches (both fly straight; the
   sim's `_movement` and the Java `BasePathfinder` treat AIR as direct), so the
   next suspects are tower targeting/range geometry against flying units and
   shot cadence (`DIVERGENCES.md` #17) — investigate in M4.2b/M4.3 with a
   per-tick unit trace of this exact minute of the match.

## Repro / inputs

- Replays: `fidelity/replays/*.json` (copied from the recording box's
  `~/replays/`, format documented in `training/record_replays.py`).
- Blue labels verified against the shuffle+FIFO model: see
  `blue_labels_model_match` per replay in `reports/m4_replay_diff.json`.
- No source file was modified to make a comparison pass; the one behavioural
  change (Finding 1) is a bug fix with its own regression test.
