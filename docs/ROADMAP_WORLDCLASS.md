# Roadmap: toward world-class Clash Royale play

Goal (Olsen, Sep 15 2026): make the bot as good as possible — world-class level.
"Good enough" is not a goal; if the current training approach cannot reach it,
the approach must be replaced. This document is the working plan for that.

## Status quo

- Self-play PPO over **crforge** (Java battle sim), CPU only.
- Live run: 40 envs (4 workers x 10 games), ~1700-2400 steps/s on one
  c3-highcpu-8 box (8 vCPU = 4 physical cores). 50M-step run in progress.
- Learning proven with control evals; 12-deck pool; deterministic eval harness;
  live dashboard; multi-game worker architecture (this repo).
- Measured ceiling of the CURRENT stack on this box: ~2-4k steps/s
  (python/scheduling-bound; the raw sim+IPC layer alone does 7.5-9.7k/s).

## The honest verdict

The current stack cannot reach world-class on any realistic budget: serious
agents (AlphaStar, OpenAI Five) train at 1e5-1e6 env-steps/s for weeks.
We are 2-3 orders of magnitude short. Four levers, ALL needed:

1. **Simulator fidelity.** Everything trains against the sim. Known gaps:
   giant +21% HP, missing cards (Minion Giant, Void, Cannon Cart), xbow
   placement limits, overtime/tiebreak rules. Wrong physics caps the ceiling
   AND breaks any sim-to-real transfer. Must be fixed regardless of engine.
2. **Throughput: GPU-batched simulator.** Vectorize thousands of battles as
   tensor ops (JAX / Warp / CUDA): state = [B, entities, ...], one kernel
   advances all B games one tick (the Isaac Gym / Brax pattern). Sim + policy
   inference + PPO update all on the GPU, minimal CPU round trips.
   Target: >=100k env-steps/s on a 5070 Ti (vs ~1.8k now). Requires a rewrite
   of the battle core as a batched tensor program, validated against crforge
   by a differential harness (scripted scenarios + statistical equivalence;
   NOT bit-identical -- see decisions).
3. **League-based self-play.** Main agents + exploiters + historical pool
   (AlphaStar lesson). Our current snapshot-pool opponent is a v0 of this.
4. **Human-replay BC warm start.** KataCR dataset (fast_hog_2.6 etc.):
   supervised init from top-player trajectories, then RL fine-tuning.
5. (later, separate track) sim-to-real: CV/execution layer (YOLO on GPU);
   ToS-sensitive if it goes near the live ladder.

## Phases

- **P0 (now, days):** fidelity audit vs the real game (card constants,
  missing cards); differential test harness v0 (record scenarios from
  crforge, compare outcomes across engines); benchmark suite / Elo ladder
  (vs rule_based, vs old snapshots, per-deck).
- **P1 (weeks):** GPU sim prototype: movement + combat + towers + the 8
  hog-2.6 cards, batched (>=1024 games) in JAX; parity runs vs crforge
  scenarios; throughput measured on the 5070 Ti.
- **P2 (weeks):** full card set + spells + overtime; GPU-native PPO
  (single process, no per-step CPU sync); obs = current 1094 layout first,
  set-transformer over entities later.
- **P3 (weeks):** league self-play + BC from KataCR + larger nets;
  continuous long runs.
- **P4 (later):** sim-to-real / CV pipeline; ladder research.

## Compute options

- A. Olsen's RTX 5070 Ti (his box): fast, free; needs his machine running
  and a workflow for remote execution.
- B. Rented GPU VM on GCP credits (24/7): recommended for P1+.
- C. Interim, no new code: CPU fleet on credits running the CURRENT stack
  (5-10 x c3-highcpu-8 ~ 10-20k steps/s) as baseline data + opponent pool.

## Decisions needed

- [ ] Standard shift: GPU sim is "validated-equivalent", not bit-identical
      with the Java sim. (Required for any GPU rewrite.)
- [ ] GPU compute: rented VM (B) vs local 5070 Ti (A)?
- [ ] Keep the current 50M CPU run as baseline while P0/P1 proceed?
      (Recommended: yes -- it finishes overnight and is our reference.)
