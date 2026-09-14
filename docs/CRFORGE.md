# CRForge — headless RL training environment (verified Sept 14 2026)

**Why:** self-play RL against a live client / private server is capped at real-time
(one 3-minute match = 3 minutes of wall clock). Private-server battles are
simulated **client-side** — verified against the Null's Royale v14.593.1 server
source (TypeScript): the server is a lobby/protocol relay, there is no battle
simulation server-side, so no server-side speedup is possible. The fast RL loop
therefore belongs in a headless simulator.

**crforge** (<https://github.com/voonhous/crforge>) is a headless Clash Royale
battle simulator in Java (Apache-2.0, actively developed) built exactly for RL:

- deterministic tick engine, 20 TPS, **no wall-clock dependency** → runs as fast as the CPU allows
- 240 card entries (`data/src/main/resources/cards/cards.json`)
- Python Gymnasium env (`crforge_gym`) over a ZMQ bridge; optional in-process JPype backend
- binary observation (flat 1079-vector, default) or JSON dict mode
- action space `MultiDiscrete([2, 4, 10])`: action type (no-op / play) × hand slot × placement zone (7 own-half + 3 spell zones)
- training stack: SB3 **MaskablePPO** with action masking (sb3-contrib), self-play opponent
  snapshots, TensorBoard logging, threaded vec env, BC pretraining from a rule-based expert

## Verified benchmark (agent-studio-01 VM, 2 vCPU, no GPU)

- 15 random-vs-random episodes, warm average: **0.74 s / episode** (~808 steps, ticks_per_step=6)
- single env: ~4,900 episodes/hour; engine ~6,500 ticks/s ≈ **327× faster than realtime**
- episode 1 incl. JVM warmup: 1.2 s

On a desktop with N cores, run N parallel envs (`--num-envs N` auto-launches N bridge
servers) for near-linear scaling.

## Quickstart (Windows box w/ RTX 5070 Ti)

```bat
:: Java 17 required (e.g. Eclipse Temurin 17)
git clone https://github.com/voonhous/crforge
cd crforge
gradlew.bat build

python -m venv .venv
.venv\Scripts\pip install -e "python/[train]"
:: Blackwell GPU torch: use the cu128 index-url (see docs/TRAINING.md)
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128

:: single-env: start a bridge server yourself, then train
gradlew.bat :gym-bridge:run
.venv\Scripts\python python/examples/train_ppo.py --timesteps 1000000 --opponent self_play

:: multi-env: auto-launches N servers
.venv\Scripts\python python/examples/train_ppo.py --num-envs 8 --opponent rule_based

:: behavior-cloning warm start (rule-based expert) then fine-tune
.venv\Scripts\python python/examples/pretrain_bc.py --episodes 200
.venv\Scripts\python python/examples/train_ppo.py --resume models/ppo_crforge_bc
```

## Role in the pipeline

1. **Offline bootstrap:** replay data (KataCR/MIT) → policy warm start (see docs/TRAINING.md)
2. **Self-play RL:** crforge, fast + unlimited (this doc)
3. **Sim-to-real:** evaluate / fine-tune against the real client via the ADB + CV pipeline
   (state-mapping note: sim state vector ↔ CV-extracted state)

## Known quirks

- `python/examples/run_episodes.py` is stale: `CRForgeEnv` now defaults to
  `binary_obs=True` (flat `np.ndarray`, shape `(1079,)`), while the example still
  indexes `obs["game_time"]` like a dict. Access the flat vector directly instead.
- Self-play with `--num-envs > 1` requires `--jpype` (single-process threaded mode).
