# Starter pack — run the CRForge PPO POC on Windows (RTX 5070 Ti box)

Goal: run the POC locally, much faster than the 2-vCPU VM: many parallel simulators (CPU
cores) + PPO. Note: the GPU is **not** the main lever for this POC — the simulation is
CPU-bound (Java) and the policy is a small MLP; the 5070 Ti becomes important later for
vision/CNN-based policies.

## 0. Prerequisites (once)

```bat
winget install EclipseAdoptium.Temurin.17.JDK
winget install Python.Python.3.12
winget install Git.Git
```

Open a NEW terminal and verify: `java -version` (must be 17.x), `python --version` (3.12.x).
If `gradlew.bat` complains about JAVA_HOME, run
`setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot"` and open a new terminal.

## 1. Clone + build crforge

```bat
git clone https://github.com/voonhous/crforge
cd crforge
gradlew.bat build
```

(First build downloads Gradle + dependencies — a few minutes.)

## 2. Python env

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e "python/[train,jpype]"
```

## 3. Run A — self-play, single env (proven path, start here)

Terminal 1 (keep open):

```bat
gradlew.bat :gym-bridge:run
```

Terminal 2:

```bat
.venv\Scripts\activate
python python/examples/train_ppo.py --opponent self_play --timesteps 2000000 --save-path models/poc_selfplay --log-dir logs/poc_selfplay --eval-episodes 20
```

## 4. Run B — two different decks (Hog cycle vs Giant beatdown), self-play

Copy `poc_decks.py` from the team repo (`clash_royale_ai/training/poc_decks.py`) into the
crforge folder, then (bridge from step 3 still running):

```bat
python poc_decks.py
```

## 5. Parallel envs (when the basics work)

```bat
:: vs fixed bots — 8 parallel simulations (auto-launches its own servers)
python python/examples/train_ppo.py --num-envs 8 --opponent rule_based --timesteps 2000000 --save-path models/poc_parallel --log-dir logs/poc_parallel

:: self-play + parallelism (in-process JVM, single process)
python python/examples/train_ppo.py --jpype --num-envs 6 --opponent self_play --timesteps 3000000 --save-path models/poc_jpype --log-dir logs/poc_jpype
```

If `--jpype` acts up, fall back to step 3 (the verified path).

## 6. Watch a trained agent play

```bat
gradlew.bat :desktop:run --args="--ai-port 9876"
```

then start an evaluation from another terminal. SPACE pauses, +/- adjusts speed (0.25x–8x).

## Monitoring

`tensorboard --logdir logs` (or just read the printed lines: win/loss/draw + fps every 50 games).

## Tips / known quirks

- Run commands from the crforge repo root (scripts walk up to find `gradlew`).
- Allow java.exe on private networks if a firewall prompt appears (loopback only).
- Budget roughly 1 core per env + 1–2 cores for PPO: `--num-envs` ≈ cores / 2.
- `examples/run_episodes.py` is stale (expects dict obs) — ignore it.
- Spells can't hit towers through the current 10-zone action space (known limitation).
- Fidelity numbers + gaps: see `README.md` in this folder (measured vs real CR values).
