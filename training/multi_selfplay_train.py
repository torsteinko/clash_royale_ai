#!/usr/bin/env python3
"""Multi-process self-play training for CRForge.

One trainer process + N worker processes, each with its OWN bridge server and its own
self-play opponent. Each worker's opponent reloads the trainer's latest snapshot from
disk when it changes, which combines true parallelism (N simulators on N cores) with
self-play — the combination `train_ppo.py` blocks for subprocess mode.

Blue plays a fixed deck (default: 2.6 Hog cycle). Worker i's opponent plays deck
`RED_POOL[i % len(RED_POOL)]`, so the agent learns one deck against several archetypes.

Run from the crforge repo root (bridge servers are launched from here):

    python multi_selfplay_train.py --num-envs 5 --steps 2000000

Tips: close any manual `:gym-bridge:run` server first if it uses the same port range
(default range starts at 9890). ~1 core per env + 1-2 for the trainer.
"""
import argparse
import atexit
import os
import subprocess
import sys
import time
from collections import deque

# All card ids verified against crforge's cards.json (240 entries, Sept 2026).
HOG = ["hogrider", "musketeer", "cannon", "skeletons", "icespirits", "fireball", "log", "zap"]
GIANT = ["giant", "witch", "minions", "musketeer", "fireball", "arrows", "valkyrie", "knight"]
LOGB = ["goblinbarrel", "princess", "rocket", "knight", "goblingang", "log", "infernotower", "icespirits"]
XBOW = ["xbow", "tesla", "log", "icespirits", "skeletons", "archer", "fireball", "knight"]
LAVA = ["lavahound", "balloon", "megaminion", "minions", "tombstone", "fireball", "zap", "arrows"]
YARD = ["graveyard", "poison", "babydragon", "tornado", "knight", "skeletons", "arrows", "icewizard"]
# Round-5 additions: more archetypes for the opponent pool + deck rotation.
RG = ["royalgiant", "fisherman", "hunter", "skeletons", "electrospirit", "lightning", "log", "ghost"]
GOLEM = ["golem", "babydragon", "darkwitch", "tornado", "lightning", "skeletons", "minions", "valkyrie"]
MINER = ["miner", "poison", "bats", "skeletons", "speargoblins", "valkyrie", "tesla", "log"]
MORTAR = ["mortar", "goblinbarrel", "knight", "bats", "log", "arrows", "princess", "goblins"]
BALLOON = ["balloon", "freeze", "bats", "skeletons", "valkyrie", "arrows", "tesla", "miner"]
PEKKA = ["pekka", "battleram", "ghost", "electrowizard", "poison", "zap", "skeletons", "minions"]
DEFAULT_DECK = ["knight", "archer", "fireball", "arrows", "giant", "musketeer", "minions", "valkyrie"]

RED_POOL = [HOG, GIANT, LOGB, XBOW, LAVA, YARD, RG, GOLEM, MINER, MORTAR, BALLOON, PEKKA]


# ---------------------------------------------------------------------------
# Server management (Windows + Linux)
# ---------------------------------------------------------------------------

def find_project_root() -> str:
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        path = start
        for _ in range(10):
            if os.path.isfile(os.path.join(path, "gradlew")):
                return path
            path = os.path.dirname(path)
    print("Error: cannot find the crforge project root (no gradlew found).")
    print("Run this script from the crforge repo root.")
    sys.exit(1)


def build_bridge_dist(project_root: str) -> str:
    bin_dir = os.path.join(project_root, "gym-bridge", "build", "install", "gym-bridge", "bin")
    script = os.path.join(bin_dir, "gym-bridge.bat" if os.name == "nt" else "gym-bridge")
    legacy = os.path.join(bin_dir, "gym-bridge")
    gradlew = os.path.join(project_root, "gradlew.bat" if os.name == "nt" else "gradlew")

    # Always refresh the distribution: gradle is a fast no-op when it is already up to
    # date, and this guarantees the servers match the current Java sources (a stale
    # install/ directory otherwise silently keeps an old build alive).
    print("Building gym-bridge distribution...")
    build_ok = False
    try:
        result = subprocess.run([gradlew, ":gym-bridge:installDist", "-q"],
                                cwd=project_root, capture_output=True, text=True, timeout=600)
        build_ok = result.returncode == 0
        if not build_ok:
            print(f"!! gradle refresh FAILED (exit {result.returncode}):")
            print(result.stderr[-2000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"!! gradle refresh failed ({exc})")
    if not build_ok:
        print("!! Continuing with the EXISTING build -- if it is stale, the startup check")
        print("!! below will abort with a clear error (no silent mid-training crash).")

    if os.path.isfile(script):
        return script
    if os.path.isfile(legacy):
        return legacy
    print("Error: gym-bridge distribution not found and could not be built.")
    sys.exit(1)


def _tcp_up(port: int, timeout: float = 0.5) -> bool:
    import socket

    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait_for_tcp(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _tcp_up(port):
            return True
        time.sleep(0.25)
    return False


def _handshake(port: int, timeout_s: float = 12.0) -> bool:
    """One clean ZMQ init/close handshake. Generous timeout: first init can be slow
    under load; a short timeout can leave the PAIR server stuck on a dead peer."""
    import json

    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    try:
        sock.connect(f"tcp://localhost:{port}")
        sock.send_string(json.dumps({
            "type": "init",
            "data": {"blueDeck": HOG, "redDeck": HOG, "level": 11, "ticksPerStep": 15},
        }))
        resp = sock.recv_string()
        if "init_ok" in resp:
            sock.send_string(json.dumps({"type": "close"}))
            try:
                sock.recv_string()
            except zmq.error.Again:
                pass
            return True
        return False
    except zmq.error.Again:
        return False
    finally:
        sock.close()
        ctx.term()


def wait_for_server(port: int, timeout: float = 60.0) -> bool:
    """TCP-poll until the port accepts, then a single clean ZMQ handshake."""
    if not _wait_for_tcp(port, timeout):
        return False
    time.sleep(0.3)
    return _handshake(port)


def probe_obs_size(port: int):
    """MANUAL DEBUG ONLY — do not call automatically at startup.

    The old startup use of this probe (init+reset+close on the first worker port)
    left the bridge server on that port occasionally wedged mid-teardown: the
    following worker's init handshake then hung forever and the whole run stalled.
    The stale-build guard now lives in the workers themselves — see
    crforge_gym.wrappers._check_binary_obs (checked on the first reset).
    """
    import json

    import numpy as np
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.RCVTIMEO, 15000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    try:
        sock.connect(f"tcp://localhost:{port}")
        sock.send_string(json.dumps({
            "type": "init",
            "data": {"blueDeck": HOG, "redDeck": HOG, "level": 11, "ticksPerStep": 15,
                     "binaryObs": True},
        }))
        if "init_ok" not in sock.recv_string():
            return None
        sock.send_string(json.dumps({"type": "reset"}))
        raw = sock.recv()
        return int(len(np.frombuffer(raw, dtype=np.float32)))
    except Exception:
        return None
    finally:
        try:
            sock.setsockopt(zmq.LINGER, 2000)
            sock.send_string(json.dumps({"type": "close"}))
            sock.close()
        except Exception:
            pass
        sock.close()
        ctx.term()


def launch_servers(script: str, base_port: int, n: int, max_restarts: int = 3):
    import tempfile

    # Refuse to start on top of orphaned servers: a crashed previous run can leave
    # JVMs holding the ports (on Windows, killing the cmd wrapper does NOT kill the
    # java child), and the startup health check would then silently "validate" the
    # OLD server instead of the freshly launched one.
    busy = [p for p in range(base_port, base_port + n) if _tcp_up(p)]
    if busy:
        print(f"!! Ports already in use: {busy}")
        print("!! Orphaned bridge server(s) from a previous run are still running.")
        print("!! Kill them first:   taskkill /IM java.exe /F   (Linux: pkill -f gym-bridge)")
        sys.exit(1)

    procs = {}

    def _start(port: int):
        cmd = ["cmd", "/c", script, str(port)] if os.name == "nt" else [script, str(port)]
        log_path = os.path.join(tempfile.gettempdir(), f"crforge_bridge_{port}.log")
        logf = open(log_path, "ab")
        procs[port] = subprocess.Popen(cmd, stdout=logf, stderr=logf)
        print(f"  server on port {port} (log: {log_path})")

    def _stop(port: int):
        p = procs.pop(port, None)
        if p is None:
            return
        if os.name == "nt":
            # Kill the whole process tree: terminating the cmd.exe wrapper alone
            # leaves the java child alive (the orphaned-server trap).
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            return
        try:
            p.terminate()
        except OSError:
            return
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    def cleanup():
        for port in list(procs):
            _stop(port)

    atexit.register(cleanup)
    for i in range(n):
        _start(base_port + i)
    print(f"Waiting for {n} bridge server(s) on ports {base_port}-{base_port + n - 1}...")
    for i in range(n):
        port = base_port + i
        ready = False
        for attempt in range(1, max_restarts + 1):
            if wait_for_server(port, timeout=90):
                ready = True
                break
            print(f"  port {port}: handshake failed (attempt {attempt}) — restarting server")
            _stop(port)
            _start(port)
        if not ready:
            print(f"Error: server on port {port} failed to start.")
            cleanup()
            sys.exit(1)
    print("All servers ready.")
    return procs


# ---------------------------------------------------------------------------
# Worker-side env + reloading self-play opponent
# ---------------------------------------------------------------------------

def make_worker(port: int, blue_deck: list, red_deck: list, snapshot_path: str,
                opponent_mode: str = "selfplay"):
    """Env factory (executed inside each SubprocVecEnv worker process)."""

    def _init():
        from crforge_gym import CRForgeEnv
        from crforge_gym.opponents import SelfPlayOpponent
        from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

        class ReloadingSelfPlayOpponent(SelfPlayOpponent):
            """Self-play opponent that reloads the trainer snapshot when its mtime changes."""

            def __init__(self):
                super().__init__(model=None)
                self._loaded_mtime = None
                self._last_check = 0.0
                self._announced = False

            def _maybe_reload(self):
                now = time.time()
                if now - self._last_check < 2.0:
                    return
                self._last_check = now
                try:
                    mtime = os.path.getmtime(snapshot_path)
                except OSError:
                    return
                if self._loaded_mtime is not None and mtime <= self._loaded_mtime:
                    return
                from sb3_contrib import MaskablePPO

                try:
                    self.model = MaskablePPO.load(snapshot_path, device="cpu")
                    self._loaded_mtime = mtime
                    if not self._announced:
                        self._announced = True
                        print(f"[worker port {port}] self-play opponent loaded snapshot", flush=True)
                except Exception:
                    pass  # file mid-write; retry on next check

            def act(self, obs_raw, player="red", obs_flat=None):
                self._maybe_reload()
                if self.model is None:
                    return None
                return super().act(obs_raw, player=player, obs_flat=obs_flat)

        if opponent_mode == "selfplay":
            opponent = ReloadingSelfPlayOpponent()
        else:
            opponent = opponent_mode  # built-in opponent name: "rule_based" / "random"

        env = CRForgeEnv(
            endpoint=f"tcp://localhost:{port}",
            ticks_per_step=15,
            opponent=opponent,
            binary_obs=True,
            blue_deck=blue_deck,
            red_deck=red_deck,
        )
        env = EpisodeStatsWrapper(env)
        env = ActionMaskedWrapper(env)
        return env

    return _init


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(snapshot_path: str, snapshot_interval: int, total_steps: int,
                    worker_decks: list[str] | None = None, eval_every: int = 0,
                    eval_endpoint: str = "", eval_targets: list | None = None,
                    eval_blue: list | None = None):
    from stable_baselines3.common.callbacks import BaseCallback

    class SnapshotCallback(BaseCallback):
        """Atomically save the model so workers pick up fresh opponents."""

        def __init__(self):
            super().__init__(0)
            self._last = 0

        def _snap(self):
            base = snapshot_path[:-4] if snapshot_path.endswith(".zip") else snapshot_path
            tmp = base + ".new.zip"
            self.model.save(tmp)
            os.replace(tmp, snapshot_path)

        def _on_training_start(self):
            self._snap()

        def _on_step(self):
            if self.num_timesteps - self._last >= snapshot_interval:
                self._last = self.num_timesteps
                self._snap()
                print(f"[snapshot] updated at {self.num_timesteps} steps", flush=True)
            return True

    class LogCallback(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.win = self.loss = self.draw = 0
            self.rewards = deque(maxlen=100)
            self._last_log = 0
            self.deck_names = worker_decks or []
            self.per_env = [[0, 0, 0] for _ in range(len(self.deck_names))]

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for i, info in enumerate(infos):
                if "episode" in info:
                    outcome = info.get("game_outcome", "unknown")
                    self.win += outcome == "win"
                    self.loss += outcome == "loss"
                    self.draw += outcome == "draw"
                    self.rewards.append(info["episode"]["r"])
                    if i < len(self.per_env):
                        self.per_env[i][0] += outcome == "win"
                        self.per_env[i][1] += outcome == "loss"
                        self.per_env[i][2] += outcome == "draw"
            done = self.win + self.loss + self.draw
            if done - self._last_log >= 25:
                self._last_log = done
                avg_r = sum(self.rewards) / max(1, len(self.rewards))
                pct = 100.0 * self.num_timesteps / total_steps
                print(f"[{self.num_timesteps}/{total_steps} ({pct:.0f}%) | ep {done}] "
                      f"win={100.0 * self.win / done:.0f}% loss={100.0 * self.loss / done:.0f}% "
                      f"draw={100.0 * self.draw / done:.0f}% | reward={avg_r:.1f}", flush=True)
                if self.per_env:
                    parts = []
                    for name, cnt in zip(self.deck_names, self.per_env):
                        n = cnt[0] + cnt[1] + cnt[2]
                        if n:
                            parts.append(f"{name} {100.0 * cnt[0] / n:.0f}%({n})")
                    if parts:
                        print("    per-deck (worker 0..N-1): " + " | ".join(parts), flush=True)
            return True

    class PeriodicEvalCallback(BaseCallback):
        """Every `eval_every` steps: 2 episodes vs the random bot per pool deck.

        Runs on a dedicated extra bridge server (base_port + num_envs), so it never
        collides with the training workers' sessions. The printed lines give a
        per-deck, absolute learning curve that self-play win% cannot provide.
        """

        def __init__(self):
            super().__init__(0)
            self._last = 0

        def _on_step(self):
            if not eval_every or self.num_timesteps - self._last < eval_every:
                return True
            self._last = self.num_timesteps
            from stable_baselines3.common.evaluation import evaluate_policy

            from crforge_gym import CRForgeEnv
            from crforge_gym.wrappers import ActionMaskedWrapper as _AMW
            from crforge_gym.wrappers import EpisodeStatsWrapper as _ESW

            parts = []
            for nm, dk in eval_targets or []:
                try:
                    el = CRForgeEnv(endpoint=eval_endpoint, ticks_per_step=15, opponent="random",
                                    binary_obs=True, blue_deck=eval_blue, red_deck=dk)
                    el = _ESW(el)
                    el = _AMW(el)
                    mean_r, _ = evaluate_policy(self.model, el, n_eval_episodes=2,
                                                deterministic=True)
                    parts.append(f"{nm} {mean_r:+.0f}")
                    el.close()
                    time.sleep(0.3)  # gentle gap between sessions (PAIR teardown)
                except Exception as exc:
                    parts.append(f"{nm} ERR({exc!r})")
            print(f"[eval@{self.num_timesteps} vs random] " + " | ".join(parts), flush=True)
            return True

    cbs = [SnapshotCallback(), LogCallback()]
    if eval_every:
        cbs.append(PeriodicEvalCallback())
    return cbs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-process self-play training for CRForge")
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--base-port", type=int, default=9890)
    parser.add_argument("--blue-deck", choices=["hog", "default"], default="hog",
                        help="(legacy) single blue deck selector; prefer --blue-pool")
    parser.add_argument("--blue-pool", default=None,
                        help="Blue deck(s) for the learner: one name, a comma list for per-worker "
                             "rotation (e.g. hog,yard,lava -> the net also learns to PLAY those "
                             "decks, not just face them), or 'all'. Default: 'hog'.")
    parser.add_argument("--red-pool", default="all",
                        help="all = six archetypes (worker i gets RED_POOL[i%%6]); hog/giant/logb/"
                             "xbow/lava/yard = single deck, or a comma list (e.g. hog,yard,lava)")
    parser.add_argument("--opponent", choices=["selfplay", "rule_based", "random"], default="selfplay",
                        help="selfplay = snapshot-reloading self-play (default); rule_based/random = fixed opponents")
    parser.add_argument("--save", default="models/ppo_multi")
    parser.add_argument("--logdir", default="logs/ppo_multi")
    parser.add_argument("--snapshot-interval", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Periodic in-training eval every N steps: 2 episodes vs the random bot "
                             "per pool deck (printed as [eval@N vs random]). 0 = off. Uses one "
                             "extra bridge server on port base+num_envs.")
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="PPO entropy bonus (higher = more exploration; default 0.005)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from --save checkpoint (or latest snapshot); "
                             "--steps is treated as the TOTAL step target")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from crforge_gym import CRForgeEnv
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

    deck_by_name = {"hog": HOG, "giant": GIANT, "logb": LOGB,
                    "xbow": XBOW, "lava": LAVA, "yard": YARD, "default": DEFAULT_DECK,
                    "rg": RG, "golem": GOLEM, "miner": MINER, "mortar": MORTAR,
                    "balloon": BALLOON, "pekka": PEKKA}

    def parse_pool(spec: str):
        if spec == "all":
            return RED_POOL
        if spec in deck_by_name:
            return [deck_by_name[spec]]
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if parts and all(p in deck_by_name for p in parts):
            return [deck_by_name[p] for p in parts]
        print(f"Error: unknown deck pool '{spec}'.")
        sys.exit(1)

    pool = parse_pool(args.red_pool)
    blue_spec = args.blue_pool or ("hog" if args.blue_deck == "hog" else "default")
    blue_pool = parse_pool(blue_spec)
    blue = blue_pool[0]

    snapshot_dir = os.path.dirname(args.save) or "."
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, "selfplay_latest.zip")

    project_root = find_project_root()
    script = build_bridge_dist(project_root)
    n_servers = args.num_envs + (1 if args.eval_every > 0 else 0)
    launch_servers(script, args.base_port, n_servers)
    eval_endpoint = f"tcp://localhost:{args.base_port + args.num_envs}"

    # Stale-build guard: moved into the workers (crforge_gym.wrappers._check_binary_obs
    # on the first reset). The old separate probe session here was the entry point of a
    # ZMQ PAIR teardown race: the probed server would occasionally wedge after
    # "Client requested close" and the following worker's init handshake hung forever,
    # stalling the whole run silently. Do NOT reintroduce an automated probe call.

    deck_name_by_tuple = {tuple(d): n for n, d in
                          (("hog", HOG), ("giant", GIANT), ("logb", LOGB),
                           ("xbow", XBOW), ("lava", LAVA), ("yard", YARD),
                           ("rg", RG), ("golem", GOLEM), ("miner", MINER),
                           ("mortar", MORTAR), ("balloon", BALLOON), ("pekka", PEKKA))}
    worker_labels = []
    for i in range(args.num_envs):
        bn = deck_name_by_tuple.get(tuple(blue_pool[i % len(blue_pool)]), f"b{i}")
        rn = deck_name_by_tuple.get(tuple(pool[i % len(pool)]), f"r{i}")
        worker_labels.append(bn if bn == rn else f"{bn}>{rn}")
    if len(blue_pool) > 1:
        print("  worker decks (blue>red): " + " | ".join(worker_labels))
    worker_decks = worker_labels

    env_fns = [
        make_worker(args.base_port + i, blue_pool[i % len(blue_pool)], pool[i % len(pool)],
                    snapshot_path, args.opponent)
        for i in range(args.num_envs)
    ]
    # NOTE: on some distros (Debian 13 / Python 3.13) the default start method is
    # "forkserver", which hangs SB3's SubprocVecEnv init. Pin "fork" explicitly.
    env = SubprocVecEnv(env_fns, start_method="fork")

    resume_path = None
    if args.resume:
        for cand in (args.save + ".zip", snapshot_path):
            if os.path.isfile(cand):
                resume_path = cand
                break

    if resume_path:
        model = MaskablePPO.load(resume_path, env=env, device="cpu")
        done = int(model.num_timesteps)
        remaining = max(0, args.steps - done)
        print(f"Resuming from {resume_path} at {done} steps -- training {remaining} more "
              f"(total target {args.steps}).")
        if remaining == 0:
            print("Target already reached; nothing to do.")
            env.close()
            sys.exit(0)
        train_steps = remaining
    else:
        if args.resume:
            print("--resume given but no checkpoint found; starting fresh.")
        model = MaskablePPO(
            "MlpPolicy", env, policy_kwargs={"net_arch": [512, 256]},
            learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
            vf_coef=0.5, max_grad_norm=0.5, seed=args.seed, verbose=1,
            tensorboard_log=args.logdir,
        )
        train_steps = args.steps

    blue_names = ",".join(dict.fromkeys(deck_name_by_tuple.get(tuple(d), "?") for d in blue_pool))
    red_names = ",".join(dict.fromkeys(deck_name_by_tuple.get(tuple(d), "?") for d in pool))
    print(f"\nTraining {train_steps} steps on {args.num_envs} parallel simulators "
          f"(blue: {blue_names}; opponent: {args.opponent}; red: {red_names})...")
    eval_targets = list(zip([deck_name_by_tuple.get(tuple(d), "?") for d in pool], pool))
    t0 = time.time()
    # NOTE: SB3 resets model.num_timesteps at the start of learn() unless told
    # otherwise. On resume we keep the counter (reset_num_timesteps=False) and pass
    # the ABSOLUTE step target, otherwise snapshots save "0 steps" and the next
    # resume/ETA is wrong.
    model.learn(total_timesteps=args.steps,
                callback=CallbackList(build_callbacks(snapshot_path, args.snapshot_interval,
                                                      args.steps, worker_labels,
                                                      args.eval_every, eval_endpoint,
                                                      eval_targets, blue_pool[0])),
                reset_num_timesteps=(resume_path is None))
    dt = time.time() - t0
    print(f"\nTraining done in {dt:.0f}s ({train_steps / dt:.0f} steps/s overall)")
    model.save(args.save)
    print(f"Model saved to {args.save}")

    env.close()
    for oppo in ["random", "rule_based"]:
        ev = CRForgeEnv(endpoint=f"tcp://localhost:{args.base_port}", ticks_per_step=15,
                        opponent=oppo, binary_obs=True, blue_deck=blue, red_deck=pool[0])
        ev = EpisodeStatsWrapper(ev)
        ev = ActionMaskedWrapper(ev)
        mean_r, std_r = evaluate_policy(model, ev, n_eval_episodes=args.eval_episodes)
        print(f"eval vs {oppo}: mean reward {mean_r:.2f} +/- {std_r:.2f}")
        ev.close()
    print("\nDone! (bridge servers are shut down automatically)")


if __name__ == "__main__":
    main()
