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
DEFAULT_DECK = ["knight", "archer", "fireball", "arrows", "giant", "musketeer", "minions", "valkyrie"]

RED_POOL = [HOG, GIANT, LOGB, XBOW, LAVA, YARD]


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
    if os.path.isfile(script):
        return script
    if os.path.isfile(legacy):
        return legacy
    print("Building gym-bridge distribution...")
    gradlew = os.path.join(project_root, "gradlew.bat" if os.name == "nt" else "gradlew")
    result = subprocess.run([gradlew, ":gym-bridge:installDist", "-q"],
                            cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    print("Build complete.")
    return script if os.path.isfile(script) else legacy


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


def launch_servers(script: str, base_port: int, n: int, max_restarts: int = 3):
    import tempfile

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

def build_callbacks(snapshot_path: str, snapshot_interval: int, total_steps: int):
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

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    outcome = info.get("game_outcome", "unknown")
                    self.win += outcome == "win"
                    self.loss += outcome == "loss"
                    self.draw += outcome == "draw"
                    self.rewards.append(info["episode"]["r"])
            done = self.win + self.loss + self.draw
            if done - self._last_log >= 25:
                self._last_log = done
                avg_r = sum(self.rewards) / max(1, len(self.rewards))
                pct = 100.0 * self.num_timesteps / total_steps
                print(f"[{self.num_timesteps}/{total_steps} ({pct:.0f}%) | ep {done}] "
                      f"win={100.0 * self.win / done:.0f}% loss={100.0 * self.loss / done:.0f}% "
                      f"draw={100.0 * self.draw / done:.0f}% | reward={avg_r:.1f}", flush=True)
            return True

    return [SnapshotCallback(), LogCallback()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-process self-play training for CRForge")
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--base-port", type=int, default=9890)
    parser.add_argument("--blue-deck", choices=["hog", "default"], default="hog")
    parser.add_argument("--red-pool", choices=["all", "hog"], default="all",
                        help="all = six archetypes (worker i gets RED_POOL[i%%6]); hog = mirror only")
    parser.add_argument("--opponent", choices=["selfplay", "rule_based", "random"], default="selfplay",
                        help="selfplay = snapshot-reloading self-play (default); rule_based/random = fixed opponents")
    parser.add_argument("--save", default="models/ppo_multi")
    parser.add_argument("--logdir", default="logs/ppo_multi")
    parser.add_argument("--snapshot-interval", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from crforge_gym import CRForgeEnv
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

    blue = HOG if args.blue_deck == "hog" else DEFAULT_DECK
    pool = RED_POOL if args.red_pool == "all" else [HOG]

    snapshot_dir = os.path.dirname(args.save) or "."
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, "selfplay_latest.zip")

    project_root = find_project_root()
    script = build_bridge_dist(project_root)
    launch_servers(script, args.base_port, args.num_envs)

    env_fns = [
        make_worker(args.base_port + i, blue, pool[i % len(pool)], snapshot_path, args.opponent)
        for i in range(args.num_envs)
    ]
    env = SubprocVecEnv(env_fns)

    model = MaskablePPO(
        "MlpPolicy", env, policy_kwargs={"net_arch": [512, 256]},
        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
        vf_coef=0.5, max_grad_norm=0.5, seed=args.seed, verbose=1,
        tensorboard_log=args.logdir,
    )

    print(f"\nTraining {args.steps} steps on {args.num_envs} parallel simulators "
          f"(blue: {'Hog 2.6' if args.blue_deck == 'hog' else 'default'}; "
          f"opponent: {args.opponent}; red pool: {len(pool)} deck(s))...")
    t0 = time.time()
    model.learn(total_timesteps=args.steps,
                callback=CallbackList(build_callbacks(snapshot_path, args.snapshot_interval, args.steps)))
    dt = time.time() - t0
    print(f"\nTraining done in {dt:.0f}s ({args.steps / dt:.0f} steps/s overall)")
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
