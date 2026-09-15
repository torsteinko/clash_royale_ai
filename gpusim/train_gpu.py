"""M5.2 — headless PPO training on the GPU sim (no browser, no wall-clock).

MaskablePPO (sb3-contrib) on `GPUSimSB3VecEnv`: ONE batched lockstep sim drives
`--envs` parallel battles; the blue side is the policy, red is a scripted
opponent on the same fixed deck ("random" = uniform legal plays, "passive").

  python3 -m gpusim.train_gpu --steps 100000 --envs 16
  python3 -m gpusim.train_gpu --resume auto --steps 300000        # continues latest.zip
  python3 -m gpusim.train_gpu --device cuda --envs 1024           # on the 5070 Ti

Logging: progress.csv (always) + TensorBoard (if tensorboard is installed).
Checkpoints: <save-dir>/ckpt_<steps>.zip + <save-dir>/latest.zip every
--ckpt-every steps; `--resume auto` picks latest.zip up.

Scope note (M5): red is a scripted fixed-deck baseline, not a mirror policy —
true league self-play (red-perspective obs + snapshot league) is M6.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .sb3_env import DEFAULT_DECK_NAMES, GPUSimSB3VecEnv, deck_from_names

CSV_HEADER = ("timesteps,wall_s,steps_per_s,ep_rew_mean,ep_len_mean,"
              "blue_wins,red_wins,draws\n")


def make_monitor_callback(save_dir: str, ckpt_every: int = 10_000, csv_every: int = 2_048):
    """SB3 callback: checkpoints (latest.zip + ckpt_<steps>.zip) and CSV progress
    rows with fps, episode return/length (from `infos["episode"]`) and W/L/D
    (from the vecenv's cumulative `results` counters)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Monitor(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.save_dir = save_dir
            self.ckpt_every = ckpt_every
            self.csv_every = csv_every
            self._t0 = 0.0
            self._last_wall = 0.0
            self._last_steps = 0
            self._last_saved_step = -1
            self._last_logged_step = -1
            self._next_ckpt = ckpt_every
            self._next_csv = csv_every

        # ------------------------------------------------------------- helpers
        def _save(self, model):
            if model.num_timesteps == self._last_saved_step:  # already saved this step
                return
            self._last_saved_step = model.num_timesteps
            model.save(os.path.join(self.save_dir, "latest"))
            path = os.path.join(self.save_dir, f"ckpt_{model.num_timesteps}")
            model.save(path)
            print(f"[ckpt] {model.num_timesteps} steps -> {path}.zip", flush=True)

        def _log_row(self, model, venv):
            if model.num_timesteps == self._last_logged_step:  # nothing new since last row
                return
            wall = time.perf_counter() - self._t0
            d_wall = wall - self._last_wall
            d_steps = model.num_timesteps - self._last_steps
            fps = d_steps / d_wall if d_wall > 1e-6 else 0.0
            eps = model.ep_info_buffer
            ep_rew = float(np.mean([e["r"] for e in eps])) if eps else float("nan")
            ep_len = float(np.mean([e["l"] for e in eps])) if eps else float("nan")
            try:
                res = np.asarray(venv.get_attr("results", indices=[0])[0])
            except Exception:
                res = np.zeros(3, dtype=np.int64)
            self._last_wall, self._last_steps = wall, model.num_timesteps
            self._last_logged_step = model.num_timesteps
            with open(os.path.join(self.save_dir, "progress.csv"), "a") as f:
                f.write(f"{model.num_timesteps},{wall:.1f},{fps:.1f},{ep_rew:.3f},{ep_len:.1f},"
                        f"{res[0]},{res[1]},{res[2]}\n")
            print(f"[log ] step {model.num_timesteps:>8}  {fps:7.0f} steps/s  "
                  f"ep_rew_mean {ep_rew:+.2f}  ep_len {ep_len:.0f}  "
                  f"W/L/D {res[0]}/{res[1]}/{res[2]}", flush=True)

        # ------------------------------------------------------- callback hooks
        def _on_training_start(self):
            self._t0 = time.perf_counter()
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, "progress.csv")
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(CSV_HEADER)
            # after a resume (num_timesteps > 0) schedule the next rows from now
            base = int(self.model.num_timesteps)
            self._last_steps = base
            self._next_ckpt = base + self.ckpt_every
            self._next_csv = base + self.csv_every
            print(f"[info] logging to {path} (next ckpt at {self._next_ckpt} steps)", flush=True)

        def _on_step(self) -> bool:
            n = self.model.num_timesteps
            if self.ckpt_every and n >= self._next_ckpt:
                self._save(self.model)
                self._next_ckpt = n + self.ckpt_every
            if self.csv_every and n >= self._next_csv:
                self._log_row(self.model, self.training_env)
                self._next_csv = n + self.csv_every
            return True

        def _on_training_end(self):
            if self.ckpt_every:
                self._save(self.model)
            self._log_row(self.model, self.training_env)

    return Monitor()


def build_vec_env(args) -> GPUSimSB3VecEnv:
    deck = deck_from_names(args.data_dir, args.deck.split(",")) if args.deck \
        else deck_from_names(args.data_dir, DEFAULT_DECK_NAMES)
    return GPUSimSB3VecEnv(data_dir=args.data_dir, num_envs=args.envs, device=args.device,
                           deck=deck, opponent=args.opponent, seed=args.seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=100_000,
                    help="target TOTAL env steps (a resumed run continues to this number)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--data-dir", default=os.environ.get("GPUSIM_DATA", "fidelity/patched"))
    ap.add_argument("--deck", default="", help="comma-separated card names (default: project deck)")
    ap.add_argument("--opponent", default="random", choices=["random", "passive"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="runs/gpu_ppo")
    ap.add_argument("--ckpt-every", type=int, default=10_000)
    ap.add_argument("--csv-every", type=int, default=2_048)
    ap.add_argument("--resume", default="", help="path to a model zip, or 'auto' for <save-dir>/latest.zip")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--net", default="128,128")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO

    vec = build_vec_env(args)
    arch = [int(x) for x in args.net.split(",") if x]
    batch = min(args.batch_size, args.n_steps * args.envs)

    tb_dir = os.path.join(args.save_dir, "tb")
    try:
        import tensorboard  # noqa: F401

        tb_kw: dict = dict(tensorboard_log=tb_dir)
    except ImportError:
        tb_kw = {}
        print("[info] tensorboard not installed — progress.csv only", flush=True)

    resume_path = ""
    if args.resume == "auto":
        cand = os.path.join(args.save_dir, "latest.zip")
        if os.path.exists(cand):
            resume_path = cand
        else:
            print(f"[info] --resume auto: no {cand}, starting fresh", flush=True)
    elif args.resume:
        resume_path = args.resume

    if resume_path:
        model = MaskablePPO.load(resume_path, env=vec, **tb_kw)
        print(f"[info] resumed {resume_path} at {model.num_timesteps} steps", flush=True)
    else:
        model = MaskablePPO(
            "MlpPolicy", vec, seed=args.seed, verbose=0, **tb_kw,
            n_steps=args.n_steps, batch_size=batch, n_epochs=args.n_epochs,
            learning_rate=args.lr, gamma=0.997, gae_lambda=0.95, ent_coef=0.01,
            policy_kwargs=dict(net_arch=arch),
        )

    remaining = args.steps - model.num_timesteps
    if remaining <= 0:
        print(f"[info] already at target ({model.num_timesteps} >= {args.steps} steps)")
        return 0

    mon = make_monitor_callback(args.save_dir, ckpt_every=args.ckpt_every,
                                csv_every=args.csv_every)
    print(f"[run ] steps +{remaining} (target {args.steps}), envs {args.envs}, "
          f"device {args.device}, opponent {args.opponent}, deck {args.deck or 'default'}",
          flush=True)
    start = model.num_timesteps
    t0 = time.perf_counter()
    model.learn(total_timesteps=remaining, callback=mon,
                reset_num_timesteps=False, progress_bar=False)
    dt = time.perf_counter() - t0
    did = model.num_timesteps - start
    print(f"[done] {model.num_timesteps} steps total; this run: {did} steps in {dt:.1f}s "
          f"({did / dt:.1f} steps/s, {did * 15 / dt:.0f} sim ticks/s)")
    print(f"[done] model: {os.path.join(args.save_dir, 'latest.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
