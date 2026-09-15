"""M5.1b smoke run: MaskablePPO on the GPU sim, CPU, ~2000 steps.

Proves the SB3 integration end-to-end: DummyVecEnv of GPUSimGymEnv battles,
masked rollouts, greedy predict with masks, optional checkpoint save.

Run:  python3 -m gpusim.train_sb3_smoke [--steps 2000] [--envs 2] [--seed 0] [--save PATH]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .sb3_env import make_sb3_vec_env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000, help="total env steps (~15 sim ticks each)")
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="optional path to save the model zip")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO

    vec = make_sb3_vec_env(n_envs=args.envs, seed=args.seed)
    model = MaskablePPO(
        "MlpPolicy", vec, seed=args.seed, verbose=0,
        n_steps=256, batch_size=64, n_epochs=4,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    dt = time.perf_counter() - t0

    # --- post-run verification -------------------------------------------------
    obs = vec.reset()
    masks = np.stack(vec.env_method("action_masks"))
    acts, _ = model.predict(obs, action_masks=masks, deterministic=True)
    legal = all(bool(masks[i][int(acts[i])]) for i in range(args.envs))
    ep_returns = [ep["r"] for ep in model.ep_info_buffer] if model.ep_info_buffer else []

    print(f"MaskablePPO smoke: {model.num_timesteps} env steps in {dt:.1f}s "
          f"({model.num_timesteps / dt:.1f} steps/s, {model.num_timesteps * 15 / dt:.0f} sim ticks/s)")
    print(f"episodes finished: {len(ep_returns)}"
          + (f", mean return {np.mean(ep_returns):+.2f}" if ep_returns else ""))
    print(f"greedy actions legal under masks: {legal}")
    if args.save:
        model.save(args.save)
        print(f"saved: {args.save}.zip")

    ok = legal and model.num_timesteps >= args.steps
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
