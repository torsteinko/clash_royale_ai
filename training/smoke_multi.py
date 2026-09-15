#!/usr/bin/env python3
"""Smoke test for the multi-game worker stack (1 JVM x 4 sessions, 1 worker process)."""
import os
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/crforge/python"))
os.chdir(os.path.expanduser("~/crforge"))

BR = os.path.expanduser("~/crforge/gym-bridge/build/install/gym-bridge/bin/gym-bridge")
PORTS = [9990, 9991, 9992, 9993]

from multi_selfplay_train import GIANT, HOG  # noqa: E402


def up(port):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    logf = open("/tmp/smoke_multi_bridge.log", "ab")
    proc = subprocess.Popen([BR, "9990", "4"], stdout=logf, stderr=logf)
    for port in PORTS:
        ok = False
        for _ in range(240):
            if up(port):
                ok = True
                break
            time.sleep(0.5)
        if not ok:
            print(f"FAIL: port {port} never up")
            proc.terminate()
            sys.exit(1)
    print("4 sessions up in one JVM")

    from crforge_gym.multi_vec_env import MultiBridgeVecEnv

    spec = {
        "ports": PORTS,
        "blue_decks": [HOG] * 4,
        "red_decks": [GIANT] * 4,
        "opponent": "random",
        "snapshot_path": None,
    }
    env = MultiBridgeVecEnv([spec], start_method="fork")
    print("spaces:", env.observation_space, env.action_space)

    obs = env.reset()
    assert obs.shape == (4, 1094) and obs.dtype == np.float32, obs.shape
    masks = np.stack(env.env_method("action_masks"))
    assert masks.shape == (4, 21) and masks.dtype == bool, (masks.shape, masks.dtype)
    print("reset OK:", obs.shape, "masks:", masks.shape, "sum(mask0):", int(masks[0].sum()))

    total = 0.0
    for i in range(60):
        actions = np.asarray([[1, 0, 0]] * 4, dtype=np.int64)
        obs, rew, dones, infos = env.step(actions)
        assert obs.shape == (4, 1094), obs.shape
        assert rew.shape == (4,) and dones.shape == (4,), (rew.shape, dones.shape)
        assert len(infos) == 4, len(infos)
        total += float(rew.sum())
        masks = np.stack(env.env_method("action_masks"))
        if i == 0:
            print("step1 OK: rew", rew.tolist(), "dones", dones.tolist())
            print("  infos[0] keys:", sorted(infos[0].keys()))
    print(f"60 steps OK, rew sum {total:.2f}")

    env.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
