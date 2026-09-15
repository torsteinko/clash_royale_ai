#!/usr/bin/env python3
"""Smoke: worker-shared selfplay model (expect exactly ONE load line for 4 games)."""
import os
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/crforge/python"))
sys.path.insert(0, os.path.expanduser("~/crforge"))
os.chdir(os.path.expanduser("~/crforge"))

BR = os.path.expanduser("~/crforge/gym-bridge/build/install/gym-bridge/bin/gym-bridge")
PORTS = [9990, 9991, 9992, 9993]
SNAP = os.path.expanduser("~/runs/pool12/models/selfplay_latest.zip")

from multi_selfplay_train import GIANT, HOG  # noqa: E402


def up(p):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", p))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    logf = open("/tmp/shared_test_bridge.log", "ab")
    proc = subprocess.Popen([BR, "9990", "4"], stdout=logf, stderr=logf)
    for port in PORTS:
        for _ in range(240):
            if up(port):
                break
            time.sleep(0.5)
    from crforge_gym.multi_vec_env import MultiBridgeVecEnv

    spec = {"ports": PORTS, "blue_decks": [HOG] * 4, "red_decks": [GIANT] * 4,
            "opponent": "selfplay", "snapshot_path": SNAP}
    env = MultiBridgeVecEnv([spec], start_method="fork")
    obs = env.reset()
    assert obs.shape == (4, 1094)
    time.sleep(3)  # let the 2s mtime check fire
    for i in range(80):
        obs, rew, dones, infos = env.step(np.asarray([[1, 0, 0]] * 4, dtype=np.int64))
    print("80 steps OK, last rew:", np.round(rew, 2).tolist())
    masks = np.stack(env.env_method("action_masks"))
    assert masks.shape == (4, 21)
    env.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("SHARED-TEST OK")


if __name__ == "__main__":
    main()
