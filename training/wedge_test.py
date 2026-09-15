#!/usr/bin/env python3
"""Wedge regression test: rapid connect -> init -> reset -> steps -> close cycles
with rotating decks (mimics the periodic-eval pattern and the old probe race).
A healthy server handles all cycles in seconds; a wedged one hangs forever.
"""
import os
import socket
import subprocess
import sys
import time

PORT = 9970
BRIDGE = os.path.expanduser("~/crforge/gym-bridge/build/install/gym-bridge/bin/gym-bridge")
HOG = ["hogrider", "musketeer", "cannon", "skeletons", "icespirits", "fireball", "log", "zap"]
GIANT = ["giant", "witch", "minions", "musketeer", "fireball", "arrows", "valkyrie", "knight"]
YARD = ["graveyard", "poison", "babydragon", "tornado", "knight", "skeletons", "arrows", "icewizard"]
GOLEM = ["golem", "babydragon", "darkwitch", "tornado", "lightning", "skeletons", "minions", "valkyrie"]
DECKS = [HOG, GIANT, YARD, GOLEM]
CYCLES = 24


def tcp_up(p, t=1.0):
    s = socket.socket()
    s.settimeout(t)
    try:
        s.connect(("127.0.0.1", p))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    sys.path.insert(0, os.path.expanduser("~/crforge/python"))
    import numpy as np

    from crforge_gym import CRForgeEnv
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

    if tcp_up(PORT):
        print(f"port {PORT} busy")
        sys.exit(1)
    logf = open(f"/tmp/wedge_test_bridge_{PORT}.log", "ab")
    proc = subprocess.Popen([BRIDGE, str(PORT)], stdout=logf, stderr=logf)
    ok = False
    for _ in range(240):
        if tcp_up(PORT):
            ok = True
            break
        time.sleep(0.5)
    if not ok:
        print("bridge never came up")
        proc.terminate()
        sys.exit(1)
    print(f"bridge up; running {CYCLES} connect/close cycles")
    worst = 0.0
    for cycle in range(CYCLES):
        red = DECKS[cycle % len(DECKS)]
        t0 = time.time()
        try:
            env = CRForgeEnv(endpoint=f"tcp://localhost:{PORT}", opponent="random",
                             binary_obs=True, blue_deck=HOG, red_deck=red, ticks_per_step=15)
            env = EpisodeStatsWrapper(env)
            env = ActionMaskedWrapper(env)
            env.reset()
            for i in range(45):
                a = (np.array([1, i % 4, (i * 5) % 15], dtype=np.int64) if i % 6 == 0
                     else np.array([0, 0, 0], dtype=np.int64))
                obs, _r, term, trunc, _info = env.step(a)
                if term or trunc:
                    break
            env.close()
            dt = time.time() - t0
            worst = max(worst, dt)
            print(f"cycle {cycle:2d}  {dt:5.2f}s")
        except Exception as exc:
            print(f"cycle {cycle:2d}  ERROR {exc!r}")
    print(f"WEDGE-TEST-DONE  worst={worst:.2f}s")
    proc.terminate()


if __name__ == "__main__":
    main()
