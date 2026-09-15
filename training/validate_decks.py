#!/usr/bin/env python3
"""Validate that candidate decks init and step in the sim (as blue AND as red).

Also tracks the entity-count field in the binary obs (index 1070) while force-
playing cards, to catch stubbed cards that never spawn anything.
"""
import os
import socket
import subprocess
import sys
import time

PORT = 9960
BRIDGE = os.path.expanduser("~/crforge/gym-bridge/build/install/gym-bridge/bin/gym-bridge")
HOG = ["hogrider", "musketeer", "cannon", "skeletons", "icespirits", "fireball", "log", "zap"]
NEW = {
    "rg": ["royalgiant", "fisherman", "hunter", "skeletons", "electrospirit", "lightning", "log", "ghost"],
    "golem": ["golem", "babydragon", "darkwitch", "tornado", "lightning", "skeletons", "minions", "valkyrie"],
    "miner": ["miner", "poison", "bats", "skeletons", "speargoblins", "valkyrie", "tesla", "log"],
    "mortar": ["mortar", "goblinbarrel", "knight", "bats", "log", "arrows", "princess", "goblins"],
    "balloon": ["balloon", "freeze", "bats", "skeletons", "valkyrie", "arrows", "tesla", "miner"],
    "pekka": ["pekka", "battleram", "ghost", "electrowizard", "poison", "zap", "skeletons", "minions"],
}


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


def exercise(env, steps=90):
    """Force-play cards and track max entity count from the binary obs."""
    import numpy as np

    obs, _ = env.reset()
    max_ent = 0
    for i in range(steps):
        if i % 6 == 0:
            a = np.array([1, i % 4, (i * 5) % 15], dtype=np.int64)
        else:
            a = np.array([0, 0, 0], dtype=np.int64)
        obs, _r, term, trunc, _info = env.step(a)
        try:
            max_ent = max(max_ent, int(obs[1070]))
        except Exception:
            pass
        if term or trunc:
            obs, _ = env.reset()
    return max_ent


def main():
    sys.path.insert(0, os.path.expanduser("~/crforge/python"))
    import numpy as np

    from crforge_gym import CRForgeEnv
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

    if tcp_up(PORT):
        print(f"port {PORT} busy")
        sys.exit(1)
    logf = open(f"/tmp/val_bridge_{PORT}.log", "ab")
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

    def run(label, bd, rd):
        try:
            env = CRForgeEnv(endpoint=f"tcp://localhost:{PORT}", opponent="random",
                             binary_obs=True, blue_deck=bd, red_deck=rd, ticks_per_step=15)
            env = EpisodeStatsWrapper(env)
            env = ActionMaskedWrapper(env)
            obs, _ = env.reset()
            obslen = obs.shape[0]
            max_ent = exercise(env)
            env.close()
            flag = "" if max_ent > 0 else "  <-- NO ENTITIES SPAWNED"
            print(f"{label:16s} OK  obs={obslen}  max_entities={max_ent}{flag}")
        except Exception as exc:
            print(f"{label:16s} FAIL {exc!r}")

    for name, deck in NEW.items():
        run(f"{name} as blue", deck, HOG)
        run(f"{name} as red", HOG, deck)
    print("VALIDATION-DONE")
    proc.terminate()


if __name__ == "__main__":
    main()
