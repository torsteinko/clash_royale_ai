#!/usr/bin/env python3
"""Record JSON replays of model games for the dashboard.

Uses a lockstep pair of sessions: env A (binary) lets the trained model play as
blue against a rule-based red whose actions are LOGGED; env B (JSON mode, same
seed, noop opponent) replays the exact same action sequence and provides the
rich raw observation (card names, hp, positions) for the replay file.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/crforge/python"))
sys.path.insert(0, os.path.expanduser("~/crforge"))

BRIDGE = os.path.expanduser("~/crforge/gym-bridge/build/install/gym-bridge/bin/gym-bridge")


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


class LoggingOpp:
    """Wraps a real opponent; records every red action it takes."""

    def __init__(self, inner):
        self.inner = inner
        self.log = []

    def act(self, obs_raw, obs_flat=None, **kw):
        a = self.inner.act(obs_raw, player="red", obs_flat=obs_flat)
        self.log.append(a)
        return a


class QueueOpp:
    """Feeds pre-recorded red actions back into the replay session."""

    def __init__(self):
        self.queue = []

    def act(self, obs_raw, obs_flat=None, **kw):
        if self.queue:
            return self.queue.pop(0)
        return None


def extract_frame(raw, t):
    """Compact frame: entities as [id, x, y, hp]; registry built separately."""
    ents = []
    for e in raw.get("entities", []):
        try:
            ents.append([e["id"], round(float(e["x"]), 2), round(float(e["y"]), 2),
                         int(float(e["hp"]))])
        except Exception:
            continue
    bl = raw.get("bluePlayer", {}) or {}
    rl = raw.get("redPlayer", {}) or {}
    return {
        "t": t,
        "gt": round(float(raw.get("gameTimeSeconds", 0)), 1),
        "be": int(bl.get("elixir", 0)),
        "re": int(rl.get("elixir", 0)),
        "ents": ents,
    }


def register_entities(raw, cards):
    for e in raw.get("entities", []):
        eid = e.get("id")
        if eid and eid not in cards:
            cards[eid] = {
                "name": e.get("name", "?"),
                "team": 0 if e.get("team") == "BLUE" else 1,
                "type": e.get("entityType", "TROOP"),
                "mov": e.get("movementType", "GROUND"),
                "maxHp": int(float(e.get("maxHp", 1) or 1)),
            }


def record_game(model, blue_deck, red_deck, blue_name, red_name, opp_name, seed, out_dir):
    import numpy as np

    from crforge_gym import CRForgeEnv
    from crforge_gym.opponents import RuleBasedOpponent
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper

    inner = RuleBasedOpponent(rng=np.random.default_rng(seed))
    logopp = LoggingOpp(inner)
    qopp = QueueOpp()

    envA = CRForgeEnv(endpoint="tcp://localhost:9980", opponent=logopp, binary_obs=True,
                      blue_deck=blue_deck, red_deck=red_deck, ticks_per_step=15)
    envA = EpisodeStatsWrapper(envA)
    envA = ActionMaskedWrapper(envA)
    envB = CRForgeEnv(endpoint="tcp://localhost:9981", opponent=qopp, binary_obs=False,
                      blue_deck=blue_deck, red_deck=red_deck, ticks_per_step=15)
    envB = EpisodeStatsWrapper(envB)

    obsA, _ = envA.reset(seed=seed)
    envB.reset(seed=seed)

    cards = {}
    frames = []
    acts = []
    register_entities(envB.unwrapped._last_obs_raw, cards)
    frames.append(extract_frame(envB.unwrapped._last_obs_raw, 0))

    done = False
    result = None
    t = 0
    while not done and t < 900:
        mask = envA.action_masks()
        action, _ = model.predict(obsA, deterministic=True, action_masks=mask)
        action = np.asarray(action, dtype=np.int64)

        # Label blue's action with the card name from B's own hand (same state).
        act_label = None
        if int(action[0]) == 1:
            braw = envB.unwrapped._last_obs_raw or {}
            hand = (braw.get("bluePlayer", {}) or {}).get("hand", [])
            hi = int(action[1]) % 4
            if hi < len(hand):
                act_label = hand[hi].get("name", "?")
        # zone -> approx tile position for the viewer ping
        from crforge_gym.env import PLACEMENT_ZONES
        zx, zy = PLACEMENT_ZONES[int(action[2])] if int(action[2]) < len(PLACEMENT_ZONES) else (9.0, 16.0)

        obsA, _r, termA, truncA, infoA = envA.step(action)
        red_a = logopp.log.pop(0) if logopp.log else None
        qopp.queue.append(red_a)
        _obsB, _rb, termB, truncB, infoB = envB.step(action)

        t += 1
        register_entities(envB.unwrapped._last_obs_raw, cards)
        frame = extract_frame(envB.unwrapped._last_obs_raw, t)
        if act_label:
            frame["act"] = [act_label, round(float(zx), 2), round(float(zy), 2)]
            acts.append([t, act_label, round(float(zx), 2), round(float(zy), 2)])
        frames.append(frame)

        done = bool(termA or truncA)
        if done:
            result = infoA.get("game_outcome", "?")
            resultB = infoB.get("game_outcome", "?")
            if result != resultB:
                print(f"  !! outcome mismatch A={result} B={resultB} (replay may drift)", flush=True)
            break

    envA.close()
    envB.close()
    time.sleep(0.3)

    ts = time.strftime("%m%d-%H%M%S")
    fname = f"{ts}_{blue_name}-vs-{red_name}_{opp_name}_s{seed}.json"
    meta = {
        "blue_deck": blue_name,
        "red_deck": red_name,
        "opponent": opp_name,
        "seed": seed,
        "result": result,
        "steps": len(frames),
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cards": cards,
    }
    out = os.path.join(out_dir, fname)
    with open(out, "w") as fh:
        json.dump({"meta": meta, "frames": frames, "acts": acts}, fh, separators=(",", ":"))
    print(f"  saved {fname}  result={result}  frames={len(frames)}", flush=True)
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/runs/pool12/models/selfplay_latest.zip"))
    parser.add_argument("--out", default=os.path.expanduser("~/replays"))
    parser.add_argument("--games", type=int, default=0, help="cap total games (0 = all)")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    from multi_selfplay_train import (BALLOON, GIANT, GOLEM, HOG, LAVA, LOGB, MINER, MORTAR,
                                      PEKKA, RG, XBOW, YARD)
    decks = {"hog": HOG, "giant": GIANT, "logb": LOGB, "xbow": XBOW, "lava": LAVA, "yard": YARD,
             "rg": RG, "golem": GOLEM, "miner": MINER, "mortar": MORTAR, "balloon": BALLOON,
             "pekka": PEKKA}

    os.makedirs(args.out, exist_ok=True)

    logf = open("/tmp/replay_bridges.log", "ab")
    procs = [subprocess.Popen([BRIDGE, str(p)], stdout=logf, stderr=logf) for p in (9980, 9981)]
    for p in (9980, 9981):
        ok = False
        for _ in range(240):
            if tcp_up(p):
                ok = True
                break
            time.sleep(0.5)
        if not ok:
            print(f"bridge {p} never came up")
            for pr in procs:
                pr.terminate()
            sys.exit(1)
    print("bridges up (9980, 9981)", flush=True)

    model = MaskablePPO.load(args.model, device="cpu")
    print(f"model loaded: {args.model}", flush=True)

    plan = [("hog", d, "rule_based") for d in decks if d != "hog"]
    plan += [(d, "hog", "rule_based") for d in ("giant", "yard", "xbow", "golem", "mortar", "balloon")]
    if args.games:
        plan = plan[: args.games]

    made = 0
    for i, (b, r, opp) in enumerate(plan):
        seed = 7000 + i
        try:
            record_game(model, decks[b], decks[r], b, r, opp, seed, args.out)
            made += 1
        except Exception as exc:
            print(f"  game {b}-vs-{r} FAILED: {exc!r}", flush=True)
    print(f"REPLAY-RUN-DONE made={made}/{len(plan)}", flush=True)
    for pr in procs:
        pr.terminate()


if __name__ == "__main__":
    main()
