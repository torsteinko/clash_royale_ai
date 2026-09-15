#!/usr/bin/env python3
"""M4.2b -- Java-side scenario runner (runs ON the training box).

Replays the fixed scenarios from `fidelity/m4_traces/<name>.scenario.json`
against the crforge gym-bridge (`ticks_per_step=1`) and emits per-tick traces
in the **m4-trace-v1** contract (docs/M4_TRACE_FORMAT.md), converted to the
gpusim coordinate frame (blue at the bottom: `y_trace = 32 - y_java`).

Java-side specifics (see docs/M4_TRACE_FORMAT.md and reports/M4.3):
  * the deck is SHUFFLED at reset (`Hand.java`), so hand slots are resolved by
    CARD NAME at play time (the plain-name assertion still runs on the gpusim
    side; both sides play the same card at the same tick/point);
  * decks are passed as card IDs (lowercase, `CardRegistry`);
  * actions with `at_tick=k` are submitted in step call number `k+1`;
  * record after every step (`reset()` -> record 0).

Run on clash-training:
    ~/venvs/clash/bin/python fidelity/m4_scenarios_java.py \
        --scenarios-dir ~/m4run/scenarios --out ~/m4run/out --seed 7 --passes 2

Then fetch: scp ~/m4run/out/*.jsonl back to fidelity/m4_traces/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

CRFORGE = os.path.expanduser("~/crforge")
sys.path.insert(0, os.path.join(CRFORGE, "python"))

BRIDGE_BIN = os.path.join(
    CRFORGE, "gym-bridge", "build", "install", "gym-bridge", "bin", "gym-bridge"
)

TICK_DT = 0.05
Y_FLIP = 32.0


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def tcp_up(port: int, timeout: float = 1.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def ensure_bridge(port: int, log_path: str, reuse: bool = False,
                  bridge_bin: str | None = None) -> subprocess.Popen | None:
    if tcp_up(port):
        # never join a foreign bridge (e.g. a training session) by accident
        if not reuse:
            raise RuntimeError(
                f"port {port} already has a listener — refusing to join it "
                f"(pass --reuse to allow, or pick a free port)")
        print(f"bridge already up on {port} (reused)", flush=True)
        return None
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "ab")
    proc = subprocess.Popen([bridge_bin or BRIDGE_BIN, str(port)], stdout=logf, stderr=logf)
    for _ in range(240):
        if tcp_up(port):
            print(f"bridge up on {port} (pid {proc.pid})", flush=True)
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"bridge never came up on port {port}")


def tower_slots(obs: dict) -> tuple[list, list]:
    th = [0.0] * 6
    ta = [0] * 6
    for side, key in ((0, "bluePlayer"), (1, "redPlayer")):
        for t in obs[key]["towers"]:
            if t["type"] == "crown":
                slot = side * 3
            else:
                slot = side * 3 + (1 if float(t["x"]) < 9.0 else 2)
            th[slot] = round(float(t["hp"]), 2)
            ta[slot] = int(bool(t["alive"]))
    return th, ta


def make_record(i: int, obs: dict) -> dict:
    th, ta = tower_slots(obs)
    units = []
    for e in obs["entities"]:
        # trace contract: units only (troops/buildings/heroes). The reference
        # also exposes towers, projectiles and area-effect zones (poison etc.)
        # as entities — they are not units.
        if e.get("entityType") in ("TOWER", "PROJECTILE", "SPELL"):
            continue
        units.append({
            "s": 0 if e["team"] == "BLUE" else 1,
            "c": norm(e["name"]),
            "x": round(float(e["x"]), 3),
            "y": round(Y_FLIP - float(e["y"]), 3),
            "hp": round(float(e["hp"]), 2),
        })
    units.sort(key=lambda u: (u["s"], u["x"], u["y"]))
    rec = {
        "i": i,
        "t": round(i * TICK_DT, 3),
        "el": [round(float(obs["bluePlayer"]["elixir"]), 3),
               round(float(obs["redPlayer"]["elixir"]), 3)],
        "th": th,
        "ta": ta,
        "cr": [int(obs["bluePlayer"]["crowns"]), int(obs["redPlayer"]["crowns"])],
        "go": 0,
        "w": -1,
    }
    if units:
        rec["u"] = units
    return rec


def run_scenario(client, manifest: dict, seed: int, duration: int) -> tuple[str, dict]:
    from collections import defaultdict

    name = manifest["name"]
    decks = manifest["decks"]
    blue_deck = [norm(c) for c in decks["blue"]]
    red_deck = [norm(c) for c in decks["red"]]

    client.init(blue_deck=blue_deck, red_deck=red_deck,
                level=manifest["level"], ticks_per_step=1, seed=seed)
    obs = client.reset(seed=seed)

    records = [make_record(0, obs)]
    by_tick: dict[int, list] = defaultdict(list)
    for a in manifest["actions"]:
        by_tick[int(a["at_tick"])].append(a)

    evidence: list[dict] = []
    for k in range(duration):
        el_before = {"blue": float(obs["bluePlayer"]["elixir"]),
                     "red": float(obs["redPlayer"]["elixir"])}
        acts: dict[str, dict | None] = {"blue": None, "red": None}
        meta: dict[str, dict | None] = {"blue": None, "red": None}
        for a in by_tick.get(k, []):
            side = "blue" if int(a["side"]) == 0 else "red"
            hand = obs[side + "Player"]["hand"]
            want = norm(a["card"])
            slot = next((i for i, h in enumerate(hand)
                         if norm(h.get("name", "")) == want), None)
            if slot is None:
                evidence.append({"tick": k, "side": side, "card": a["card"],
                                 "ok": False, "reason": "card-not-in-hand",
                                 "hand": [h.get("name") for h in hand]})
                continue
            acts[side] = {"handIndex": slot, "x": float(a["x"]),
                          "y": round(Y_FLIP - float(a["y"]), 3)}
            meta[side] = a
            if int(a["slot"]) != slot:
                evidence.append({"tick": k, "side": side, "card": a["card"],
                                 "slots_differ": [int(a["slot"]), slot]})
        res = client.step(blue_action=acts["blue"], red_action=acts["red"])
        obs2 = res["observation"]
        for side in ("blue", "red"):
            if meta[side] is None:
                continue
            delta = float(obs2[side + "Player"]["elixir"]) - el_before[side]
            # ~0.0179 elixir regen per tick; any spend is >= 1.0
            spent = delta < 0.01
            evidence.append({"tick": k, "side": side, "card": meta[side]["card"],
                             "slot": acts[side]["handIndex"], "x": acts[side]["x"],
                             "y": acts[side]["y"], "elixir_delta": round(delta, 4),
                             "spent": spent, "ok": bool(spent)})
        records.append(make_record(k + 1, obs2))
        obs = obs2

    client.close()
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    attempted = (sum(1 for e in evidence if "slot" in e)
                 + sum(1 for e in evidence if e.get("reason") == "card-not-in-hand"))
    run_info = {
        "format": "m4-java-run-v1",
        "name": name,
        "seed": seed,
        "ticks_per_step": 1,
        "duration_ticks": duration,
        "bridge": "crforge gym-bridge (clash-training)",
        "actions_ok": sum(1 for e in evidence if e.get("ok") is True),
        "actions_total": attempted,
        "evidence": evidence,
    }
    return text, run_info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-dir", default=os.path.expanduser("~/m4run/scenarios"))
    ap.add_argument("--out", default=os.path.expanduser("~/m4run/out"))
    ap.add_argument("--port", type=int, default=9971)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--reuse", action="store_true",
                    help="allow reusing an existing bridge on --port (default: refuse)")
    ap.add_argument("--bridge-bin", default=None,
                    help="bridge launcher (default: the gradle install script); use the "
                         "patched-data wrapper to run the reference engine with the M0 spec data")
    ap.add_argument("--duration", type=int, default=0, help="override manifest duration")
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    args = ap.parse_args(argv)

    from crforge_gym.bridge import BridgeClient  # noqa: E402

    os.makedirs(args.out, exist_ok=True)
    proc = ensure_bridge(args.port, os.path.join(args.out, f"bridge_{args.port}.log"),
                         reuse=args.reuse, bridge_bin=args.bridge_bin)
    endpoint = f"tcp://localhost:{args.port}"

    names = [p[:-len(".scenario.json")] for p in sorted(os.listdir(args.scenarios_dir))
             if p.endswith(".scenario.json")]
    if args.only:
        want = {s for s in args.only.split(",") if s}
        names = [n for n in names if n in want]
    print(f"scenarios: {names}", flush=True)

    results = {}
    t0 = time.time()
    for name in names:
        manifest = json.load(open(os.path.join(args.scenarios_dir, f"{name}.scenario.json")))
        duration = args.duration or int(manifest["duration_ticks"])
        texts = []
        run_info = None
        for p in range(args.passes):
            client = BridgeClient(endpoint, binary_obs=False)
            text, run_info = run_scenario(client, manifest, args.seed, duration)
            texts.append(text)
            det = "" if p == 0 else (" MATCH" if texts[p] == texts[0] else " DIFF!")
            print(f"  {name} pass {p+1}/{args.passes}: {len(text.splitlines())} records{det}",
                  flush=True)
        deterministic = all(t == texts[0] for t in texts)
        (open(os.path.join(args.out, f"{name}.java.jsonl"), "w").write(texts[0]))
        run_info["deterministic"] = deterministic
        run_info["passes"] = args.passes
        json.dump(run_info, open(os.path.join(args.out, f"{name}.java.run.json"), "w"), indent=1)
        results[name] = {"deterministic": deterministic,
                         "actions_ok": run_info["actions_ok"],
                         "actions_total": run_info["actions_total"]}
    dt = time.time() - t0
    print(f"done in {dt:.0f}s: {json.dumps(results, indent=1)}", flush=True)
    if proc is not None:
        proc.terminate()
        print(f"bridge pid {proc.pid} terminated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
