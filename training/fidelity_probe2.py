#!/usr/bin/env python3
"""CRForge fidelity probe #2 — elixir phases, fireball, invalid actions, full 5:00 clock."""
import json
import time

import numpy as np

from crforge_gym import CRForgeEnv

REPORT = "/tmp/crforge_fidelity_report2.txt"
lines = []


def log(s=""):
    lines.append(str(s))
    print(s, flush=True)


env = CRForgeEnv(endpoint="tcp://localhost:9876", opponent="noop",
                 binary_obs=False, ticks_per_step=1, level=11)
obs, info = env.reset(seed=7)
raw = env.unwrapped._last_obs_raw


def gt():
    return float(raw.get("gameTimeSeconds", 0.0))


def mult():
    return float(raw.get("elixirMultiplier", 1.0))


def elx():
    return float(raw.get("bluePlayer", {}).get("elixir", 0.0))


def hand():
    return raw.get("bluePlayer", {}).get("hand", [])


def ents():
    return raw.get("entities", [])


def tw(side):
    return raw.get(side + "Player", {}).get("towers", [])


def princess_hp(side, left=True):
    for t in tw(side):
        if t.get("type") == "princess" and ((t.get("x", 9) < 9) == left):
            return t["hp"]
    return None


log("=" * 60)
log("probe2: seed 7, noop, tps=1 — phases/spells/invalid/full clock")
log(f"start hand={[c.get('id') for c in hand()]} elixir={elx():.1f}")
series = []   # (t, elx, mult, rpl, rpr)
spell_events = []  # (t, zone, rpl_before, rpr_before)
invalid_tests = []
plays_log = []


def step(a):
    global obs, raw, term, trunc
    obs, r, term, trunc, inf = env.step(np.asarray(a, dtype=np.int64))
    raw = env.unwrapped._last_obs_raw
    return r, term, trunc, inf


def find_affordable_troop():
    for i, c in enumerate(hand()):
        if c.get("type") == "TROOP" and elx() >= float(c.get("cost", 99)):
            return i, c
    return None, None


done = False
knight_done = False
drained_120 = False
drained_240 = False
invalid_done = False
fireball_1 = False
fireball_2 = False
spawn_check = {}  # count blue entities after each test

while not done:
    t = gt()
    a = np.array([0, 0, 0], dtype=np.int64)

    # 1) knight at 2s
    if not knight_done and t >= 2.0:
        i, c = find_affordable_troop()
        if i is not None:
            before = elx()
            a = np.array([1, i, 0], dtype=np.int64)
            r, term, trunc, inf = step(a)
            plays_log.append((t, c.get("id"), 0, before, elx(), bool(inf.get("action_failed"))))
            knight_done = True
            continue

    # 2) fireball at enemy-left princess zone (7), then zone 8
    if knight_done and not fireball_1 and t >= 30.0:
        fb = next(((i, c) for i, c in enumerate(hand()) if c.get("id") == "fireball"), None)
        if fb and elx() >= fb[1].get("cost", 4):
            i, c = fb
            before_pl, before_pr = princess_hp("red", True), princess_hp("red", False)
            a = np.array([1, i, 7], dtype=np.int64)
            r, term, trunc, inf = step(a)
            spell_events.append((t, 7, before_pl, before_pr, bool(inf.get("action_failed"))))
            fireball_1 = True
            continue
    if fireball_1 and not fireball_2 and t >= 36.0:
        fb = next(((i, c) for i, c in enumerate(hand()) if c.get("id") == "fireball"), None)
        if fb and elx() >= fb[1].get("cost", 4):
            i, c = fb
            before_pl, before_pr = princess_hp("red", True), princess_hp("red", False)
            a = np.array([1, i, 8], dtype=np.int64)
            r, term, trunc, inf = step(a)
            spell_events.append((t, 8, before_pl, before_pr, bool(inf.get("action_failed"))))
            fireball_2 = True
            continue

    # 3) invalid action test at ~t=45: troop to enemy-half zone 8 (should fail cleanly)
    if not invalid_done and t >= 45.0:
        i, c = find_affordable_troop()
        if i is not None:
            before = elx()
            n_before = len([e for e in ents() if e.get("team") == "BLUE" and e.get("entityType") == "TROOP"])
            a = np.array([1, i, 8], dtype=np.int64)
            r, term, trunc, inf = step(a)
            n_after = len([e for e in ents() if e.get("team") == "BLUE" and e.get("entityType") == "TROOP"])
            invalid_tests.append((t, c.get("id"), before, elx(), bool(inf.get("action_failed")), n_before, n_after))
            invalid_done = True
            continue

    # 4) drain just before 120s and 240s to keep elixir below cap for slope sampling
    if not drained_120 and t >= 117.0:
        i, c = find_affordable_troop()
        if i is not None:
            a = np.array([1, i, 0], dtype=np.int64)
            r, term, trunc, inf = step(a)
            plays_log.append((t, c.get("id"), 0, elx(), elx(), bool(inf.get("action_failed"))))
            drained_120 = True
            continue
    if not drained_240 and t >= 237.0:
        i, c = find_affordable_troop()
        if i is not None:
            a = np.array([1, i, 0], dtype=np.int64)
            r, term, trunc, inf = step(a)
            plays_log.append((t, c.get("id"), 0, elx(), elx(), bool(inf.get("action_failed"))))
            drained_240 = True
            continue

    r, term, trunc, inf = step(a)
    done = bool(term) or bool(trunc)
    series.append((gt(), elx(), mult(), princess_hp("red", True), princess_hp("red", False)))

log(f"match: end t={gt():.1f} over={raw.get('isOvertime')} term={term} trunc={trunc} "
    f"crowns b={raw.get('bluePlayer',{}).get('crowns')} r={raw.get('redPlayer',{}).get('crowns')}")
log("")

changes, prev = [], None
for (t, _e, m, *_x) in series:
    if m != prev:
        changes.append((round(t, 2), m))
        prev = m
log(f"multiplier timeline: {changes}   (expect 1 -> 2 @120s -> 3 @240s)")
log("")


def slope(t0, t1):
    seg = [(t, e) for (t, e, *_x) in series if t0 <= t <= t1 and 0.2 <= e <= 9.8]
    if len(seg) < 3:
        return None
    ts = np.array([s[0] for s in seg]); es = np.array([s[1] for s in seg])
    return float(np.polyfit(ts, es, 1)[0])


for t0, t1, label in [(5, 17, "1x early"), (121, 136, "after 120s (expect 2x)"),
                      (241, 256, "after 240s (expect 3x)")]:
    s = slope(t0, t1)
    log(f"regen {label:24s}: " + (f"{s:.4f}/s -> 1 per {1/s:.2f}s" if s else "n/a (capped)"))

log("")
log("---- FIREBALL (casts aimed at enemy LEFT princess zone7, then RIGHT zone8) ----")
for (t, zone, bpl, bpr, failed) in spell_events:
    log(f"cast t={t:.2f} zone={zone} failed={failed} | red princess-L {bpl}->{princess_hp('red', True)}  "
        f"princess-R {bpr}->{princess_hp('red', False)}")
# post-cast deltas from series
for (t, zone, bpl, bpr, failed) in spell_events:
    w = [(tt, pl, pr) for (tt, _e, _m, pl, pr) in series if t <= tt <= t + 2.5]
    if len(w) >= 2:
        log(f"  within 2.5s: princess-L -{w[0][1] - w[-1][1]}  princess-R -{w[0][2] - w[-1][2]}")

log("")
log("---- INVALID ACTION (troop -> enemy-half zone 8) ----")
for (t, cid, before, after, failed, nb, na) in invalid_tests:
    log(f"t={t:.2f} played {cid} to zone8: elixir {before:.2f}->{after:.2f} spent={before-after:.2f} "
        f"failed={failed} blue_troops {nb}->{na} (expect: failed=True, spent=0, no new troop)")

log("")
log("---- PLAYS ----")
for p in plays_log:
    log(f"t={p[0]:6.2f} {p[1]:10s} zone={p[2]} elixir {p[3]:.2f}->{p[4]:.2f} failed={p[5]}")

# knight solo hit analysis on princess-L (up to first 60s only)
log("")
log("---- PRINCESS-L DAMAGE EVENTS (first 100s) ----")
prev_hp, ev = None, []
for (t, _e, _m, pl, _pr) in series:
    if t > 100:
        break
    if prev_hp is not None and pl is not None and pl < prev_hp:
        ev.append((round(t, 2), prev_hp - pl))
    prev_hp = pl
log(f"events: {ev[:30]}")

env.close()
with open(REPORT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {REPORT}")
