#!/usr/bin/env python3
"""CRForge fidelity probe — does the sim behave like real Clash Royale?

Checks against known real values (tournament level 11):
- start elixir = 5; regen = 1 per 2.8s; x2 in final minute; x3 in final OT minute
- tower maxHp: King 4824, Princess 2534
- Knight: cost 3, deploy ~1s, damage 202 per 1.2s, HP 1766
- Fireball (lvl 11): area 688, crown-tower damage 25-30% (172-207)
- match structure: 3:00 regular, OT to 5:00 cap, outcome

Writes /tmp/crforge_fidelity_report.txt (+ .json). JSON-mode obs, ticks_per_step=1.
"""
import json
import sys
import time

import numpy as np

from crforge_gym import CRForgeEnv

REPORT = "/tmp/crforge_fidelity_report.txt"
lines = []


def log(s=""):
    lines.append(str(s))
    print(s, flush=True)


def main():
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

    def tower_map(side):
        ts = raw.get(side + "Player", {}).get("towers", [])
        out = {"crown": None, "pl": None, "pr": None}
        for t in ts:
            if t.get("type") == "crown":
                out["crown"] = t
            elif t.get("type") == "princess":
                if t.get("x", 9) < 9:
                    out["pl"] = t
                else:
                    out["pr"] = t
        return out

    log("=" * 64)
    log("CRForge fidelity probe — seed 7, level 11, opponent noop, tps=1")
    log("=" * 64)
    bt, rt = tower_map("blue"), tower_map("red")
    log(f"blue towers: crown hp={bt['crown']['maxHp']} @({bt['crown']['x']},{bt['crown']['y']}) | "
        f"princess-L hp={bt['pl']['maxHp']} @({bt['pl']['x']},{bt['pl']['y']}) | "
        f"princess-R hp={bt['pr']['maxHp']} @({bt['pr']['x']},{bt['pr']['y']})")
    log(f"red  towers: crown hp={rt['crown']['maxHp']} @({rt['crown']['x']},{rt['crown']['y']}) | "
        f"princess-L hp={rt['pl']['maxHp']} @({rt['pl']['x']},{rt['pl']['y']}) | "
        f"princess-R hp={rt['pr']['maxHp']} @({rt['pr']['x']},{rt['pr']['y']})")
    log(f"start: t={gt():.1f} elixir={elx():.1f} mult={mult():.1f} hand={[c.get('id') for c in hand()]}")
    log("")

    series = []       # (t, elx, mult, rpl_hp, rpr_hp, bpl_hp, bpr_hp)
    plays = []        # (t, card_id, slot, zone, cost, elixir_before, elixir_after, failed)
    knight_track = [] # (t, x, y, hp)
    last_knight_seen = None

    done = False
    steps = 0
    t_wall = time.time()
    first_troop_play_t = None
    knight_played_at = None

    while not done:
        a = np.array([0, 0, 0], dtype=np.int64)
        h = hand()
        affordable = [(i, c) for i, c in enumerate(h) if elx() >= float(c.get("cost", 99))]
        # Prefer FIRST troop play to be knight (it is in hand at t=0 per seed 7)
        choice = None
        if first_troop_play_t is None and gt() >= 2.0:
            kn = [(i, c) for i, c in affordable if c.get("id") == "knight"]
            if kn:
                choice = kn[0]
        elif first_troop_play_t is not None:
            # cycle: play first affordable troop (prefer knight), else fireball if affordable
            troops = [(i, c) for i, c in affordable if c.get("type") == "TROOP"]
            if troops:
                kn = [(i, c) for i, c in troops if c.get("id") == "knight"]
                choice = kn[0] if kn else troops[0]
            else:
                spells = [(i, c) for i, c in affordable if c.get("id") == "fireball"]
                if spells and gt() >= 20.0:
                    choice = spells[0]

        if choice is not None:
            i, c = choice
            zone = 0 if c.get("type") == "TROOP" else 8
            before = elx()
            a = np.array([1, i, zone], dtype=np.int64)
            obs, r, term, trunc, inf = env.step(a)
            raw = env.unwrapped._last_obs_raw
            plays.append((gt(), c.get("id"), i, zone, float(c.get("cost", 0)), before, elx(),
                          bool(inf.get("action_failed"))))
            if c.get("type") == "TROOP" and first_troop_play_t is None:
                first_troop_play_t = gt()
                knight_played_at = gt() if c.get("id") == "knight" else knight_played_at
            continue

        obs, r, term, trunc, inf = env.step(a)
        raw = env.unwrapped._last_obs_raw
        steps += 1

        rpl, rpr = tower_map("red"), tower_map("blue")
        series.append((gt(), elx(), mult(),
                       rpl["pl"]["hp"], rpl["pr"]["hp"], rpr["pl"]["hp"], rpr["pr"]["hp"]))

        ks = [e for e in ents() if e.get("name") == "Knight" and e.get("team") == "BLUE"]
        if ks:
            k = ks[0]
            if last_knight_seen is None or k["hp"] != last_knight_seen:
                knight_track.append((gt(), k.get("x"), k.get("y"), k.get("hp")))
            last_knight_seen = k["hp"]

        done = bool(term) or bool(trunc)

    wall = time.time() - t_wall
    log(f"match done: t={gt():.1f}s overtime={raw.get('isOvertime')} term={term} trunc={trunc} "
        f"crowns blue={raw.get('bluePlayer',{}).get('crowns')} red={raw.get('redPlayer',{}).get('crowns')} "
        f"({steps} steps, {wall:.0f}s wall)")
    log("")

    # ---------------- analysis ----------------
    log("---- PLAYS ----")
    for p in plays:
        log(f"t={p[0]:6.2f} play {p[1]:12s} slot={p[2]} zone={p[3]} cost={p[4]:.0f} "
            f"elixir {p[5]:.2f}->{p[6]:.2f} failed={p[7]}")

    log("")
    log("---- ELIXIR ----")
    # multiplier timeline
    changes = []
    prev = None
    for (t, _e, m, *_rest) in series:
        if m != prev:
            changes.append((t, m))
            prev = m
    log(f"multiplier timeline: {[(round(t,1), m) for t,m in changes]}")

    def slope(t0, t1, lo=0.0, hi=9.9):
        seg = [(t, e) for (t, e, *_r) in series if t0 <= t <= t1 and lo <= e <= hi]
        if len(seg) < 3:
            return None
        ts = np.array([s[0] for s in seg]); es = np.array([s[1] for s in seg])
        s = float(np.polyfit(ts, es, 1)[0])
        return s

    for (t0, t1, label) in [(5, 100, "regular 1x"),
                            (125, 175, "regular last min"),
                            (185, 235, "OT first min"),
                            (245, 295, "OT last min")]:
        s = slope(t0, t1)
        if s:
            log(f"elixir regen {label:16s}: {s:.4f} elixir/s -> 1 per {1/s:.2f}s "
                f"(real: 1x=2.80s, 2x=1.40s, 3x=0.93s)")
        else:
            log(f"elixir regen {label:16s}: n/a (capped or no samples)")

    log("")
    log("---- KNIGHT ----")
    if knight_played_at is not None:
        log(f"played at t={knight_played_at:.2f}")
    if knight_track:
        log(f"first seen t={knight_track[0][0]:.2f} pos=({knight_track[0][1]},{knight_track[0][2]}) hp={knight_track[0][3]}")
        log(f"last seen  t={knight_track[-1][0]:.2f} pos=({knight_track[-1][1]},{knight_track[-1][2]}) hp={knight_track[-1][3]}")
    # hits on red princess-L
    hits = []
    prev_hp = None
    for (t, _e, _m, rpl, *_r) in series:
        if prev_hp is not None and rpl < prev_hp:
            hits.append((t, prev_hp - rpl))
        prev_hp = rpl
    if hits:
        dmg = [h[1] for h in hits]
        gaps = [round(hits[i+1][0] - hits[i][0], 2) for i in range(len(hits)-1)]
        log(f"red princess-L took {len(hits)} hits: dmgs={dmg[:12]} gaps={gaps[:12]}")
        log(f"last hit t={hits[-1][0]:.2f}; princess-L hp now {series[-1][3]}")

    log("")
    log("---- FIREBALL ----")
    fb = [p for p in plays if p[1] == "fireball"]
    log(f"fireball casts: {len(fb)}")
    if fb:
        # look for red tower hp drops within 3s after each cast
        any_drop = False
        for p in fb:
            t_cast = p[0]
            window = [(t, rpl, rpr) for (t, _e, _m, rpl, rpr, *_r) in series if t_cast <= t <= t_cast + 3.0]
            if len(window) >= 2:
                d_pl = window[0][1] - window[-1][1]
                d_pr = window[0][2] - window[-1][2]
                if d_pl or d_pr:
                    any_drop = True
                    log(f"cast t={t_cast:.2f}: red princess-L hp -{d_pl}, princess-R hp -{d_pr} within 3s")
        if not any_drop:
            log("no red-tower hp drop detected within 3s of casts (zone 8 may be out of princess radius)")

    log("")
    log("---- FINAL STATE ----")
    log(f"end t={gt():.1f} mult={mult():.1f}")
    log(f"red  princess-L hp={tower_map('red')['pl']['hp']}/{tower_map('red')['pl']['maxHp']}  "
        f"princess-R hp={tower_map('red')['pr']['hp']}/{tower_map('red')['pr']['maxHp']}")
    log(f"blue princess-L hp={tower_map('blue')['pl']['hp']}  king hp={tower_map('blue')['crown']['hp']}/{tower_map('blue')['crown']['maxHp']}")

    env.close()

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    summary = {
        "towers": {
            "king_maxHp": bt["crown"]["maxHp"],
            "princess_maxHp": bt["pl"]["maxHp"],
        },
        "knight_hits": hits,
        "mult_timeline": changes,
        "plays": plays,
    }
    with open("/tmp/crforge_fidelity_report.json", "w") as f:
        json.dump(summary, f, default=str, indent=1)
    print(f"\nreport written to {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
