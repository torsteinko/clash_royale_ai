#!/usr/bin/env python3
"""Probe 3: fireball zone reachability (zones 7/8/9) + overtime structure (noop run to 5:00)."""
import numpy as np

from crforge_gym import CRForgeEnv


def princess(towers, left):
    for t in towers:
        if t.get("type") == "princess" and ((t.get("x", 9) < 9) == left):
            return t["hp"]
    return None


print("======== PART A: fireball at spell zones ========")
for zone in [7, 8, 9]:
    env = CRForgeEnv(endpoint="tcp://localhost:9876", opponent="noop", binary_obs=False,
                     ticks_per_step=1, level=11)
    obs, info = env.reset(seed=7)
    raw = env.unwrapped._last_obs_raw
    done = False
    cast = False
    t_cast = None
    before_cast = None
    last_seen = None
    samples = []
    while not done and float(raw["gameTimeSeconds"]) < 26.0:
        h = raw["bluePlayer"]["hand"]
        fb = next(((i, c) for i, c in enumerate(h) if c.get("id") == "fireball"), None)
        t = float(raw["gameTimeSeconds"])
        if not cast and fb and float(raw["bluePlayer"]["elixir"]) >= 4 and t >= 15.0:
            i, _c = fb
            before_cast = (princess(raw["redPlayer"]["towers"], True),
                           princess(raw["redPlayer"]["towers"], False))
            obs, r, term, trunc, inf = env.step(np.array([1, i, zone], dtype=np.int64))
            raw = env.unwrapped._last_obs_raw
            cast = True
            t_cast = t
            continue
        obs, r, term, trunc, inf = env.step(np.array([0, 0, 0], dtype=np.int64))
        raw = env.unwrapped._last_obs_raw
        dt = float(raw["gameTimeSeconds"]) - (t_cast or 0)
        if cast and 0 <= dt <= 4.0:
            samples.append((round(dt, 2), princess(raw["redPlayer"]["towers"], True),
                            princess(raw["redPlayer"]["towers"], False)))
            last_seen = samples[-1]
        done = bool(term) or bool(trunc)
    env.close()
    if before_cast and last_seen:
        print(f"zone {zone}: cast t={t_cast:.2f} | princess-L {before_cast[0]} -> {last_seen[1]} "
              f"(dmg {before_cast[0] - last_seen[1]}) | princess-R {before_cast[1]} -> {last_seen[2]} "
              f"(dmg {before_cast[1] - last_seen[2]})")
        print(f"   samples every ~1s: {samples[::20]}")

print()
print("======== PART B: full noop run — overtime structure ========")
env = CRForgeEnv(endpoint="tcp://localhost:9876", opponent="noop", binary_obs=False,
                 ticks_per_step=1, level=11)
obs, info = env.reset(seed=11)
raw = env.unwrapped._last_obs_raw
mults = []
prev = None
done = False
t = 0.0
while not done:
    obs, r, term, trunc, inf = env.step(np.array([0, 0, 0], dtype=np.int64))
    raw = env.unwrapped._last_obs_raw
    t = float(raw["gameTimeSeconds"])
    m = float(raw.get("elixirMultiplier", 1))
    o = bool(raw.get("isOvertime", False))
    key = (m, o)
    if key != prev:
        mults.append((round(t, 2), m, o))
        prev = key
    done = bool(term) or bool(trunc)
print(f"noop run: end t={t:.1f} term={term} trunc={trunc} crowns b/r="
      f"{raw['bluePlayer']['crowns']}/{raw['redPlayer']['crowns']}")
print(f"multiplier/overtime timeline: {mults}")
env.close()
