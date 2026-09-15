"""M2 projectile flight tests.

Run:  python3 -m gpusim.tests.test_projectiles
"""
from __future__ import annotations

import os

import torch

from gpusim.env import TICK_DT, make_sim

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")
PRINCESS_HP = 3052.0


def _sim(b=2):
    return make_sim(DATA, batch_size=b)


def _first_hit_time(sim, unit_name):
    """AttackStateMachine timeline: deploy, then windup = max(0, cd - min(deploy, loadTime))."""
    ui = sim.t.unit_index[unit_name]
    cd = float(sim.t.u_cooldown[ui])
    loadt = float(sim.t.u_loadtime[ui])
    dep = float(sim.t.u_deploy[ui])
    return dep + max(0.0, cd - min(dep, loadt))


def test_projectile_flight_time():
    """Musketeer at ~7.1 tiles from the tower: damage must arrive AFTER flight time."""
    sim = _sim()
    mus = sim.t.card_index["musketeer"]
    # position so edge-range check passes (range 6.0 + 0.5 + 1.0 tower radius)
    sim.deploy(0, torch.full((2,), mus), torch.full((2,), 9.0), torch.full((2,), 11.0))
    ui = sim.t.unit_index["musketeer"]
    speed = float(sim.t.u_proj_speed[ui])
    assert speed > 0, "musketeer must have projectile data"
    t_shot = _first_hit_time(sim, "musketeer")
    sim.tick(n=int((t_shot + 0.1) / TICK_DT))
    assert bool(sim.s.p_active.any()), "a projectile must be in flight right after the shot"
    hp_mid = float(sim.s.tower_hp[0, 4])
    prod = float(sim.t.u_proj_radius[ui])
    min_flight = max(0.0, (7.1 - (prod + 1.0)) / speed)
    # sample at half the minimum flight: the arrow cannot have landed yet
    sim.tick(n=max(1, int((min_flight * 0.5) / TICK_DT)))
    assert float(sim.s.tower_hp[0, 4]) == hp_mid, "damage must not land before the projectile arrives"
    # after flight + margin the tower must be hit
    sim.tick(n=int((min_flight * 0.5 + 0.5) / TICK_DT))
    assert float(sim.s.tower_hp[0, 4]) < PRINCESS_HP, "projectile must damage the tower"


def test_projectile_damage_value():
    sim = _sim(1)
    mus = sim.t.card_index["musketeer"]
    ui = sim.t.unit_index["musketeer"]
    dmg = float(sim._dmg[ui])
    speed = float(sim.t.u_proj_speed[ui])
    sim.deploy(0, torch.tensor([mus]), torch.tensor([9.0]), torch.tensor([11.0]))
    t_shot = _first_hit_time(sim, "musketeer")
    flight = 7.1 / speed
    # after the first impact, before the second shot (cadence = cooldown)
    sim.tick(n=int((t_shot + flight + 0.15) / TICK_DT))
    hp = float(sim.s.tower_hp[0, 4])
    assert abs(hp - (PRINCESS_HP - dmg)) < 1e-3, f"tower hp {hp}, expected {PRINCESS_HP - dmg}"


def test_melee_still_instant():
    sim = _sim(1)
    kni = sim.t.card_index["knight"]
    sim.deploy(0, torch.tensor([kni]), torch.tensor([4.5]), torch.tensor([8.5]))
    # Java sync port timeline: activation ~1.05s (SYNC+1 tick) + deploy anim 1.0s
    # + windup 0.5s => first melee hit lands ~2.55s (damage applies directly, no projectile)
    sim.tick(n=int((2.55 + 0.2) / TICK_DT))
    assert not bool(sim.s.p_active.any()), "melee units must not spawn projectiles"
    assert float(sim.s.tower_hp[0, 4]) < PRINCESS_HP, "melee damage lands instantly at the hit moment"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
