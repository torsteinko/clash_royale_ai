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


def test_projectile_flight_time():
    """Musketeer at ~7.1 tiles from the tower: damage must arrive AFTER flight time."""
    sim = _sim()
    mus = sim.t.card_index["musketeer"]
    # position so edge-range check passes (range 6.0 + 0.5 + 1.0 tower radius)
    sim.deploy(0, torch.full((2,), mus), torch.full((2,), 9.0), torch.full((2,), 11.0))
    ui = sim.t.unit_index["musketeer"]
    speed = float(sim.t.u_proj_speed[ui])
    assert speed > 0, "musketeer must have projectile data"
    # first shot at ~1.0s (deploy 1.0s, cooldown 1.0s ticking down in parallel)
    sim.tick(n=int(1.2 / TICK_DT))
    assert bool(sim.s.p_active.any()), "a projectile must be in flight at t=1.2s"
    hp_mid = float(sim.s.tower_hp[0, 4])
    # impact at ~1.0s + (7.1 - 1.5) / speed = ~1.34s; one more tick (1.25s) is safely before it
    sim.tick(n=1)
    assert float(sim.s.tower_hp[0, 4]) == hp_mid, "damage must not land before the projectile arrives"
    # after flight + margin the tower must be hit
    sim.tick(n=int(1.5 / TICK_DT))
    assert float(sim.s.tower_hp[0, 4]) < PRINCESS_HP, "projectile must damage the tower"


def test_projectile_damage_value():
    sim = _sim(1)
    mus = sim.t.card_index["musketeer"]
    dmg = float(sim._dmg[sim.t.unit_index["musketeer"]])
    sim.deploy(0, torch.tensor([mus]), torch.tensor([9.0]), torch.tensor([11.0]))
    # run until exactly one impact (shot at 1.0s; second shot at 2.0s, impact 2.5s+)
    sim.tick(n=int(2.2 / TICK_DT))
    hp = float(sim.s.tower_hp[0, 4])
    assert abs(hp - (PRINCESS_HP - dmg)) < 1e-3, f"tower hp {hp}, expected {PRINCESS_HP - dmg}"


def test_melee_still_instant():
    sim = _sim(1)
    kni = sim.t.card_index["knight"]
    sim.deploy(0, torch.tensor([kni]), torch.tensor([4.5]), torch.tensor([8.5]))
    sim.tick(n=int(1.3 / TICK_DT))
    assert not bool(sim.s.p_active.any()), "melee units must not spawn projectiles"
    assert float(sim.s.tower_hp[0, 4]) < PRINCESS_HP, "melee damage lands instantly"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
