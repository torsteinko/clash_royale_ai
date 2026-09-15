"""M2 pathing + king activation tests (BasePathfinder port, Tower activation rule).

Run:  python3 -m gpusim.tests.test_pathing
"""
from __future__ import annotations

import os

import torch

from gpusim.env import TICK_DT, make_sim

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")


def _sim(b=1):
    return make_sim(DATA, batch_size=b)


def test_bridge_routing():
    """A targetless ground unit must cross the river only at a bridge (x near 3.5 or 14.5)."""
    sim = _sim()
    kni = sim.t.card_index["knight"]
    sim.deploy(0, torch.tensor([kni]), torch.tensor([9.0]), torch.tensor([20.0]))
    crossed = checked = False
    for _ in range(40):  # up to 60s in 1.5s chunks
        sim.tick(n=int(1.5 / TICK_DT))
        idx = sim.s.u_active[0].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            break
        i = int(idx[0])
        x, y = float(sim.s.u_x[0, i]), float(sim.s.u_y[0, i])
        if 15.0 < y < 17.0:
            checked = True
            d = min(abs(x - 3.5), abs(x - 14.5))
            assert d < 2.0, f"unit in river zone at x={x:.2f} (must be on a bridge)"
        if y < 14.0:
            crossed = True
            break
    assert checked, "unit must pass through the river zone"
    assert crossed, "unit must reach the other side"


def test_king_activation_by_damage():
    sim = _sim()
    assert not bool(sim.s.king_active.any())
    sim.s.tower_hp[0, 0] -= 100.0  # chip the blue king
    sim.tick(2)
    assert bool(sim.s.king_active[0, 0]), "a damaged king tower must activate"


def test_king_activation_by_princess_loss():
    sim = _sim()
    sim.s.tower_hp[0, 1] = 0.0  # blue left princess falls
    sim.tick(2)
    assert bool(sim.s.king_active[0, 0]), "a fallen princess must activate the king"
    assert not bool(sim.s.king_active[0, 1]), "the other side's king stays inactive"


def test_king_attacks_when_active():
    sim = _sim()
    kni = sim.t.card_index["knight"]
    # intruder next to the blue king before activation
    sim.deploy(1, torch.tensor([kni]), torch.tensor([9.0]), torch.tensor([27.0]))
    sim.tick(int(2.0 / TICK_DT))
    # activate and check that incoming damage continues (king joins the princesses)
    sim.s.tower_hp[0, 0] -= 100.0
    sim.tick(int(0.5 / TICK_DT))
    assert bool(sim.s.king_active[0, 0])
    # at least one blue tower landed damage on the intruder
    k_idx = [i for i, a in enumerate(sim.s.u_active[0]) if a]
    assert k_idx, "intruder alive"
    assert float(sim.s.u_hp[0, int(k_idx[0])]) < float(sim._hp[sim.t.unit_index["knight"]]), \
        "towers must damage the intruder"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
