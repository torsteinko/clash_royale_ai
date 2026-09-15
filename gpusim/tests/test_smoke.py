"""CPU smoke tests for the batched sim core (milestone M1).

Run:  python3 -m gpusim.tests.test_smoke     (or pytest gpusim/tests -q)
"""
from __future__ import annotations

import os

import torch

from gpusim.env import ELIXIR_PERIOD, TICK_DT, make_sim

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")


def _sim(b=4):
    return make_sim(DATA, batch_size=b)


def test_data_loaded():
    sim = _sim()
    assert len(sim.t) > 100, "card table should have all cards"
    assert "knight" in sim.t.card_index
    # patched values: knight 690 base, giant 1930 base (M0 spec)
    ki = sim.t.unit_index["knight"]
    gi = sim.t.unit_index["giant"]
    assert abs(float(sim.t.u_health[ki]) - 690) < 1e-6
    assert abs(float(sim.t.u_health[gi]) - 1930) < 1e-6
    # level 11 scaling: knight 690 * 2.5596 = 1766 (matches the real-game value)
    assert abs(float(sim._hp[ki]) - 1766.12) < 0.5


def test_deploy_elixir_and_spawn():
    sim = _sim()
    knight = sim.t.card_index["knight"]
    idx = torch.tensor([knight, knight, -1, -1])
    x = torch.full((4,), 9.0)
    y = torch.full((4,), 20.0)
    ok = sim.deploy(side=0, card_idx=idx, x=x, y=y)
    assert ok.tolist() == [True, True, False, False], "deploy mask wrong"
    assert abs(float(sim.s.elixir[0, 0]) - 2.0) < 1e-5, "cost 3 not deducted"
    assert int(sim.s.u_active[0].sum()) == 1, "knight not spawned"
    assert int(sim.s.u_active[2].sum()) == 0, "no card -> no spawn"
    ci = int(sim.s.u_card[0, sim.s.u_active[0].nonzero()[0]])
    assert sim.t.names[ci] == "Knight"


def test_elixir_regen():
    sim = _sim()
    sim.tick(n=int(round(ELIXIR_PERIOD / TICK_DT)))
    assert abs(float(sim.s.elixir[0, 0]) - 6.0) < 0.05, "one elixir per 2.8s"


def test_movement_direction():
    sim = _sim()
    knight = sim.t.card_index["knight"]
    sim.deploy(side=0, card_idx=torch.tensor([knight] * 4), x=torch.full((4,), 9.0), y=torch.full((4,), 20.0))
    y0 = float(sim.s.u_y[0, sim.s.u_active[0].nonzero()[0]])
    sim.tick(n=20)  # 1 second
    y1 = float(sim.s.u_y[0, sim.s.u_active[0].nonzero()[0]])
    assert y1 < y0, "blue units must march toward the red side"


def test_multi_spawn_stagger():
    sim = _sim()
    sk = sim.t.card_index.get("skeletons")
    assert sk is not None
    sim.deploy(side=0, card_idx=torch.tensor([sk] * 4), x=torch.full((4,), 9.0), y=torch.full((4,), 20.0))
    assert int(sim.s.u_active[0].sum()) == 1, "first skeleton immediate"
    sim.tick(n=10)  # 0.5 s -> at least one more of the 4 staggered spawns
    assert int(sim.s.u_active[0].sum()) >= 2, "staggered spawns should fire"


def test_determinism():
    def run():
        sim = _sim(2)
        knight = sim.t.card_index["knight"]
        sim.deploy(side=0, card_idx=torch.tensor([knight, -1]), x=torch.full((2,), 9.0), y=torch.full((2,), 20.0))
        sim.tick(n=50)
        return sim.s.u_x.clone(), sim.s.elixir.clone()

    x1, e1 = run()
    x2, e2 = run()
    assert torch.equal(x1, x2) and torch.equal(e1, e2), "sim must be deterministic"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
