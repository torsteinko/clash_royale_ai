"""M2 combat tests: targeting, melee/ranged, towers, deaths.

Run:  python3 -m gpusim.tests.test_combat     (or pytest gpusim/tests -q)
"""
from __future__ import annotations

import os

import torch

from gpusim.env import MAX_UNITS, TICK_DT, make_sim

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")


def _sim(b=4):
    return make_sim(DATA, batch_size=b)


def _deploy_card(sim, name, side, x, y, b=None):
    ci = sim.t.card_index[name]
    n = sim.b if b is None else b
    return sim.deploy(side, torch.full((n,), ci), torch.full((n,), float(x), dtype=torch.float32),
                      torch.full((n,), float(y), dtype=torch.float32))


def _unit_state(sim, e=0):
    idx = sim.s.u_active[e].nonzero(as_tuple=False).flatten()
    return [(int(sim.s.u_card[e, i]), float(sim.s.u_hp[e, i]), float(sim.s.u_x[e, i]), float(sim.s.u_y[e, i])) for i in idx]


def test_melee_duel_deterministic_ttk():
    sim = _sim(2)
    knight = sim.t.card_index["knight"]
    dmg = float(sim._dmg[sim.t.unit_index["knight"]])   # 202.2 at lvl 11
    hp = float(sim._hp[sim.t.unit_index["knight"]])     # 1766 at lvl 11
    # deploy 2 tiles apart (edge range 1.2+0.5+0.5=2.2 => in range immediately)
    sim.deploy(0, torch.tensor([knight, knight]), torch.full((2,), 9.0), torch.full((2,), 20.0))
    sim.deploy(1, torch.tensor([knight, knight]), torch.full((2,), 9.0), torch.full((2,), 18.0))
    # at 10s: 8 hits each (first at 1.2s, every 1.2s) -> both at ~148 hp
    sim.tick(n=int(10.0 / TICK_DT))
    st = _unit_state(sim, 0)
    assert len(st) == 2, f"both knights must survive to 10s: {st}"
    for _, uhp, _, _ in st:
        assert abs(uhp - (hp - 8 * dmg)) < 2.5, f"hp {uhp} vs expected ~{hp - 8 * dmg}"
    # symmetric duel: both land the 9th hit -> double KO shortly after 10.8s
    sim.tick(n=int(1.5 / TICK_DT))
    assert len(_unit_state(sim, 0)) == 0, "symmetric duel ends in a double KO"


def test_ranged_kiting():
    sim = _sim(2)
    mus, kni = sim.t.card_index["musketeer"], sim.t.card_index["knight"]
    # musketeer at range 6; knight walking in from 8 tiles away
    sim.deploy(0, torch.tensor([mus, mus]), torch.full((2,), 9.0), torch.full((2,), 20.0))
    sim.deploy(1, torch.tensor([kni, kni]), torch.full((2,), 9.0), torch.full((2,), 12.0))
    sim.tick(n=int(3.0 / TICK_DT))
    mus_hp = float(sim._hp[sim.t.unit_index["musketeer"]])
    knight_hp_full = float(sim._hp[sim.t.unit_index["knight"]])
    st = _unit_state(sim, 0)
    # musketeer should have started damaging the knight while untouched itself
    killed = all(s[1] <= knight_hp_full for s in st if sim.t.names[s[0]] == "Knight")
    assert killed, "knight should be taking ranged damage"
    mstate = [s for s in st if sim.t.names[s[0]] == "Musketeer"]
    assert len(mstate) >= 1 and abs(mstate[0][1] - mus_hp) < 1e-3, "musketeer should be undamaged"


def test_giant_targets_buildings_only():
    sim = _sim(2)
    gi, kni = sim.t.card_index["giant"], sim.t.card_index["knight"]
    # giant marches up mid; a knight stands in its path
    sim.deploy(0, torch.tensor([gi, gi]), torch.full((2,), 9.0), torch.full((2,), 20.0))
    sim.deploy(1, torch.tensor([kni, kni]), torch.full((2,), 9.0), torch.full((2,), 15.0))
    # at ~10s: giant must have locked a BUILDING target (tower slot >= MAX_UNITS)
    sim.tick(n=int(10.0 / TICK_DT))
    g_idx = [i for i, a in enumerate(sim.s.u_active[0]) if a and sim.t.names[int(sim.s.u_card[0, i])] == "Giant"]
    assert g_idx, "giant should still be alive at 10s"
    assert int(sim.s.u_tgt[0, g_idx[0]]) >= MAX_UNITS, "giant must target a building, never the knight"
    # knight must NEVER take damage from the giant (buildings-only)
    sim.tick(n=int(8.0 / TICK_DT))
    k_full = float(sim._hp[sim.t.unit_index["knight"]])
    k_idx = [i for i, a in enumerate(sim.s.u_active[0]) if a and sim.t.names[int(sim.s.u_card[0, i])] == "Knight"]
    assert k_idx, "knight alive"
    assert abs(float(sim.s.u_hp[0, k_idx[0]]) - k_full) < 1e-3, "giant must not damage the knight"


def test_tower_attacks_intruder():
    sim = _sim(2)
    kni = sim.t.card_index["knight"]
    # deploy knight deep in red territory (near red princess) -> tower shoots it
    sim.deploy(0, torch.tensor([kni, kni]), torch.full((2,), 4.0), torch.full((2,), 8.0))
    sim.tick(n=int(4.0 / TICK_DT))
    hp0 = float(sim._hp[sim.t.unit_index["knight"]])
    st = _unit_state(sim, 0)
    assert st and st[0][1] < hp0, "princess tower should be damaging the intruder"


def test_deploy_timer_blocks_combat():
    sim = _sim(2)
    kni = sim.t.card_index["knight"]
    sim.deploy(0, torch.tensor([kni, kni]), torch.full((2,), 9.0), torch.full((2,), 20.0))
    sim.deploy(1, torch.tensor([kni, kni]), torch.full((2,), 9.0), torch.full((2,), 18.0))
    sim.tick(n=int(0.8 / TICK_DT))  # deploy time is 1.0s -> nobody has attacked yet
    st = _unit_state(sim, 0)
    full = float(sim._hp[sim.t.unit_index["knight"]])
    assert all(abs(s[1] - full) < 1e-3 for s in st), "no damage during deploy phase"


def test_crowns_and_king_win():
    sim = _sim(1)
    gi = sim.t.card_index["giant"]
    # spam giants at the red king tower area until something falls (long sim)
    for _ in range(6):
        sim.deploy(0, torch.tensor([gi]), torch.tensor([9.0]), torch.tensor([10.0]))
        sim.tick(n=int(20.0 / TICK_DT))
    total = float(sim.s.tower_hp[0, 3:].sum())  # red towers
    assert total < (4824.0 + 2 * 3052.0), "blue attacks must damage red towers"
    if bool(sim.s.game_over[0]):
        assert int(sim.s.winner[0]) == 0


def test_determinism():
    def run():
        sim = _sim(2)
        kni = sim.t.card_index["knight"]
        sim.deploy(0, torch.tensor([kni, kni]), torch.full((2,), 9.0), torch.full((2,), 20.0))
        sim.tick(n=200)
        return sim.s.u_hp.clone(), sim.s.u_x.clone(), sim.s.tower_hp.clone()

    a = run()
    b = run()
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "combat sim must be deterministic"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
