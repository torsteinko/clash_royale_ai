"""M3 tests: card cycle/hand, spells (crown-tower %), deployment zones.

Run:  python3 -m gpusim.tests.test_m3
"""
from __future__ import annotations

import os

import torch

from gpusim.env import TICK_DT, make_sim

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")
DECK = ["Knight", "Archer", "Giant", "Musketeer", "Fireball", "Zap", "Log", "Cannon"]


def _sim(b=2):
    return make_sim(DATA, batch_size=b)


def _C(sim, name):
    return sim.t.card_index[name.lower().replace(" ", "")]


SDECK = ["Fireball", "Zap", "Log", "Knight", "Archer", "Giant", "Musketeer", "Cannon"]


def _deck_spell(sim):
    return [_C(sim, n) for n in SDECK]


def _deck(sim):
    return [sim.t.card_index[n.lower().replace(" ", "")] for n in DECK]


def test_hand_cycle_and_elixir():
    sim = _sim()
    deck = _deck(sim)
    sim.set_deck(0, deck)
    assert sim.s.hand[0, 0, :].tolist() == deck[:4]
    ok = sim.play(0, 0, torch.full((2,), 9.0), torch.full((2,), 20.0))
    assert ok.tolist() == [True, True], "knight (3 elixir) must play"
    assert abs(float(sim.s.elixir[0, 0]) - 2.0) < 1e-5
    # played card rotates to the back; next card enters the hand
    assert int(sim.s.hand[0, 0, 0]) == deck[4], "fireball should enter the hand"
    assert int(sim.s.cycle[0, 0, 0]) == deck[0], "knight should rotate into the cycle"
    assert int(sim.s.cycle_pos[0, 0]) == 1
    # insufficient elixir -> no play, no rotation
    ok2 = sim.play(0, 2, torch.full((2,), 9.0), torch.full((2,), 20.0))  # giant, 5 elixir
    assert ok2.tolist() == [False, False]
    assert int(sim.s.hand[0, 0, 2]) == deck[2]


def test_deployment_zones():
    sim = _sim(1)
    sim.set_deck(0, _deck(sim))
    sim.s.elixir[0, 0] = 10.0
    # enemy half: rejected (no pocket yet)
    ok = sim.play(0, 0, torch.tensor([9.0]), torch.tensor([10.0]))
    assert ok.tolist() == [False], "cannot deploy on the enemy half"
    # own half: accepted
    ok = sim.play(0, 0, torch.tensor([9.0]), torch.tensor([20.0]))
    assert ok.tolist() == [True]
    # destroy the red left princess -> that pocket opens
    sim.s.tower_hp[0, 4] = 0.0
    sim.tick(2)
    sim.s.elixir[0, 0] = 10.0
    ok = sim.play(0, 0, torch.tensor([4.0]), torch.tensor([10.0]))
    assert ok.tolist() == [True], "pocket must open after the enemy princess falls"


def test_fireball_spell_damage():
    sim = _sim(1)
    deck = _deck_spell(sim)
    sim.set_deck(0, deck)
    sim.s.elixir[0, 0] = 10.0
    # a red knight next to the red princess tower (3.5, 6.5)
    kni = _C(sim, "Knight")
    sim.deploy(1, torch.tensor([kni]), torch.tensor([3.5]), torch.tensor([8.0]))
    slot_fb = 0  # Fireball is first in SDECK -> first hand slot
    ui_k = sim.t.unit_index["knight"]
    fac = sim.fac
    fb_dmg = float(sim.t.spell_damage[_C(sim, "Fireball")]) * fac
    crown = 1.0 + float(sim.t.spell_crown_pct[_C(sim, "Fireball")]) / 100.0
    # sanity: 269 * 2.5596 = 688.5 -> x0.3 = 206.5 (matches the historical probe)
    assert abs(fb_dmg - 688.5) < 1.5 and abs(fb_dmg * crown - 206.5) < 1.5
    ok = sim.play(0, slot_fb, torch.tensor([3.5]), torch.tensor([6.5]))
    assert ok.tolist() == [True]
    # tower took the reduced (crown) damage; knight took full
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - fb_dmg * crown)) < 0.6
    knight_hp = float(sim._hp[ui_k]) - fb_dmg
    k_idx = int(sim.s.u_active[0].nonzero()[0, 0])
    assert abs(float(sim.s.u_hp[0, k_idx]) - knight_hp) < 0.6
    assert abs(float(sim.s.elixir[0, 0]) - 6.0) < 1e-5  # 10 - 4


def test_zap_crown_percent():
    sim = _sim(1)
    deck = _deck_spell(sim)
    sim.set_deck(0, deck)
    sim.s.elixir[0, 0] = 10.0
    slot_zap = 1  # Zap is second in SDECK
    zap = _C(sim, "Zap")
    dmg = float(sim.t.spell_damage[zap]) * sim.fac
    crown = 1.0 + float(sim.t.spell_crown_pct[zap]) / 100.0
    ok = sim.play(0, slot_zap, torch.tensor([3.5]), torch.tensor([6.5]))
    assert ok.tolist() == [True]
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - dmg * crown)) < 0.6


def test_full_cycle_draw_order_no_empty_slots():
    """Regression: playing past 4 cards must draw the played cards back in FIFO
    order (exact Java Hand semantics). The old `cycle_pos % 8` walked off the
    4-slot queue into uninitialised slots — hand slots became -1 (found by the
    M4.2 replay diff; see reports/m4_replay_diff.md)."""
    sim = _sim(1)
    deck = _deck(sim)  # hand = deck[0:4], draw queue = deck[4:8]
    sim.set_deck(0, deck)
    drawn = []
    for _ in range(8):
        sim.s.elixir[0, 0] = 10.0
        ok = sim.play(0, 0, torch.full((1,), 9.0), torch.full((1,), 20.0))
        assert bool(ok[0])
        hand = sim.s.hand[0, 0].tolist()
        assert all(c >= 0 for c in hand), f"hand must never contain -1: {hand}"
        drawn.append(int(sim.s.hand[0, 0, 0]))
    # queue drains deck[4:8], then the played cards return in play order
    assert drawn[0:4] == deck[4:8]
    assert drawn[4] == deck[0], "first played card must come back after the queue drains"
    assert drawn[5] == deck[4], "then the card played on the second draw"
    # cycle_pos stays a 4-slot ring
    assert int(sim.s.cycle_pos[0, 0]) == 0  # 8 plays -> (0+8) % 4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
