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

# M3.6 Log tests: the deck's first four are the opening hand (Log at slot 2,
# Knight slot 3, MegaMinion slot 0 — same ordering the log_roll scenario uses)
LOG_DECK = ["MegaMinion", "Fireball", "Log", "Knight", "Musketeer", "Giant", "Archer", "Zap"]


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
    """Java `Arena.isValidPlacement` tile rules, ported 1:1 (M4.2b).

    Points are Java tiles: tile = (floor(x), floor(32 - y)) in tiles.
    """
    sim = _sim(1)

    def reset_hand():
        sim.set_deck(0, _deck(sim))
        sim.s.elixir[0, 0] = 10.0

    reset_hand()
    # enemy half: rejected — (9,10) is Java tile (9,22), red zone (no pocket yet)
    ok = sim.play(0, 0, torch.tensor([9.0]), torch.tensor([10.0]))
    assert ok.tolist() == [False], "cannot deploy on the enemy half"
    reset_hand()
    # own half: accepted — (9,20) is Java tile (9,12), blue zone
    ok = sim.play(0, 0, torch.tensor([9.0]), torch.tensor([20.0]))
    assert ok.tolist() == [True]
    reset_hand()
    # own live princess tower tile: TOWER tiles are never deployable (even the owner)
    ok = sim.play(0, 0, torch.tensor([3.5]), torch.tensor([25.5]))
    assert ok.tolist() == [False], "cannot deploy onto a live tower tile"
    reset_hand()
    # river row (Java tiles 15/16): not deployable — (9,16.5) is Java tile (9,15)
    ok = sim.play(0, 0, torch.tensor([9.0]), torch.tensor([16.5]))
    assert ok.tolist() == [False], "cannot deploy into the river"
    # destroy the red left princess -> the 4-row pocket in her lane opens
    sim.s.tower_hp[0, 4] = 0.0
    sim.tick(2)
    reset_hand()
    ok = sim.play(0, 0, torch.tensor([4.0]), torch.tensor([13.5]))
    assert ok.tolist() == [True], "pocket must open after the enemy princess falls"
    reset_hand()
    # but only 4 rows past the river (Java pocket rows 17..20 == y 11..14)
    ok = sim.play(0, 0, torch.tensor([4.0]), torch.tensor([10.0]))
    assert ok.tolist() == [False], "pocket is 4 rows deep, not the whole enemy half"


def test_log_spell_as_deploy_own_side_only():
    """The Log is `spellAsDeploy` in the reference data: Match.validateAction
    routes it through Arena.isValidPlacement (own side only), while plain spells
    (Fireball) may be cast anywhere in bounds."""
    sim = _sim(1)
    sim.set_deck(0, _deck_spell(sim))  # slot 0 = Fireball, slot 2 = Log
    sim.s.elixir[0, 0] = 10.0
    # Log on the enemy half -> rejected (no elixir spent)
    ok = sim.play(0, 2, torch.tensor([9.0]), torch.tensor([10.0]))
    assert ok.tolist() == [False], "spellAsDeploy must respect the own-side rule"
    assert abs(float(sim.s.elixir[0, 0]) - 10.0) < 1e-6, "rejected cast must not spend elixir"
    # Log on the own half -> accepted
    ok = sim.play(0, 2, torch.tensor([9.0]), torch.tensor([20.0]))
    assert ok.tolist() == [True], "own-side Log cast must succeed"
    # Fireball (plain spell) on the enemy half -> still accepted
    ok = sim.play(0, 0, torch.tensor([9.0]), torch.tensor([10.0]))
    assert ok.tolist() == [True], "plain spells may be placed anywhere in bounds"


def test_fireball_spell_damage():
    """M3.3 + M3.4: the fireball flies to the cast point (sync 1.0 s, then
    10 tiles/s from the blue crown tower) and applies its AOE on arrival."""
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
    # sanity: 269 * 2.56 = 688.6 -> x0.30 = 206.6 (integer-scaled at impact)
    assert abs(fb_dmg - 688.5) < 1.5 and abs(fb_dmg * crown - 206.5) < 1.5
    ok = sim.play(0, slot_fb, torch.tensor([3.5]), torch.tensor([6.5]))
    assert ok.tolist() == [True]
    assert abs(float(sim.s.elixir[0, 0]) - 6.0) < 1e-5  # 10 - 4, spent at cast time
    # impact tick measured at 66 (20 ticks sync + ~46 ticks flight); no damage
    # may land before the projectile arrives
    sim.tick(65)
    assert float(sim.s.tower_hp[0, 4]) == 3052.0, "damage must wait for the flight"
    sim.tick(1)
    # tower took the reduced (crown) damage; knight took the full floor-scaled 688
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 206.0)) < 0.6
    knight_hp = 1766.0 - 688.0  # floor(690 * 2.56) - floor(269 * 2.56)
    k_idx = int(sim.s.u_active[0].nonzero()[0, 0])
    assert abs(float(sim.s.u_hp[0, k_idx]) - knight_hp) < 0.6


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
    # Zap is a direct area spell (no projectile): the AreaEffect applies one
    # sync step later -> first damage lands on tick 21; floor(floor(75*2.56)*0.3) = 57
    sim.tick(20)
    assert float(sim.s.tower_hp[0, 4]) == 3052.0, "damage must wait for the sync step"
    sim.tick(1)
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 57.0)) < 0.6


def test_poison_zone_ticks_and_crown_pct():
    """M3.5: ticking zones (reference AreaEffect entity + TickingHandler).

    Poison: hitSpeed 0.25 s, lifeDuration 8 s -> 32 damage ticks; per-tick
    damage = scaleCard(round(36 * 0.25)) = 23 at level 11; towers take the
    buff's crown percent (-75 -> 25 %): floor(23 * 25 / 100) = 5.
    """
    sim = _sim(1)
    deck = [_C(sim, n) for n in ("Poison", "Earthquake", "Knight", "Archer",
                                 "Giant", "Fireball", "Zap", "Log")]
    sim.set_deck(0, deck)
    sim.s.elixir[0, 0] = 10.0
    kni = _C(sim, "Knight")
    sim.deploy(1, torch.tensor([kni]), torch.tensor([3.5]), torch.tensor([7.2]))
    assert abs(float(sim.t.spell_tick_dmg_base[_C(sim, "Poison")]) - 9.0) < 1e-6  # round(36*0.25)
    ok = sim.play(0, 0, torch.tensor([3.5]), torch.tensor([6.5]))
    assert ok.tolist() == [True]
    # cast fires after the 1.05 s sync (tick 21); first zone tick at tick 25
    sim.tick(24)
    assert float(sim.s.tower_hp[0, 4]) == 3052.0, "no tick before hitSpeed accumulates"
    sim.tick(1)
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 5.0)) < 1e-3
    k_idx = int(sim.s.u_active[0].nonzero()[0, 0])
    assert abs(float(sim.s.u_hp[0, k_idx]) - (1766.0 - 23.0)) < 1e-3
    # 32 ticks total (8 s / 0.25 s): the tower ends at 3052 - 32 * 5
    sim.tick(155)
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 32 * 5.0)) < 1e-3
    sim.tick(10)  # zone expired -> no further damage
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 32 * 5.0)) < 1e-3
    assert not bool(sim.s.z_active.any()), "poison zone must expire after 8 s"


def test_earthquake_zone_building_bonus():
    """M3.5: Earthquake: hitSpeed 0.1 s, life 3 s -> 30 ticks; units take
    scaleCard(round(32 * 0.1)) = 7; towers get the +350 % building bonus
    first (floor(7 * 450 / 100) = 31) and then the crown percent
    (-35 -> floor(31 * 65 / 100) = 20)."""
    sim = _sim(1)
    deck = [_C(sim, n) for n in ("Poison", "Earthquake", "Knight", "Archer",
                                 "Giant", "Fireball", "Zap", "Log")]
    sim.set_deck(0, deck)
    sim.s.elixir[0, 0] = 10.0
    kni = _C(sim, "Knight")
    sim.deploy(1, torch.tensor([kni]), torch.tensor([3.5]), torch.tensor([7.2]))
    ok = sim.play(0, 1, torch.tensor([3.5]), torch.tensor([6.5]))
    assert ok.tolist() == [True]
    # first zone tick at tick 22 (0.1 s = 2 ticks after the sync cast)
    sim.tick(21)
    assert float(sim.s.tower_hp[0, 4]) == 3052.0
    sim.tick(1)
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 20.0)) < 1e-3
    k_idx = int(sim.s.u_active[0].nonzero()[0, 0])
    assert abs(float(sim.s.u_hp[0, k_idx]) - (1766.0 - 7.0)) < 1e-3
    # 30 ticks total (3 s / 0.1 s): the tower ends at 3052 - 30 * 20
    sim.tick(58)
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 30 * 20.0)) < 1e-3
    sim.tick(5)
    assert abs(float(sim.s.tower_hp[0, 4]) - (3052.0 - 30 * 20.0)) < 1e-3
    assert not bool(sim.s.z_active.any()), "earthquake zone must expire after 3 s"


def test_log_spell_as_deploy_chain_ground_hit():
    """M3.6: The Log is spellAsDeploy — the deploy projectile spawns AT the cast
    point (never the crown tower) and travels `forward` = 3 game units, then the
    rolling piercing sub-projectile rolls 10.1 tiles at raw speed 200 (float32:
    166.6666717529297 game units per tick).

    Timeline (Log cast between ticks 0/1, red knight deployed at tick 0: spawn
    tick 21, deploy until tick 41): the cast fires during tick 20, the roll spawns
    the same tick and starts moving at tick 21. The 2.5-tile minDistance gate
    opens on the 15th motion tick (tick 35 — the float32 travel sum lands exactly
    on 2500.0). The knight at (4.0, 12.8) first comes within the 2.45-tile hit
    radius at tick 34; the gate must block that hit and let the single hit land
    at tick 35: scaleCard(105) = 268 damage, directional pushback 700 (35 game
    units per tick for 10 ticks), roll deactivates at range end (tick 81)."""
    sim = _sim(1)
    deck = [_C(sim, n) for n in LOG_DECK]
    sim.set_deck(0, deck)
    sim.set_deck(1, deck)
    kni = _C(sim, "Knight")
    sim.deploy(1, torch.tensor([kni]), torch.tensor([4.0]), torch.tensor([12.8]))
    ok = sim.play(0, 2, torch.tensor([4.0]), torch.tensor([17.05]))
    assert ok.tolist() == [True], "log (2 elixir) must play"
    s = sim.s
    kn = 0  # unit slot of the knight (first placed)
    sim.tick(20)
    # cast fired at tick 20: the roll exists AT the deploy point + 3 units forward
    rolls = [j for j in range(s.p_active.shape[1])
             if bool(s.p_active[0, j]) and bool(s.p_pierce[0, j])]
    assert len(rolls) == 1, "exactly one rolling projectile after the sync"
    j = rolls[0]
    assert float(s.p_fx[0, j]) / 65536 == 4000.0, "roll starts at the deploy point x"
    assert float(s.p_fy[0, j]) / 65536 == 17047.0, "roll starts 3 game units forward of the cast point"
    assert abs(float(s.p_ox[0, j]) * 1000 - 4000) < 1e-3  # origin = deploy point
    assert abs(float(s.p_oy[0, j]) * 1000 - 17050) < 1e-3
    assert float(s.p_step[0, j]) == 166.6666717529297, "float32 speed*dt step"
    assert float(s.p_range[0, j]) == 10100.0 and float(s.p_mind[0, j]) == 2500.0
    assert float(s.p_dmg[0, j]) == 268.0, "floor(105 * 2.56) at level 11"
    assert float(s.p_diry[0, j]) == -1.0  # blue rolls toward the enemy side
    sim.tick(13)  # -> tick 33: knight spawned (tick 21) in its deploy
    hp0 = float(s.u_hp[0, kn])
    assert abs(hp0 - 1766.0) < 1e-3, f"knight hp={hp0}"
    sim.tick(1)  # -> tick 34: first tick inside the hit radius, gate still closed
    assert float(s.u_hp[0, kn]) == hp0, "no hits before minDistance (2500) is traveled"
    sim.tick(1)  # -> tick 35: gate opens, single hit lands
    assert abs(float(s.u_hp[0, kn]) - (hp0 - 268.0)) < 1e-3
    assert float(s.u_kb_time[0, kn]) > 0.0, "directional knockback applies"
    # knockback: 700 / 1.0 = 700 units/s for 0.5 s = 35 units/tick, 10 ticks
    sim.tick(9)  # -> tick 44: 10 knockback ticks total (35..44)
    assert float(s.u_kb_time[0, kn]) <= 0.0
    assert abs(float(s.u_y[0, kn]) - 12.45) < 1e-6, f"y={float(s.u_y[0, kn])}"
    sim.tick(1)  # -> tick 45: walking resumes (+0.05)
    assert abs(float(s.u_y[0, kn]) - 12.50) < 1e-6
    # no second hit (hit-once per entity) and hp stable through the roll's life
    sim.tick(36)  # -> tick 81: roll deactivates at range end (61st motion tick)
    assert float(s.u_hp[0, kn]) == hp0 - 268.0, "each entity is hit at most once"
    assert not bool(s.p_active[0, j]), "roll deactivates at range end (~tick 81)"
    assert abs(float(s.p_fy[0, j]) / 65536 - 6880.0) < 2.0, \
        f"final whole-unit y={float(s.p_fy[0, j]) / 65536}"


def test_log_air_immune_and_crown_tower_damage():
    """M3.6: the rolling log (aoeToGround, hitsAir false) passes the flying red
    MegaMinion in its path without touching it, and lands 15 % crown damage on
    the red left princess tower: floor(268 * (100 - 85) / 100) = 40."""
    sim = _sim(1)
    deck = [_C(sim, n) for n in LOG_DECK]
    sim.set_deck(0, deck)
    sim.set_deck(1, deck)
    mm = _C(sim, "MegaMinion")
    sim.deploy(1, torch.tensor([mm]), torch.tensor([5.0]), torch.tensor([9.0]))
    ok = sim.play(0, 2, torch.tensor([4.0]), torch.tensor([17.05]))
    assert ok.tolist() == [True]
    s = sim.s
    hp_seen = None
    min_dist = 1e18
    for _ in range(81):  # through the full roll lifetime (tick 81)
        sim.tick(1)
        rolls = [j for j in range(s.p_active.shape[1])
                 if bool(s.p_active[0, j]) and bool(s.p_pierce[0, j])]
        if not rolls:
            continue
        j = rolls[0]
        rx = float(s.p_fx[0, j]) / 65536
        ry = float(s.p_fy[0, j]) / 65536
        for sl in range(64):
            if bool(s.u_active[0, sl]) and int(s.u_side[0, sl]) == 1:
                d = ((float(s.u_x[0, sl]) * 1000 - rx) ** 2
                     + (float(s.u_y[0, sl]) * 1000 - ry) ** 2) ** 0.5
                min_dist = min(min_dist, d)
                hp = float(s.u_hp[0, sl])
                assert hp_seen is None or hp >= hp_seen, \
                    f"air unit slot {sl} must not take log damage"
                hp_seen = hp
    assert hp_seen is not None, "the MegaMinion must be in the air"
    assert min_dist <= 2550.0, f"air unit must have been inside the hit radius: {min_dist}"
    # crown-tower damage: exactly 40 (15 % of 268), landed while the roll passes
    assert abs(float(s.tower_hp[0, 4]) - (3052.0 - 40.0)) < 1e-3, \
        f"red left princess: {float(s.tower_hp[0, 4])}"


def test_knockback_resets_attack_and_blocks_attacks():
    """M3.6 (Java LogSpellTest.knockback_resetsAttackAnimation): an entity being
    knocked back resets its in-progress attack and cannot attack while the
    displacement lasts (CombatSystem.processEntityCombat early return)."""
    sim = _sim(1)
    deck = [_C(sim, n) for n in LOG_DECK]
    sim.set_deck(0, deck)
    sim.set_deck(1, deck)
    kni = _C(sim, "Knight")
    sim.deploy(0, torch.tensor([kni]), torch.tensor([4.0]), torch.tensor([15.0]))
    sim.deploy(1, torch.tensor([kni]), torch.tensor([4.0]), torch.tensor([12.0]))
    s = sim.s
    red = 1  # red knight spawned into slot 1 after the blue one in slot 0
    for _ in range(200):
        sim.tick(1)
        if bool(s.u_attacking[0, red]):
            break
    assert bool(s.u_attacking[0, red]), "red knight must reach attack windup"
    sim._start_knockback(0, red, 0.0, -1.0, 700.0)
    blue_hp = float(s.u_hp[0, 0])
    sim.tick(1)
    assert not bool(s.u_attacking[0, red]), "attack state must be reset by knockback"
    assert float(s.u_windup[0, red]) == 0.0
    sim.tick(9)  # remaining knockback ticks
    assert float(s.u_hp[0, 0]) == blue_hp, "no attack lands during the knockback"
    assert float(s.u_kb_time[0, red]) <= 0.0


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


def test_time_limit_boundaries_match_java():
    """M3.7: the time-limit checks must land on the same tick as the Java
    reference. `GameEngine.checkTimeLimit` runs at step 13 while the frame
    counter increments at the end of the tick, so the first check that reads
    frame >= 3600 runs during tick 3601 — the match ends at elapsed 180.05 s,
    not 180.00 s; same shift for the overtime end (300.05 s)."""
    # (a) full-length run: blue leads 1-0 at the horn -> match ends at tick 3601
    sim = _sim(1)
    sim.s.crowns[0, 0] = 1
    sim.tick(3600)
    assert abs(float(sim.s.time[0]) - 180.0) < 0.05, "180.00 s = tick 3600"
    assert not bool(sim.s.game_over[0]), "must not end during tick 3600 (Java ends in 3601)"
    sim.tick(1)
    assert bool(sim.s.game_over[0]) and int(sim.s.winner[0]) == 0

    # (b) tied at the horn -> overtime; OT ends at 300.05 with the HP tiebreak
    sim2 = _sim(1)
    sim2.s.time[0] = 179.95
    sim2.tick(2)  # 180.05
    assert not bool(sim2.s.game_over[0]), "tied crowns -> overtime, no end"
    sim2.s.tower_hp[0, 4] = 1000.0  # blue now leads on total tower HP
    sim2.s.time[0] = 299.95
    sim2.tick(1)  # 300.00
    assert not bool(sim2.s.game_over[0]), "must not end during tick 6000 (Java ends in 6001)"
    sim2.tick(1)  # 300.05
    assert bool(sim2.s.game_over[0]) and int(sim2.s.winner[0]) == 0, "HP tiebreak -> blue"

    # (c) equal crowns and equal tower HP at OT end -> draw (winner == 2)
    sim3 = _sim(1)
    sim3.s.time[0] = 299.95
    sim3.tick(2)
    assert bool(sim3.s.game_over[0]) and int(sim3.s.winner[0]) == 2


def test_elixir_phase_boundaries_match_java():
    """M3.7: x2 regen starts with the tick that ends at 120.10 s and x3 at
    240.10 s (Java: enterDoubleElixir fires during tick 2401 step 13, after
    that tick's regen at step 2, so it only affects the next tick)."""
    sim = _sim(1)
    sim.s.time[0] = 119.95
    sim.s.elixir[0, 0] = 3.0
    sim.tick(1)  # 120.00
    e0 = float(sim.s.elixir[0, 0])
    sim.tick(1)  # 120.05 -> still x1
    e1 = float(sim.s.elixir[0, 0])
    sim.tick(1)  # 120.10 -> first x2 tick
    e2 = float(sim.s.elixir[0, 0])
    assert abs((e1 - e0) - 0.05 / 2.8) < 1e-4, "tick at 120.05 must still be x1"
    assert abs((e2 - e1) - 2 * 0.05 / 2.8) < 1e-4, "tick at 120.10 must be x2"

    sim.s.time[0] = 239.95
    sim.s.elixir[0, 0] = 3.0
    sim.tick(1)  # 240.00 -> x2
    a = float(sim.s.elixir[0, 0])
    sim.tick(1)  # 240.05 -> x2
    b = float(sim.s.elixir[0, 0])
    sim.tick(1)  # 240.10 -> first x3 tick
    c = float(sim.s.elixir[0, 0])
    assert abs((b - a) - 2 * 0.05 / 2.8) < 1e-4, "tick at 240.00 must still be x2"
    assert abs((c - b) - 3 * 0.05 / 2.8) < 1e-4, "tick at 240.10 must be x3"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
