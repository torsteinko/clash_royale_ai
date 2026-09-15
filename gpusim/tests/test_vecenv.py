"""M5 v0 tests: vectorized env interface (obs contract, masks, stepping).

Run:  python3 -m gpusim.tests.test_vecenv
"""
from __future__ import annotations

import os

import torch

from gpusim.vecenv import N_ACTIONS, OBS_DIM, GPUSimVecEnv

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")
DECK = ["Knight", "Archer", "Giant", "Musketeer", "Fireball", "Zap", "Log", "Cannon"]


def _vv(b=4):
    from gpusim.cards import load_tables

    t = load_tables(DATA)
    deck = [t.card_index[n.lower()] for n in DECK]
    return GPUSimVecEnv(DATA, num_envs=b, deck_blue=deck, deck_red=deck)


def test_obs_contract_and_masks():
    venv = _vv()
    obs = venv.reset()
    assert obs.shape == (4, OBS_DIM)
    masks = venv.action_masks()
    assert masks.shape == (4, N_ACTIONS)
    assert bool(masks[:, 0].all()), "no-op must always be legal"
    assert int(masks.sum()) > 4, "some plays must be legal at 5 elixir"
    # zone mask: own-half zone 4 (9.0, 18.5) legal for troops at start
    assert bool(masks[:, 1 + 0 * 15 + 4].all()), "own-half troop place must be legal"
    # enemy-half zone 8 (9.0, 12.0) illegal before any pocket opens
    assert not bool(masks[:, 1 + 0 * 15 + 8].any()), "enemy-half troop place must be illegal"


def test_stepping_and_rewards_finite():
    venv = _vv()
    venv.reset()
    g = torch.Generator().manual_seed(7)
    for _ in range(80):
        m = venv.action_masks()
        acts = torch.tensor([int(torch.multinomial(m[i].float(), 1, generator=g)[0]) for i in range(4)])
        out = venv.step(acts)
        assert out.obs.shape == (4, OBS_DIM)
        assert torch.isfinite(out.reward).all()
        assert out.done.shape == (4,)
    # after 80 steps (60s) something must have happened: some elixir spent or units alive
    assert bool(venv.sim.s.u_active.any()) or abs(float(venv.sim.s.elixir[0, 0]) - 5.0) > 0.1


def test_determinism():
    a = _vv()
    b = _vv()
    a.reset()
    b.reset()
    g = torch.Generator().manual_seed(3)
    seq = []
    for _ in range(40):
        m = a.action_masks()
        acts = torch.tensor([int(torch.multinomial(m[i].float(), 1, generator=g)[0]) for i in range(4)])
        seq.append(acts)
    out_a = None
    for acts in seq:
        out_a = a.step(acts)
    out_b = None
    for acts in seq:
        out_b = b.step(acts)
    assert torch.equal(out_a.obs, out_b.obs), "identical actions must give identical obs"
    assert torch.equal(out_a.reward, out_b.reward)


def test_partial_reset_only_masked_rows():
    """M5 auto-reset hook: resetting selected rows restarts them exactly like a
    fresh battle (state + deck + masks) while neighbours keep playing."""
    from gpusim.env import ELIXIR_START

    venv = _vv(b=2)
    fresh = _vv(b=1)
    fresh_obs = fresh.reset()[0]
    venv.reset()
    # spend elixir + deploy on both rows, then advance
    venv.apply_actions(0, torch.tensor([1 + 0 * 15 + 4, 1 + 1 * 15 + 1]))
    venv.sim.tick(45)
    t_before = float(venv.sim.s.time[1])
    assert t_before > 2.0

    obs = venv.reset(env_mask=torch.tensor([True, False]))
    s = venv.sim.s
    # row 0: fully fresh (state, deck, cycle) and obs identical to a new battle
    assert float(s.time[0]) == 0.0
    assert float(s.elixir[0, 0]) == ELIXIR_START
    assert not s.u_active[0].any() and not s.p_active[0].any()
    assert s.hand[0, 0].tolist() == fresh.sim.s.hand[0, 0].tolist()
    assert torch.equal(obs[0], fresh_obs)
    # row 1: untouched progress, deck state preserved
    assert float(s.time[1]) == t_before
    assert abs(float(s.elixir[1, 0]) - ELIXIR_START) > 1e-6 or bool(s.u_active[1].any())

    # the batch keeps stepping normally after a partial reset
    out = venv.step(torch.tensor([0, 0]))
    assert out.obs.shape == (2, OBS_DIM)
    assert float(venv.sim.s.time[0]) > 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
