"""M5.1b tests: Gymnasium wrapper + SB3 MaskablePPO integration.

Run:  python3 -m gpusim.tests.test_sb3

The wrapper `GPUSimGymEnv` must satisfy the Gymnasium API (checked with
gymnasium's own `check_env`), expose `action_masks()` with the same values as
the batched vecenv, and support an actual MaskablePPO update + greedy predict
whose actions are always legal under the masks.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from gpusim.vecenv import N_ACTIONS, OBS_DIM, GPUSimVecEnv
from gpusim.sb3_env import DEFAULT_DECK_NAMES, GPUSimGymEnv, deck_from_names, make_sb3_vec_env

DATA = os.environ.get("GPUSIM_DATA", "fidelity/patched")


def test_gym_api_shapes_and_masks():
    env = GPUSimGymEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and isinstance(info, dict)
    m = env.action_masks()
    assert m.dtype == np.bool_ and m.shape == (N_ACTIONS,)
    assert m[0], "no-op must always be legal"
    assert m.sum() > 1, "some plays must be legal at 5 elixir"
    # step a legal non-noop action: 5-tuple, finite reward
    a = int(np.flatnonzero(m)[-1])
    obs2, rew, term, trunc, info2 = env.step(a)
    assert obs2.shape == (OBS_DIM,) and np.isfinite(rew)
    assert isinstance(term, bool) and trunc is False


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = GPUSimGymEnv()
    check_env(env, skip_render_check=True)


def test_masks_match_batched_vecenv():
    """GPUSimGymEnv is a B=1 view of GPUSimVecEnv: masks and obs must be identical."""
    env = GPUSimGymEnv()
    deck = deck_from_names(DATA, DEFAULT_DECK_NAMES)
    vec = GPUSimVecEnv(DATA, num_envs=1, deck_blue=deck, deck_red=deck)
    env.reset(seed=0)
    vec.reset()
    assert np.array_equal(env.action_masks(), vec.action_masks()[0].cpu().numpy())
    assert np.allclose(env.venv._obs()[0].cpu().numpy(), vec._obs()[0].cpu().numpy(), atol=0)


def test_maskable_ppo_smoke_and_predict():
    from sb3_contrib import MaskablePPO

    vec = make_sb3_vec_env(n_envs=2, seed=1)
    model = MaskablePPO("MlpPolicy", vec, seed=1, verbose=0,
                        n_steps=64, batch_size=32, n_epochs=1,
                        policy_kwargs=dict(net_arch=[32, 32]))
    model.learn(total_timesteps=128)
    assert model.num_timesteps == 128

    obs = vec.reset()
    masks = np.stack(vec.env_method("action_masks"))
    acts, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert acts.shape == (2,)
    assert all(bool(masks[i][int(a)]) for i, a in enumerate(acts)), "greedy actions must be legal"

    # save/load roundtrip preserves the deterministic prediction
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m")
        model.save(path)
        model2 = MaskablePPO.load(path, env=vec)
        acts2, _ = model2.predict(obs, action_masks=masks, deterministic=True)
        assert np.array_equal(acts, acts2)


def test_batched_vecenv_api_and_masks():
    """GPUSimSB3VecEnv: SB3 VecEnv API + masks via env_method (MaskablePPO path)."""
    from gpusim.sb3_env import GPUSimSB3VecEnv
    from stable_baselines3.common.vec_env import VecEnv

    vec = GPUSimSB3VecEnv(num_envs=4, seed=0, opponent="random")
    assert isinstance(vec, VecEnv)
    obs = vec.reset()
    assert obs.shape == (4, OBS_DIM) and obs.dtype == np.float32
    masks = vec.env_method("action_masks")
    assert isinstance(masks, list) and len(masks) == 4
    assert masks[0].shape == (N_ACTIONS,) and masks[0][0]
    stacked = np.stack(masks)
    out = vec.step(np.zeros(4, dtype=np.int64))
    obs2, rew, dones, infos = out
    assert obs2.shape == (4, OBS_DIM) and rew.shape == (4,) and dones.shape == (4,)
    assert len(infos) == 4 and "TimeLimit.truncated" in infos[0]
    assert stacked.shape == (4, N_ACTIONS)


def test_batched_vecenv_random_opponent_spends_elixir():
    from gpusim.env import ELIXIR_PERIOD, ELIXIR_START
    from gpusim.sb3_env import GPUSimSB3VecEnv

    # passive: red elixir is exactly start + regen after n steps
    n = 10
    passive = GPUSimSB3VecEnv(num_envs=1, seed=0, opponent="passive")
    passive.reset()
    for _ in range(n):
        passive.step(np.zeros(1, dtype=np.int64))
    red_passive = float(passive.venv.sim.s.elixir[0, 1])
    # float32 accumulation over 150 ticks -> use a loose epsilon
    assert abs(red_passive - (ELIXIR_START + n * 15 * 0.05 / ELIXIR_PERIOD)) < 1e-3

    # random: red must have paid for at least one card (or own units on the map)
    rand = GPUSimSB3VecEnv(num_envs=1, seed=0, opponent="random")
    rand.reset()
    for _ in range(n):
        rand.step(np.zeros(1, dtype=np.int64))
    s = rand.venv.sim.s
    red_units = bool((s.u_active & (s.u_side == 1)).any())
    red_elixir = float(s.elixir[0, 1])
    assert red_units or red_elixir < red_passive - 1.0, \
        f"random red did not play (elixir {red_elixir:.2f}, units {red_units})"


def test_batched_vecenv_auto_reset_partial():
    """A finished battle is auto-reset alone; neighbours keep their state."""
    from gpusim.sb3_env import GPUSimSB3VecEnv

    vec = GPUSimSB3VecEnv(num_envs=3, seed=0, opponent="passive")
    vec.reset()
    vec.step(np.zeros(3, dtype=np.int64))  # t = 0.75 everywhere
    s = vec.venv.sim.s
    s.game_over[0] = True
    s.winner[0] = 0
    s.time[0] = 300.0
    obs, rew, dones, infos = vec.step(np.zeros(3, dtype=np.int64))
    assert dones.tolist() == [True, False, False]
    info0 = infos[0]
    assert info0["winner"] == 0
    assert info0["episode"]["l"] == 2
    assert info0["terminal_observation"].shape == (OBS_DIM,)
    assert "episode" not in infos[1]
    # NOTE: a partial reset replaces the SimState object -> re-fetch s
    s = vec.venv.sim.s
    # row 0 restarted, row 1/2 kept playing
    assert float(s.time[0]) == 0.0 and float(s.elixir[0, 0]) == 5.0
    assert not s.u_active[0].any()
    assert float(s.time[1]) > 1.0 and float(s.time[2]) > 1.0
    # freshly restarted row has the deck re-applied (hand = first 4 deck cards)
    assert s.hand[0, 0].tolist() == vec.deck[:4]
    # and its observation equals a truly fresh battle
    from gpusim.sb3_env import GPUSimGymEnv

    fresh = GPUSimGymEnv()
    fresh_obs, _ = fresh.reset(seed=0)
    assert np.array_equal(obs[0], fresh_obs)


def test_played_card_leaves_hand_and_elixir_pays():
    """A masked-legal play must (a) rotate the played card out of the hand and
    (b) pay its elixir cost (minus 15 ticks of regeneration)."""
    from gpusim.env import ELIXIR_PERIOD

    env = GPUSimGymEnv()
    env.reset(seed=0)
    m = env.action_masks()
    plays = [i for i in range(1, N_ACTIONS) if m[i]]
    assert plays, "need at least one legal play"
    a = plays[0]
    slot = (a - 1) // 15
    hand_before = env.venv.sim.s.hand[0, 0].cpu().numpy().copy()
    card = int(hand_before[slot])
    assert card >= 0
    cost = float(env.venv.sim._costs[card])
    elixir_before = float(env.venv.sim.s.elixir[0, 0])

    obs, rew, term, trunc, info = env.step(a)

    hand_after = env.venv.sim.s.hand[0, 0].cpu().numpy()
    elixir_after = float(env.venv.sim.s.elixir[0, 0])
    assert card not in hand_after, "played card must rotate out of the hand"
    expected = elixir_before - cost + 15 * 0.05 / ELIXIR_PERIOD
    assert abs(elixir_after - expected) < 0.01, f"elixir {elixir_after} != {expected}"
    assert np.isfinite(rew)



if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
