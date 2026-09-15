"""M5.1b/M5.2 — Gymnasium / Stable-Baselines3 adapters for the batched sim.

Integration levels (keep this thin — the sim owns the physics, SB3 owns the RL):

  * `GPUSimGymEnv` — a single-battle `gymnasium.Env` around `GPUSimVecEnv`
    (num_envs=1). It exposes `action_masks()` (the method sb3-contrib's
    `MaskablePPO` needs), so it can be used directly, wrapped in
    `sb3_contrib.common.wrappers.ActionMasker`, or placed inside SB3 vector
    envs where masks are collected via `vec_env.env_method("action_masks")`.

  * `make_sb3_vec_env(n_envs, ...)` — an SB3 `DummyVecEnv` of `n_envs`
    independent single battles (one `BatchedCRSim` per env, B=1 each). SB3's
    VecEnv owns the episode lifecycle: it auto-resets finished battles, so the
    env itself deliberately does NOT auto-reset.

  * `GPUSimSB3VecEnv` — the throughput path: an SB3 `VecEnv` over ONE batched
    lockstep `GPUSimVecEnv`. Finished battles are auto-reset individually via
    the sim's partial reset, so neighbours keep playing. Masks are exposed
    through `env_method("action_masks")` (MaskablePPO calls this itself).

Obs/reward/action semantics are exactly those of `gpusim/vecenv.py` (obs 421,
61 actions, reward v0). Determinism: the sim is a pure function of
(state, actions, tick); with a fixed seed and identical action sequences two
runs are bit-identical (tested in test_vecenv/test_sb3).

Smoke: `python -m gpusim.train_sb3_smoke` · Trainer: `python -m gpusim.train_gpu`.
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .vecenv import N_ACTIONS, OBS_DIM, GPUSimVecEnv

DEFAULT_DECK_NAMES = ["Knight", "Archer", "Giant", "Musketeer",
                      "Fireball", "Zap", "Log", "Cannon"]

OPPONENTS = ("passive", "random")


def deck_from_names(data_dir: str, names: list[str]) -> list[int]:
    """Card indices for a list of English card names (see gpusim/cards.py)."""
    from .cards import load_tables

    t = load_tables(data_dir)
    return [t.card_index[n.lower()] for n in names]


def default_data_dir() -> str:
    return os.environ.get("GPUSIM_DATA", "fidelity/patched")


class GPUSimGymEnv(gym.Env):
    """One Clash Royale battle as a Gymnasium env (blue = the learning side)."""

    metadata = {"render_modes": []}

    def __init__(self, data_dir: str | None = None, deck: list[int] | None = None,
                 deck_names: list[str] | None = None, deck_red: list[int] | None = None,
                 level: int = 11):
        super().__init__()
        self.data_dir = data_dir or default_data_dir()
        if deck is None:
            deck = deck_from_names(self.data_dir, deck_names or DEFAULT_DECK_NAMES)
        self.deck = list(deck)
        self.venv = GPUSimVecEnv(self.data_dir, num_envs=1, device="cpu", level=level,
                                 deck_blue=self.deck, deck_red=deck_red if deck_red is not None else self.deck)
        self.action_space = spaces.Discrete(N_ACTIONS)
        # every obs field is a normalized ratio in [0, 1] except time/180,
        # which peaks at 300/180 = 1.67 in overtime -> [0, 2] is a safe bound.
        self.observation_space = spaces.Box(low=0.0, high=2.0,
                                            shape=(OBS_DIM,), dtype=np.float32)

    # ---------------------------------------------------------------- masking
    def action_masks(self) -> np.ndarray:
        """(N_ACTIONS,) bool — legal actions for the current state (SB3 contract)."""
        return self.venv.action_masks()[0].cpu().numpy()

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.venv.reset()
        return obs[0].cpu().numpy().astype(np.float32), {}

    def step(self, action: int):
        a = torch.as_tensor([int(action)], dtype=torch.long)
        out = self.venv.step(a)
        obs = out.obs[0].cpu().numpy().astype(np.float32)
        rew = float(out.reward[0])
        done = bool(out.done[0])
        info: dict = {}
        if done:
            s = self.venv.sim.s
            info["winner"] = int(s.winner[0])
            info["crowns"] = (int(s.crowns[0, 0]), int(s.crowns[0, 1]))
            info["match_time"] = float(s.time[0])
        # terminated only: a match ends by game rules (king down / 3:00 crown
        # lead / 5:00 overtime decision); there is no external time limit.
        return obs, rew, done, False, info

    def render(self):
        return None


def make_sb3_vec_env(n_envs: int = 4, data_dir: str | None = None,
                     deck: list[int] | None = None, deck_names: list[str] | None = None,
                     level: int = 11, seed: int | None = None):
    """SB3 `DummyVecEnv` of n_envs independent battles (auto-reset by SB3)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make():
        return GPUSimGymEnv(data_dir=data_dir, deck=deck, deck_names=deck_names, level=level)

    vec = DummyVecEnv([_make for _ in range(n_envs)])
    if seed is not None:
        vec.seed(seed)
    return vec


class GPUSimSB3VecEnv(VecEnv):
    """SB3 `VecEnv` over ONE batched `GPUSimVecEnv` (lockstep, B envs at once).

    The whole batch advances tick-by-tick inside a single `BatchedCRSim`, so
    this scales to thousands of parallel battles on a GPU. Finished battles are
    auto-reset individually (partial reset) and reported through the standard
    SB3 `infos` (``episode``/``terminal_observation``/``TimeLimit.truncated``).

    `opponent`: "passive" (red never plays) or "random" (red samples uniformly
    from its legal masked actions each step — the fixed-deck baseline opponent).
    Blue drives the policy; both sides play in the same pre-tick window.
    """

    render_mode = None

    def __init__(self, data_dir: str | None = None, num_envs: int = 16, device: str = "cpu",
                 deck: list[int] | None = None, deck_names: list[str] | None = None,
                 level: int = 11, opponent: str = "random", seed: int | None = None):
        assert opponent in OPPONENTS, f"opponent must be one of {OPPONENTS}"
        data_dir = data_dir or default_data_dir()
        if deck is None:
            deck = deck_from_names(data_dir, deck_names or DEFAULT_DECK_NAMES)
        self.venv = GPUSimVecEnv(data_dir, num_envs=num_envs, device=device, level=level,
                                 deck_blue=deck, deck_red=deck)
        self.deck = list(deck)
        self.opponent = opponent
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(int(seed))
        self._actions = np.zeros(num_envs, dtype=np.int64)
        self._ep_rew = np.zeros(num_envs, dtype=np.float64)
        self._ep_len = np.zeros(num_envs, dtype=np.int64)
        # cumulative match results for logging: [blue wins, red wins, draws]
        self.results = np.zeros(3, dtype=np.int64)
        obs_space = spaces.Box(low=0.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)
        act_space = spaces.Discrete(N_ACTIONS)
        super().__init__(num_envs, obs_space, act_space)

    # ------------------------------------------------------------- SB3 VecEnv
    def reset(self):
        self._ep_rew[:] = 0.0
        self._ep_len[:] = 0
        return self.venv.reset().cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        a = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        self._actions = a

    def _opponent_actions(self) -> torch.Tensor:
        if self.opponent == "passive":
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.venv.sim.device)
        masks = self.venv.action_masks(side=1).float()
        return torch.multinomial(masks, 1, generator=self._gen).squeeze(1)

    def step_wait(self):
        self.venv.apply_actions(1, self._opponent_actions())
        out = self.venv.step(torch.as_tensor(self._actions))
        obs = out.obs.cpu().numpy()
        rewards = out.reward.cpu().numpy().astype(np.float64)
        dones = out.done.cpu().numpy()
        self._ep_rew += rewards
        self._ep_len += 1

        infos: list[dict] = [{} for _ in range(self.num_envs)]
        for info in infos:
            info["TimeLimit.truncated"] = False
        done_rows = np.flatnonzero(dones)
        if done_rows.size:
            s = self.venv.sim.s
            for i in done_rows.tolist():
                infos[i]["terminal_observation"] = obs[i].copy()
                winner = int(s.winner[i])
                infos[i]["winner"] = winner
                infos[i]["episode"] = {"r": float(self._ep_rew[i]),
                                       "l": int(self._ep_len[i])}
                if winner == 0:
                    self.results[0] += 1
                elif winner == 1:
                    self.results[1] += 1
                else:
                    self.results[2] += 1
            mask = torch.zeros(self.num_envs, dtype=torch.bool)
            mask[done_rows] = True
            fresh = self.venv.reset(env_mask=mask).cpu().numpy()
            obs = obs.copy()
            obs[done_rows] = fresh[done_rows]
            self._ep_rew[done_rows] = 0.0
            self._ep_len[done_rows] = 0
        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._gen.manual_seed(int(seed))
        return [seed] * self.num_envs

    # masks for MaskablePPO: env_method("action_masks") -> list of (61,) bool
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            side = int(method_kwargs.get("side", 0))
            masks = self.venv.action_masks(side=side).cpu().numpy()
            idx = list(self._get_indices(indices))
            return [masks[i] for i in idx]
        raise AttributeError(f"{type(self).__name__} does not expose env_method({method_name!r})")

    def has_attr(self, attr_name: str) -> bool:
        return attr_name in ("action_masks", "num_envs", "venv", "render_mode", "results")

    def get_attr(self, attr_name: str, indices=None):
        idx = list(self._get_indices(indices))
        if attr_name == "render_mode":
            return [self.render_mode for _ in idx]
        if attr_name == "num_envs":
            return [self.num_envs for _ in idx]
        if attr_name == "venv":
            return [self.venv for _ in idx]
        if attr_name == "results":
            return [self.results.copy() for _ in idx]
        if attr_name == "action_masks":
            return self.env_method("action_masks", indices=indices)
        raise AttributeError(f"{type(self).__name__} has no attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        if attr_name == "opponent":
            assert value in OPPONENTS, f"opponent must be one of {OPPONENTS}"
            self.opponent = value
            return
        raise AttributeError(f"{type(self).__name__} cannot set attribute {attr_name!r}")

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        idx = list(self._get_indices(indices))
        return [False for _ in idx]
