"""Vectorized RL interface over BatchedCRSim (milestone M5 — v0).

Training-facing API: reset / step / action_masks over a batch of parallel
battles, shaped for a PPO trainer (SB3-style).

Action space (61 discrete actions per env):
    0            = no-op
    1..60        = play: 1 + slot*15 + zone   (slot 0..3 = hand slot, zone 0..14)

Zone table (approximates the Java stack's 15 deployment zones):
    zones 0-6  : own half (blue);  zones 7-14 : enemy half (red)

Observation (float32, 421 per env) — layout is a STABLE CONTRACT:
    [0] own elixir / 10          [5 + i*6 + k] for i in 0..63: unit slots
    [1] enemy elixir / 10            k: 0 active, 1 side, 2 x/18, 3 y/32,
    [2] time / 180                        4 hp fraction, 5 unit idx / 130
    [3] own crowns / 3           [389 + j*4 + k] for j in 0..5: towers
    [4] enemy crowns / 3             k: 0 alive, 1 hp fraction, 2 x/18, 3 y/32
                                 [413 + s*2 + k] for s in 0..3: hand slots
                                     k: 0 cost / 10, 1 card idx / 250

Reward v0 (documented approximation of the Java RewardCalculator; tune in M4):
    +0.005 * (enemy tower hp lost) - 0.005 * (own tower hp lost)
    +10 * crowns gained by the blue agent, -10 * crowns conceded
    +30 win / -10 loss (terminal step)
Done: game over or time >= 300 s. No auto-reset in v0 (caller resets the batch).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .env import KING_RADIUS, T_KING_HP, T_PRINCESS_HP, BatchedCRSim, make_sim

STEP_TICKS = 15          # one env step = 0.75 s (parity with the Java training stack)
OBS_DIM = 421
N_ACTIONS = 1 + 4 * 15   # noop + 4 slots x 15 zones

ZONES = [
    (3.5, 22.0), (9.0, 22.0), (14.5, 22.0),      # 0-2 own half back
    (3.5, 18.5), (9.0, 18.5), (14.5, 18.5),      # 3-5 own half front
    (9.0, 25.5),                                  # 6 own king area
    (3.5, 12.0), (9.0, 12.0), (14.5, 12.0),      # 7-9 enemy front
    (3.5, 8.0), (9.0, 8.0), (14.5, 8.0),         # 10-12 enemy mid
    (3.5, 6.0), (14.5, 6.0),                      # 13-14 enemy tower-adjacent
]


@dataclass
class VecStep:
    obs: torch.Tensor        # (B, OBS_DIM)
    reward: torch.Tensor     # (B,)
    done: torch.Tensor       # (B,) bool
    info: dict


class GPUSimVecEnv:
    """Lockstep vectorized env: B battles, one discrete action per env per step."""

    def __init__(self, data_dir: str, num_envs: int, device: str = "cpu",
                 level: int = 11, deck_blue: list[int] | None = None,
                 deck_red: list[int] | None = None):
        self.b = num_envs
        self.device = device
        self.sim: BatchedCRSim = make_sim(data_dir, batch_size=num_envs, device=device, level=level)
        self.deck_blue = deck_blue
        self.deck_red = deck_red
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self) -> torch.Tensor:
        self.sim.reset()
        if self.deck_blue is not None:
            self.sim.set_deck(0, self.deck_blue)
        if self.deck_red is not None:
            self.sim.set_deck(1, self.deck_red)
        s = self.sim.s
        self._prev_hp = self._tower_totals()
        self._prev_crowns = s.crowns.clone()
        return self._obs()

    def _tower_totals(self):
        s = self.sim.s
        own = torch.where(s.tower_alive[:, :3], s.tower_hp[:, :3], torch.zeros_like(s.tower_hp[:, :3])).sum(1)
        ene = torch.where(s.tower_alive[:, 3:], s.tower_hp[:, 3:], torch.zeros_like(s.tower_hp[:, 3:])).sum(1)
        return own.clone(), ene.clone()

    # ------------------------------------------------------------ action masks
    def action_masks(self) -> torch.Tensor:
        """(B, 61) bool: no-op always allowed; a play needs a non-empty affordable
        hand slot and (troops only) a legal deployment zone."""
        s = self.sim.s
        dev = self.sim.device
        masks = torch.zeros(self.b, N_ACTIONS, dtype=torch.bool, device=dev)
        masks[:, 0] = True
        zone_xy = torch.tensor(ZONES, dtype=torch.float32, device=dev)
        for slot in range(4):
            card = s.hand[:, 0, slot]
            cclamp = card.clamp(0, len(self.sim.t.names) - 1)
            ok = card >= 0
            if not bool(ok.any()):
                continue
            afford = s.elixir[:, 0] >= self.sim._costs[cclamp] - 1e-6
            is_spell = self.sim.t.card_types[cclamp] == 1
            for z in range(15):
                x = zone_xy[z, 0].expand(self.b)
                y = zone_xy[z, 1].expand(self.b)
                zok = torch.where(is_spell, torch.ones_like(ok), self.sim._zone_ok(0, x, y))
                masks[:, 1 + slot * 15 + z] = ok & afford & zok
        return masks

    # ------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor) -> VecStep:
        assert actions.shape == (self.b,)
        dev = self.sim.device
        a = actions.to(dev)
        noop = a <= 0
        idx = (a - 1).clamp(min=0)
        slot = (idx // 15).long()
        zone = (idx % 15).long()
        ztab = torch.tensor(ZONES, dtype=torch.float32, device=dev)
        zx = ztab[zone, 0]
        zy = ztab[zone, 1]

        # one play() call per hand slot, restricted to the envs that chose it
        for sl in range(4):
            sel = (~noop) & (slot == sl)
            if bool(sel.any()):
                self.sim.play(0, sl, zx, zy, env_mask=sel)

        self.sim.tick(STEP_TICKS)

        s = self.sim.s
        own, ene = self._tower_totals()
        rew = 0.005 * (self._prev_hp[1] - ene) - 0.005 * (own - self._prev_hp[0])
        rew = rew + 10.0 * (s.crowns[:, 0] - self._prev_crowns[:, 0]).float()
        rew = rew - 10.0 * (s.crowns[:, 1] - self._prev_crowns[:, 1]).float()
        done = s.game_over | (s.time >= 300.0)
        win = done & (s.winner == 0)
        lose = done & (s.winner == 1)
        rew = rew + torch.where(win, torch.full_like(rew, 30.0),
                                torch.where(lose, torch.full_like(rew, -10.0), torch.zeros_like(rew)))
        self._prev_hp = (own.clone(), ene.clone())
        self._prev_crowns = s.crowns.clone()
        return VecStep(obs=self._obs(), reward=rew, done=done, info={})

    # -------------------------------------------------------------------- obs
    def _obs(self) -> torch.Tensor:
        s = self.sim.s
        t = self.sim.t
        b, dev = self.b, self.sim.device
        head = torch.stack([
            s.elixir[:, 0] / 10.0,
            s.elixir[:, 1] / 10.0,
            s.time / 180.0,
            s.crowns[:, 0].float() / 3.0,
            s.crowns[:, 1].float() / 3.0,
        ], dim=1)
        hp_max = self.sim._hp[s.u_unit].clamp(min=1.0)
        hp_frac = (s.u_hp / hp_max).clamp(0.0, 1.0)
        units = torch.stack([
            s.u_active.float(),
            s.u_side.float(),
            s.u_x / 18.0,
            s.u_y / 32.0,
            hp_frac,
            s.u_unit.float() / 130.0,
        ], dim=2).reshape(b, -1)
        tower_full = torch.where(self.sim.tower_king, torch.full_like(s.tower_hp, T_KING_HP),
                                 torch.full_like(s.tower_hp, T_PRINCESS_HP))
        towers = torch.stack([
            s.tower_alive.float(),
            (s.tower_hp / tower_full).clamp(0.0, 1.0),
            self.sim.tower_pos[:, :, 0] / 18.0,
            self.sim.tower_pos[:, :, 1] / 32.0,
        ], dim=2).reshape(b, -1)
        hand0 = s.hand[:, 0, :]                       # blue's hand only (B,4)
        cclamp = hand0.clamp(0, len(t.names) - 1)
        hand = torch.stack([
            self.sim._costs[cclamp] / 10.0,
            hand0.float().clamp(min=0.0) / 250.0,
        ], dim=2).reshape(b, -1)
        obs = torch.cat([head, units, towers, hand], dim=1)
        assert obs.shape[1] == OBS_DIM, f"obs dim {obs.shape[1]} != {OBS_DIM}"
        return obs
