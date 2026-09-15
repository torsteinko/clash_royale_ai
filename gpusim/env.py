"""Batched Clash Royale simulation core (tensor-based, device-agnostic).

v0 (milestone M1): tensor state, elixir economy, deployment with formation
offsets, level scaling, basic movement. Combat/pathing/spells land in M2/M3 —
divergences from the Java sim are tracked in DIVERGENCES.md and this module
must not silently approximate: TODO-marked mechanics are OFF until ported.

Coordinate system mirrors crforge: x in [0, 18], y in [0, 32] tiles,
origin top-left; blue (bottom) side y >= 16.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .cards import CardTable

MAX_UNITS = 256          # per env
TICK_DT = 0.05           # 20 ticks/s — matches the reference tick rate (verify in M4)
ELIXIR_START = 5.0
ELIXIR_MAX = 10.0
ELIXIR_PERIOD = 2.8      # seconds per elixir (single elixir phase)
MATCH_OT = 180.0         # regular time; overtime rules land in M3

# Tower geometry (crforge arena, tiles). Placeholders until M2 verifies.
TOWER_KING = {"blue": (9.0, 29.0), "red": (9.0, 3.0)}
TOWER_PRINCESS = {"blue": [(3.5, 25.5), (14.5, 25.5)], "red": [(3.5, 6.5), (14.5, 6.5)]}
TOWER_HP_KING = 4824.0        # verified L2 probe (fidelity)
TOWER_HP_PRINCESS = 3052.0


@dataclass
class SimState:
    """SoA tensor state. B = batch size."""

    elixir: torch.Tensor            # (B, 2) [blue, red]
    time: torch.Tensor              # (B,)
    tower_hp: torch.Tensor          # (B, 2, 3) [king, princess_l, princess_r] x [blue, red]

    u_active: torch.Tensor          # (B, MAX_UNITS) bool
    u_side: torch.Tensor            # (B, MAX_UNITS) 0=blue, 1=red
    u_card: torch.Tensor            # (B, MAX_UNITS) card idx
    u_unit: torch.Tensor            # (B, MAX_UNITS) unit idx
    u_hp: torch.Tensor              # (B, MAX_UNITS) current hp
    u_x: torch.Tensor               # (B, MAX_UNITS)
    u_y: torch.Tensor               # (B, MAX_UNITS)
    u_cd: torch.Tensor              # (B, MAX_UNITS) seconds until next attack

    spawn_pending: torch.Tensor     # (B, 2, 8) queued multi-unit deploys (card idx, -1 empty)
    spawn_timer: torch.Tensor       # (B, 2, 8)

    @property
    def batch_size(self) -> int:
        return int(self.elixir.shape[0])


class BatchedCRSim:
    """Batched simulator. All envs tick in lockstep."""

    def __init__(self, tables: CardTable, batch_size: int, device: str = "cpu", level: int = 11):
        self.t = tables
        self.device = torch.device(device)
        self.b = batch_size
        self.level = level
        self._mk = self.t.unit_of_card.to(self.device)
        self._costs = self.t.costs.to(self.device)
        self._hp = self.t.scaled(self.t.u_health, level).to(self.device)
        self.reset()

    # ---------------------------------------------------------------- reset
    def reset(self) -> SimState:
        b, dev = self.b, self.device
        self.s = SimState(
            elixir=torch.full((b, 2), ELIXIR_START, device=dev),
            time=torch.zeros(b, device=dev),
            tower_hp=torch.stack([
                torch.tensor([TOWER_HP_KING, TOWER_HP_PRINCESS, TOWER_HP_PRINCESS], device=dev),
                torch.tensor([TOWER_HP_KING, TOWER_HP_PRINCESS, TOWER_HP_PRINCESS], device=dev),
            ]).expand(b, 2, 3).clone(),
            u_active=torch.zeros(b, MAX_UNITS, dtype=torch.bool, device=dev),
            u_side=torch.zeros(b, MAX_UNITS, dtype=torch.long, device=dev),
            u_card=torch.full((b, MAX_UNITS), -1, dtype=torch.long, device=dev),
            u_unit=torch.zeros(b, MAX_UNITS, dtype=torch.long, device=dev),
            u_hp=torch.zeros(b, MAX_UNITS, device=dev),
            u_x=torch.zeros(b, MAX_UNITS, device=dev),
            u_y=torch.zeros(b, MAX_UNITS, device=dev),
            u_cd=torch.zeros(b, MAX_UNITS, device=dev),
            spawn_pending=torch.full((b, 2, 8), -1, dtype=torch.long, device=dev),
            spawn_timer=torch.zeros(b, 2, 8, device=dev),
        )
        return self.s

    # ------------------------------------------------------------ deployment
    def deploy(self, side: int, card_idx: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Deploy a card for `side` in every env where card_idx >= 0.

        Returns a (B,) bool mask of successful deployments (elixir paid).
        Multi-unit cards queue staggered spawns (summonDeployDelay).
        """
        assert side in (0, 1)
        s, dev = self.s, self.device
        card_idx = card_idx.to(dev)
        want = card_idx >= 0
        cost = self._costs[card_idx.clamp(min=0)]
        afford = s.elixir[:, side] >= cost - 1e-6
        ok = want & afford

        s.elixir[:, side] = torch.where(ok, s.elixir[:, side] - cost, s.elixir[:, side])
        if not bool(ok.any()):
            return ok

        # First unit spawns immediately at the deployment point; extras go to
        # the stagger queue with formation offsets (resolved in tick()).
        self._spawn_units(side, card_idx, x, y, ok)
        return ok

    def _spawn_units(self, side, card_idx, x, y, mask):
        s = self.s
        for e in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            ci = int(card_idx[e])
            count = int(self.t.spawn_count[ci])
            ui = int(self.t.unit_of_card[ci])
            offs = self.t.formation_offsets[ci]
            hp = float(self._hp[ui])
            for k in range(count):
                ox, oy = offs[k % len(offs)] if offs else (0.0, 0.0)
                slot = self._free_slot(e)
                if slot < 0:
                    break  # capacity: drop overflow (tracked via DIVERGENCES)
                if k == 0:
                    self._place(e, slot, side, ci, ui, hp, float(x[e]) + ox, float(y[e]) + oy)
                else:
                    # queue extra units (delay model per card)
                    self._queue_spawn(e, side, ci, float(x[e]) + ox, float(y[e]) + oy, k)

    def _free_slot(self, e: int) -> int:
        inactive = (~self.s.u_active[e]).nonzero(as_tuple=False).flatten()
        return int(inactive[0]) if inactive.numel() else -1

    def _place(self, e, slot, side, ci, ui, hp, xx, yy):
        s = self.s
        s.u_active[e, slot] = True
        s.u_side[e, slot] = side
        s.u_card[e, slot] = ci
        s.u_unit[e, slot] = ui
        s.u_hp[e, slot] = hp
        s.u_x[e, slot] = xx
        s.u_y[e, slot] = yy
        s.u_cd[e, slot] = self.t.u_cooldown[ui]

    def _queue_spawn(self, e, side, ci, xx, yy, k):
        s = self.s
        q = (s.spawn_pending[e, side] == -1).nonzero(as_tuple=False).flatten()
        if q.numel() == 0:
            return
        j = int(q[0])
        s.spawn_pending[e, side, j] = ci
        # position stored by convention: k-th queued unit spawns at deploy pos
        s.spawn_timer[e, side, j] = k * float(self.t.summon_delay[ci]) + TICK_DT
        self._queue_pos = getattr(self, "_queue_pos", {})
        self._queue_pos[(e, side, j)] = (xx, yy)

    # ------------------------------------------------------------------ tick
    def tick(self, n: int = 1, dt: float = TICK_DT) -> None:
        for _ in range(n):
            self._tick_once(dt)

    def _tick_once(self, dt: float) -> None:
        s = self.s
        s.time += dt
        # elixir regen (single-elixir phase; double/triple in M3)
        s.elixir = torch.clamp(s.elixir + dt / ELIXIR_PERIOD, max=ELIXIR_MAX)

        # staggered spawns
        due = s.spawn_timer <= 0
        pending = s.spawn_pending >= 0
        fired = due & pending
        if bool(fired.any()):
            for e, side, j in torch.nonzero(fired, as_tuple=False).tolist():
                ci = int(s.spawn_pending[e, side, j])
                ui = int(self.t.unit_of_card[ci])
                xx, yy = self._queue_pos.get((e, side, j), (9.0, 16.0 + 8.0 * (1 - side)))
                slot = self._free_slot(e)
                if slot >= 0:
                    self._place(e, slot, side, ci, ui, float(self._hp[ui]), xx, yy)
                s.spawn_pending[e, side, j] = -1
        s.spawn_timer = torch.clamp(s.spawn_timer - dt, min=0)

        # movement v0: march toward the enemy king tower (straight line).
        # TODO(M2): bridge waypoints + occupancy + collision. DIVERGENCE #1.
        if bool(s.u_active.any()):
            tgt_y = torch.where(s.u_side == 0,
                                torch.full_like(s.u_y, TOWER_KING["red"][1]),
                                torch.full_like(s.u_y, TOWER_KING["blue"][1]))
            tgt_x = torch.full_like(s.u_x, 9.0)
            dx, dy = tgt_x - s.u_x, tgt_y - s.u_y
            dist = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
            speed = self.t.u_speed.to(self.device)[s.u_unit]
            step = speed * dt / 20.0  # crforge speed units -> tiles/s (verify M4)
            s.u_x = torch.where(s.u_active, s.u_x + dx / dist * step, s.u_x)
            s.u_y = torch.where(s.u_active, s.u_y + dy / dist * step, s.u_y)


def make_sim(data_dir: str, batch_size: int, device: str = "cpu", level: int = 11) -> BatchedCRSim:
    from .cards import load_tables

    return BatchedCRSim(load_tables(data_dir, device=device), batch_size, device=device, level=level)
