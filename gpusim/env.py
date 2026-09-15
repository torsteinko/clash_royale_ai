"""Batched Clash Royale simulation core (tensor-based, device-agnostic).

M2 slice: targeting (nearest + lock/retention), melee/ranged combat with
cooldowns, tower attacks, deaths, crowns, time-limit decisions, double/triple
elixir phases. Ported 1:1 from the Java reference (TargetingSystem.java,
CombatSystem.java, GameEngine.java tick order, GameUnits.java speed formula)
except where noted in DIVERGENCES.md.

Coordinate system: x in [0, 18], y in [0, 32] tiles, origin top-left;
blue (bottom) side = y >= 16. Level-1 base values + level factor scaling.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from .cards import CardTable, level_factor

MAX_UNITS = 64           # per env (overflow tracked in DIVERGENCES.md #10)
MAX_PROJ = 128           # per env
N_TOWERS = 6             # slot order: [kingB, pB_l, pB_r, kingR, pR_l, pR_r]
TICK_DT = 0.05           # 20 ticks/s (GameEngine.TICKS_PER_SECOND = 20)
ELIXIR_START = 5.0
ELIXIR_MAX = 10.0
ELIXIR_PERIOD = 2.8      # seconds per elixir at rate 1
DOUBLE_ELIXIR_T = 120.0  # activated at match_end - 60s
TRIPLE_ELIXIR_T = 240.0  # overtime + 60s
MATCH_END_T = 180.0
MATCH_FINAL_T = 300.0    # overtime end (regular 180s + OT 120s)

# Tower stats — exact from reference Tower.java + LevelScaling (DEFAULT_TOWER_LEVEL 11)
T_KING_HP = 4824.0
T_PRINCESS_HP = 3052.0
T_PRINCESS_DMG = 109.0
T_PRINCESS_CD = 0.8
T_PRINCESS_RANGE = 7.5
T_RADIUS = 1.0            # princess collision radius
KING_RADIUS = 1.5
T_KING_DMG = 109.0
T_KING_CD = 1.0
T_KING_RANGE = 7.0        # Tower.CROWN_RANGE

# Arena / pathing constants (Arena.java + BasePathfinder.java)
RIVER_Y_MIN_T = 15.0      # tiles; river zone spans (15, 17)
RIVER_Y_MAX_T = 17.0
RIVER_CENTER_T = 16.0
BRIDGE_LX = 3.5           # left bridge center (spans x 2..5)
BRIDGE_RX = 14.5          # right bridge center (spans x 13..16)
BRIDGE_ALIGN_T = 1.0      # walk straight if within 1 tile of the bridge center
APPROACH_TOL_T = 0.2
BOUNDARY_EPS_T = 0.1

KING_POS = {"blue": (9.0, 29.0), "red": (9.0, 3.0)}
PRINCESS_POS = {"blue": [(3.5, 25.5), (14.5, 25.5)], "red": [(3.5, 6.5), (14.5, 6.5)]}

SIDE_HALF_Y = 16.0       # blue deploys at y >= 16, red at y <= 16 (river at 15-17)
RANGED_THRESHOLD = 2.0   # Combat.RANGED_THRESHOLD: range >= 2 tiles => ranged
TARGET_RETENTION = 1.5   # TargetingSystem.TARGET_RETENTION_RANGE_MULTIPLIER


@dataclass
class SimState:
    elixir: torch.Tensor        # (B, 2)
    time: torch.Tensor          # (B,)
    tower_hp: torch.Tensor      # (B, 6)
    tower_alive: torch.Tensor   # (B, 6) bool
    tower_cd: torch.Tensor      # (B, 6)
    king_active: torch.Tensor   # (B, 2) bool — king tower activated (attacks)
    crowns: torch.Tensor        # (B, 2) int
    game_over: torch.Tensor     # (B,) bool
    winner: torch.Tensor        # (B,) -1 none, 0 blue, 1 red, 2 draw

    u_active: torch.Tensor      # (B, MAX) bool
    u_side: torch.Tensor        # (B, MAX) long
    u_card: torch.Tensor        # (B, MAX) long
    u_unit: torch.Tensor        # (B, MAX) long
    u_hp: torch.Tensor          # (B, MAX)
    u_x: torch.Tensor           # (B, MAX)
    u_y: torch.Tensor           # (B, MAX)
    u_deploy: torch.Tensor      # (B, MAX) deploy countdown, seconds
    u_atk: torch.Tensor         # (B, MAX) current cooldown countdown (AttackStateMachine.currentCooldown)
    u_windup: torch.Tensor      # (B, MAX) remaining windup
    u_load: torch.Tensor        # (B, MAX) accumulated load time (capped at loadTime)
    u_attacking: torch.Tensor   # (B, MAX) bool
    u_tgt: torch.Tensor         # (B, MAX) target slot id (-1 = none); >= MAX = tower
    u_locked: torch.Tensor      # (B, MAX) bool

    spawn_pending: torch.Tensor  # (B, 2, 8) long, card idx (-1 empty)
    spawn_timer: torch.Tensor    # (B, 2, 8)
    spawn_pos: torch.Tensor      # (B, 2, 8, 2)

    p_active: torch.Tensor       # (B, MAX_PROJ) bool
    p_x: torch.Tensor            # (B, MAX_PROJ)
    p_y: torch.Tensor            # (B, MAX_PROJ)
    p_dmg: torch.Tensor          # (B, MAX_PROJ)
    p_speed: torch.Tensor        # (B, MAX_PROJ) tiles/s
    p_radius: torch.Tensor       # (B, MAX_PROJ)
    p_target: torch.Tensor       # (B, MAX_PROJ) target slot id
    p_tx: torch.Tensor           # (B, MAX_PROJ) last known target pos
    p_ty: torch.Tensor
    p_life: torch.Tensor         # (B, MAX_PROJ) seconds remaining

    hand: torch.Tensor           # (B, 2, 4) card idx (-1 empty)
    cycle: torch.Tensor          # (B, 2, 8) the 8-card rotation queue
    cycle_pos: torch.Tensor      # (B, 2) long

    @property
    def batch_size(self) -> int:
        return int(self.elixir.shape[0])


class BatchedCRSim:
    def __init__(self, tables: CardTable, batch_size: int, device: str = "cpu", level: int = 11):
        self.t = tables
        self.device = torch.device(device)
        self.b = batch_size
        self.level = level
        self.fac = level_factor(level)
        self._costs = self.t.costs.to(self.device)
        self._hp = (self.t.u_health * self.fac).to(self.device)
        self._dmg = (self.t.u_damage * self.fac).to(self.device)
        self._radius = self.t.u_radius.to(self.device)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, env_mask: torch.Tensor | None = None) -> SimState:
        """Reset all rows, or only the rows selected by ``env_mask`` (B,) bool.

        Geometry tables (tower_pos/tower_king/tower_side) are rebuilt for the
        whole batch (row-independent). The partial form is the auto-reset hook
        used by the SB3 vector env (M5): a finished battle restarts without
        touching its neighbours in the lockstep batch.
        """
        b, dev = self.b, self.device
        # tower geometry tables
        pos = []
        king_flags = []
        for side, key in ((0, "blue"), (1, "red")):
            pos.append(KING_POS[key])
            pos.extend(PRINCESS_POS[key])
            king_flags.append([True, False, False])
        self.tower_pos = torch.tensor(pos, dtype=torch.float32, device=dev).expand(b, N_TOWERS, 2).clone()
        self.tower_king = torch.tensor(sum(king_flags, []), dtype=torch.bool, device=dev)
        self.tower_side = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=dev)

        new = SimState(
            elixir=torch.full((b, 2), ELIXIR_START, device=dev),
            time=torch.zeros(b, device=dev),
            tower_hp=torch.where(self.tower_king, torch.full((N_TOWERS,), T_KING_HP, device=dev),
                                 torch.full((N_TOWERS,), T_PRINCESS_HP, device=dev)).expand(b, N_TOWERS).clone(),
            tower_alive=torch.ones(b, N_TOWERS, dtype=torch.bool, device=dev),
            tower_cd=torch.zeros(b, N_TOWERS, device=dev),
            king_active=torch.zeros(b, 2, dtype=torch.bool, device=dev),
            crowns=torch.zeros(b, 2, dtype=torch.long, device=dev),
            game_over=torch.zeros(b, dtype=torch.bool, device=dev),
            winner=torch.full((b,), -1, dtype=torch.long, device=dev),
            u_active=torch.zeros(b, MAX_UNITS, dtype=torch.bool, device=dev),
            u_side=torch.zeros(b, MAX_UNITS, dtype=torch.long, device=dev),
            u_card=torch.full((b, MAX_UNITS), -1, dtype=torch.long, device=dev),
            u_unit=torch.zeros(b, MAX_UNITS, dtype=torch.long, device=dev),
            u_hp=torch.zeros(b, MAX_UNITS, device=dev),
            u_x=torch.zeros(b, MAX_UNITS, device=dev),
            u_y=torch.zeros(b, MAX_UNITS, device=dev),
            u_deploy=torch.zeros(b, MAX_UNITS, device=dev),
            u_atk=torch.zeros(b, MAX_UNITS, device=dev),
            u_windup=torch.zeros(b, MAX_UNITS, device=dev),
            u_load=torch.zeros(b, MAX_UNITS, device=dev),
            u_attacking=torch.zeros(b, MAX_UNITS, dtype=torch.bool, device=dev),
            u_tgt=torch.full((b, MAX_UNITS), -1, dtype=torch.long, device=dev),
            u_locked=torch.zeros(b, MAX_UNITS, dtype=torch.bool, device=dev),
            spawn_pending=torch.full((b, 2, 8), -1, dtype=torch.long, device=dev),
            spawn_timer=torch.zeros(b, 2, 8, device=dev),
            spawn_pos=torch.zeros(b, 2, 8, 2, device=dev),
            p_active=torch.zeros(b, MAX_PROJ, dtype=torch.bool, device=dev),
            p_x=torch.zeros(b, MAX_PROJ, device=dev),
            p_y=torch.zeros(b, MAX_PROJ, device=dev),
            p_dmg=torch.zeros(b, MAX_PROJ, device=dev),
            p_speed=torch.zeros(b, MAX_PROJ, device=dev),
            p_radius=torch.zeros(b, MAX_PROJ, device=dev),
            p_target=torch.full((b, MAX_PROJ), -1, dtype=torch.long, device=dev),
            p_tx=torch.zeros(b, MAX_PROJ, device=dev),
            p_ty=torch.zeros(b, MAX_PROJ, device=dev),
            p_life=torch.zeros(b, MAX_PROJ, device=dev),
            hand=torch.full((b, 2, 4), -1, dtype=torch.long, device=dev),
            cycle=torch.full((b, 2, 8), -1, dtype=torch.long, device=dev),
            cycle_pos=torch.zeros(b, 2, dtype=torch.long, device=dev),
        )
        if env_mask is None or bool(env_mask.all()):
            self.s = new
        else:
            old = self.s
            m = env_mask.to(dev)
            for f in dataclasses.fields(SimState):
                o, n = getattr(old, f.name), getattr(new, f.name)
                mask = m.view(-1, *([1] * (o.dim() - 1)))
                setattr(new, f.name, torch.where(mask, n, o))
            self.s = new
        return self.s

    # ------------------------------------------------------------ deployment
    def deploy(self, side: int, card_idx: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        for e in torch.nonzero(ok, as_tuple=False).flatten().tolist():
            ci = int(card_idx[e])
            count = int(self.t.spawn_count[ci])
            ui = int(self.t.unit_of_card[ci])
            offs = self.t.formation_offsets[ci]
            hp = float(self._hp[ui])
            for k in range(count):
                ox, oy = offs[k % len(offs)] if offs else (0.0, 0.0)
                xx, yy = float(x[e]) + ox, float(y[e]) + oy
                if k == 0:
                    slot = self._free_slot(e)
                    if slot < 0:
                        break
                    self._place(e, slot, side, ci, ui, hp, xx, yy)
                else:
                    self._queue_spawn(e, side, ci, xx, yy, k)
        return ok

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
        s.u_deploy[e, slot] = float(self.t.u_deploy[ui])
        s.u_atk[e, slot] = 0.0  # AttackStateMachine starts with currentCooldown = 0
        s.u_windup[e, slot] = 0.0
        s.u_load[e, slot] = 0.0
        s.u_attacking[e, slot] = False
        s.u_tgt[e, slot] = -1
        s.u_locked[e, slot] = False

    def _queue_spawn(self, e, side, ci, xx, yy, k):
        s = self.s
        q = (s.spawn_pending[e, side] == -1).nonzero(as_tuple=False).flatten()
        if q.numel() == 0:
            return
        j = int(q[0])
        s.spawn_pending[e, side, j] = ci
        s.spawn_timer[e, side, j] = k * float(self.t.summon_delay[ci]) + TICK_DT
        s.spawn_pos[e, side, j, 0] = xx
        s.spawn_pos[e, side, j, 1] = yy

    # ------------------------------------------------------- hand / card cycle
    def set_deck(self, side: int, card_idxs: list[int],
                 env_mask: torch.Tensor | None = None) -> None:
        """Set the 8-card deck for `side` (hand = first 4, cycle = rest).

        `env_mask` (B,) bool restricts the write to selected rows (used after a
        partial reset so only restarted battles get a fresh hand).
        """
        assert len(card_idxs) == 8
        s = self.s
        rows = range(self.b) if env_mask is None \
            else torch.nonzero(env_mask, as_tuple=False).flatten().tolist()
        for e in rows:
            for k, ci in enumerate(card_idxs):
                if k < 4:
                    s.hand[e, side, k] = ci
                else:
                    s.cycle[e, side, k - 4] = ci
            s.cycle_pos[e, side] = 0

    def _zone_ok(self, side: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Deployment zone rule: own half; enemy-side pocket opens when their princess falls."""
        s = self.s
        if side == 0:
            base = y >= SIDE_HALF_Y
            pocket = (~s.tower_alive[:, 4] & (x < 9.0)) | (~s.tower_alive[:, 5] & (x >= 9.0))
        else:
            base = y <= SIDE_HALF_Y
            pocket = (~s.tower_alive[:, 1] & (x < 9.0)) | (~s.tower_alive[:, 2] & (x >= 9.0))
        return base | pocket

    def play(self, side: int, slot: int, x: torch.Tensor, y: torch.Tensor,
             env_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Play hand-slot `slot` for `side` (spells: anywhere; troops: own half).

        `env_mask` (B,) optionally restricts which envs may act. Returns a (B,)
        success mask. On success the played card rotates to the back.
        """
        s, dev = self.s, self.device
        assert side in (0, 1) and 0 <= slot < 4
        card = s.hand[:, side, slot]
        valid = card >= 0
        if env_mask is not None:
            valid = valid & env_mask
        cclamp = card.clamp(0, len(self.t.names) - 1)
        is_spell = valid & (self.t.card_types[cclamp] == 1)

        # troops: zone check + standard deploy path (handles cost + spawning)
        troop_ok = valid & ~is_spell & self._zone_ok(side, x, y)
        troop_card = torch.where(troop_ok, card, torch.full_like(card, -1))
        ok_troop = self.deploy(side, troop_card, x, y)

        # spells: cast (anywhere on the arena)
        ok_spell = torch.zeros_like(valid)
        cast_mask = valid & is_spell
        if bool(cast_mask.any()):
            ok_spell = self._cast_spell(side, card, x, y, cast_mask)

        ok = ok_troop | ok_spell
        for e in torch.nonzero(ok, as_tuple=False).flatten().tolist():
            # 4-slot draw queue: draw the next card into the played slot, put the
            # played card back at the drawn position (the queue's back). This is
            # an exact FIFO equivalent of the Java Hand model (played -> back of
            # cycle, next -> empty slot, draw new next); `cycle_pos % 4`, NOT
            # % 8 — the queue holds 4 pending cards, and drawing past them would
            # read uninitialised slots (-1).
            pos = int(s.cycle_pos[e, side]) % 4
            played = int(s.hand[e, side, slot])
            s.hand[e, side, slot] = s.cycle[e, side, pos]
            s.cycle[e, side, pos] = played
            s.cycle_pos[e, side] = (s.cycle_pos[e, side] + 1) % 4
        return ok

    def _cast_spell(self, side: int, card: torch.Tensor, x: torch.Tensor,
                    y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Instant area spell: damage enemy units + towers in radius (crown-tower % applied)."""
        s, dev = self.s, self.device
        cclamp = card.clamp(0, len(self.t.names) - 1)
        cost = self._costs[cclamp]
        fire = mask & (s.elixir[:, side] >= cost - 1e-6)
        if not bool(fire.any()):
            return fire
        s.elixir[:, side] = torch.where(fire, s.elixir[:, side] - cost, s.elixir[:, side])

        dmg = (self.t.spell_damage[cclamp] * self.fac)          # level-scaled
        rad = self.t.spell_radius[cclamp]
        hits_air = self.t.spell_hits_air[cclamp] > 0.5
        hits_ground = self.t.spell_hits_ground[cclamp] > 0.5

        # enemy units (incl. buildings): in-radius by center distance + radii
        enemy = s.u_active & (s.u_side == 1 - side)
        dx = s.u_x - x.unsqueeze(1)
        dy = s.u_y - y.unsqueeze(1)
        d2 = dx * dx + dy * dy
        eff = (rad.unsqueeze(1) + self._radius[s.u_unit]) ** 2
        mt = self.t.u_move_type[s.u_unit]
        type_ok = torch.where(mt == 1, hits_air.unsqueeze(1), hits_ground.unsqueeze(1))
        hit_u = enemy & fire.unsqueeze(1) & type_ok & (d2 <= eff)
        if bool(hit_u.any()):
            # Java Health.takeDamage: current -= min(damage, current) -> HP floors at 0
            s.u_hp = (s.u_hp - torch.where(hit_u, dmg.unsqueeze(1), torch.zeros_like(s.u_hp))).clamp(min=0.0)

        # enemy towers: crown-tower damage percent applies
        crown_mult = 1.0 + self.t.spell_crown_pct[cclamp] / 100.0
        dmg_t = dmg * crown_mult
        t_rad = torch.where(self.tower_king, KING_RADIUS, T_RADIUS)
        dx_t = self.tower_pos[:, :, 0] - x.unsqueeze(1)
        dy_t = self.tower_pos[:, :, 1] - y.unsqueeze(1)
        d2_t = dx_t * dx_t + dy_t * dy_t
        eff_t = (rad.unsqueeze(1) + t_rad.unsqueeze(0)) ** 2
        enemy_t = self.tower_side.unsqueeze(0) == (1 - side)
        hit_t = s.tower_alive & enemy_t & fire.unsqueeze(1) & (d2_t <= eff_t)
        if bool(hit_t.any()):
            s.tower_hp = (s.tower_hp - torch.where(hit_t, dmg_t.unsqueeze(1), torch.zeros_like(s.tower_hp))).clamp(min=0.0)
        return fire

    # ------------------------------------------------------------------ tick
    def tick(self, n: int = 1, dt: float = TICK_DT) -> None:
        for _ in range(n):
            self._tick_once(dt)

    def _tick_once(self, dt: float) -> None:
        s, dev = self.s, self.device
        s.time = s.time + dt

        # elixir regen with double/triple phases (GameEngine.checkTimeLimit gates)
        rate = 1.0 + (s.time >= DOUBLE_ELIXIR_T).float() + (s.time >= TRIPLE_ELIXIR_T).float()
        s.elixir = torch.clamp(s.elixir + (dt / ELIXIR_PERIOD) * rate.unsqueeze(-1), max=ELIXIR_MAX)

        # staggered spawns
        s.spawn_timer = torch.clamp(s.spawn_timer - dt, min=0)
        fired = (s.spawn_timer <= 0) & (s.spawn_pending >= 0)
        if bool(fired.any()):
            for e, side, j in torch.nonzero(fired, as_tuple=False).tolist():
                ci = int(s.spawn_pending[e, side, j])
                ui = int(self.t.unit_of_card[ci])
                slot = self._free_slot(e)
                if slot >= 0:
                    self._place(e, slot, side, ci, ui, float(self._hp[ui]),
                                float(s.spawn_pos[e, side, j, 0]), float(s.spawn_pos[e, side, j, 1]))
                s.spawn_pending[e, side, j] = -1

        # timers
        s.u_deploy = torch.clamp(s.u_deploy - dt, min=0)
        s.tower_cd = torch.clamp(s.tower_cd - dt, min=0)

        # king activation: damage taken OR a friendly princess destroyed (Tower activation rule)
        blue_princess_dead = (~s.tower_alive[:, 1]) | (~s.tower_alive[:, 2])
        red_princess_dead = (~s.tower_alive[:, 4]) | (~s.tower_alive[:, 5])
        s.king_active = s.king_active | torch.stack([blue_princess_dead, red_princess_dead], dim=1) \
            | (s.tower_hp[:, [0, 3]] < (T_KING_HP - 1e-6))

        self._tower_attacks(dt)
        self._unit_combat(dt)
        self._projectiles(dt)
        self._movement(dt)
        self._resolve_deaths()
        self._time_limit()

    # --------------------------------------------------------------- combat
    def _target_pads(self):
        """Build padded target tensors: units [0,MAX) then towers [MAX,MAX+6)."""
        s = self.s
        b = s.elixir.shape[0]
        t_alive = torch.cat([s.u_active, s.tower_alive], dim=1)         # (B, MAX+6)
        t_side = torch.cat([s.u_side, self.tower_side.expand(b, N_TOWERS)], dim=1)
        t_x = torch.cat([s.u_x, self.tower_pos[..., 0]], dim=1)
        t_y = torch.cat([s.u_y, self.tower_pos[..., 1]], dim=1)
        t_rad = torch.cat([self._radius[s.u_unit], torch.where(self.tower_king, KING_RADIUS, T_RADIUS).expand(b, N_TOWERS)], dim=1)
        # move type: units from table, towers = BUILDING (2)
        t_move = torch.cat([self.t.u_move_type[s.u_unit], torch.full((b, N_TOWERS), 2, dtype=torch.long, device=self.device)], dim=1)
        return t_alive, t_side, t_x, t_y, t_rad, t_move

    def _unit_combat(self, dt: float) -> None:
        s, dev = self.s, self.device
        T = MAX_UNITS + N_TOWERS
        b = s.elixir.shape[0]
        t_alive, t_side, t_x, t_y, t_rad, t_move = self._target_pads()

        # attacker-side tensors
        a_active = s.u_active & (s.u_deploy <= 0)
        a_side = s.u_side
        a_rng = self.t.u_range[s.u_unit]
        a_sight = self.t.u_sight[s.u_unit]
        a_rad = self._radius[s.u_unit]
        a_tt = self.t.u_target_type[s.u_unit]
        a_only_b = self.t.u_only_buildings[s.u_unit]

        # squared distances attacker->target (B, MAX, T)
        dx = s.u_x.unsqueeze(2) - t_x.unsqueeze(1)
        dy = s.u_y.unsqueeze(2) - t_y.unsqueeze(1)
        d2 = dx * dx + dy * dy

        # candidate mask
        enemy = t_side.unsqueeze(1) != a_side.unsqueeze(2)
        # target type compatibility (TargetingSystem.canTarget):
        # targetOnlyBuildings overrides everything; otherwise by movement dimension.
        only_b = (a_only_b > 0).unsqueeze(2)              # (B,MAX,1)
        is_building = (t_move == 2).unsqueeze(1)          # (B,1,T)
        tgt_ground = (t_move == 0).unsqueeze(1)           # (B,1,T)
        tgt_air = (t_move == 1).unsqueeze(1)              # (B,1,T)
        tt = a_tt.unsqueeze(2)                            # (B,MAX,1)
        base_ok = (tt == 0) | ((tt == 1) & (tgt_ground | is_building)) | ((tt == 2) & tgt_air)
        type_ok = torch.where(only_b, is_building.expand_as(base_ok), base_ok)
        sight_sq = (a_sight.unsqueeze(2) + a_rad.unsqueeze(2) + t_rad.unsqueeze(1)) ** 2
        cand = enemy & t_alive.unsqueeze(1) & type_ok & (d2 <= sight_sq) & a_active.unsqueeze(2)

        # nearest selection
        BIG = torch.full_like(d2, float("inf"))
        d2_masked = torch.where(cand, d2, BIG)
        nearest = d2_masked.argmin(dim=2)
        has_any = cand.any(dim=2)

        # lock retention: keep current target while still valid within sight*1.5
        # (Java TargetingSystem.isValidTarget: type/team/alive + retention radius)
        cur = s.u_tgt.clone()
        cur_valid = (cur >= 0) & (cur < T)
        cur_idx = cur.clamp(0, T - 1)
        cur_enemy = torch.gather(enemy, 2, cur_idx.unsqueeze(2)).squeeze(2)
        cur_alive = torch.gather(t_alive.unsqueeze(1).expand(b, MAX_UNITS, T), 2, cur_idx.unsqueeze(2)).squeeze(2)
        cur_typeok = torch.gather(type_ok, 2, cur_idx.unsqueeze(2)).squeeze(2)
        cur_d2_raw = torch.gather(d2, 2, cur_idx.unsqueeze(2)).squeeze(2)
        ret = ((a_sight * TARGET_RETENTION).unsqueeze(2) + a_rad.unsqueeze(2) + t_rad.unsqueeze(1)) ** 2
        cur_ret_sq = torch.gather(ret, 2, cur_idx.unsqueeze(2)).squeeze(2)
        keep = cur_valid & cur_enemy & cur_alive & cur_typeok & (cur_d2_raw <= cur_ret_sq)
        tbo = a_only_b > 0  # targetOnlyBuildings: always retarget nearest building
        new_tgt = torch.where(keep & ~tbo, cur, torch.where(has_any, nearest, torch.full_like(nearest, -1)))
        s.u_tgt = new_tgt

        # attack range (edge-to-edge inclusive)
        rng_sq = (a_rng.unsqueeze(2) + a_rad.unsqueeze(2) + t_rad.unsqueeze(1)) ** 2
        sel = new_tgt.clamp(0, T - 1)
        sel_rng_sq = torch.gather(rng_sq, 2, sel.unsqueeze(2)).squeeze(2)
        in_range = (new_tgt >= 0) & (d2_masked.gather(2, sel.unsqueeze(2)).squeeze(2) <= sel_rng_sq)
        s.u_locked = in_range

        # AttackStateMachine timers: cooldown--, windup-- while attacking, load++ otherwise (capped)
        s.u_atk = torch.clamp(s.u_atk - dt, min=0)
        s.u_windup = torch.where(s.u_attacking, s.u_windup - dt, s.u_windup)
        s.u_load = torch.where(s.u_attacking, s.u_load,
                               torch.minimum(s.u_load + dt, self.t.u_loadtime[s.u_unit]))
        # cancel an in-progress attack when the target leaves range
        cancel = s.u_attacking & ~in_range
        s.u_attacking = s.u_attacking & ~cancel
        s.u_windup = torch.where(cancel, torch.zeros_like(s.u_windup), s.u_windup)
        # start a new attack sequence: windup = max(0, cooldown - accumulated load); load consumed
        start = in_range & a_active & ~s.u_attacking & (s.u_atk <= 0)
        s.u_windup = torch.where(start, torch.clamp(self.t.u_cooldown[s.u_unit] - s.u_load, min=0), s.u_windup)
        s.u_load = torch.where(start, torch.zeros_like(s.u_load), s.u_load)
        s.u_attacking = s.u_attacking | start

        # execute attack: melee (and projectile-less ranged) hit instantly;
        # ranged units with projectile data fire a flying projectile (M2).
        # epsilon: float32 windup accumulation can leave ~1e-7 residue; without it the
        # attack slips one tick (25-tick cycles instead of 24 — DIVERGENCES #17).
        fire = s.u_attacking & (s.u_windup <= 1e-5) & in_range & a_active
        if bool(fire.any()):
            dmg = self._dmg[s.u_unit]
            proj_speed = self.t.u_proj_speed[s.u_unit]
            is_ranged = a_rng >= RANGED_THRESHOLD
            via_proj = fire & is_ranged & (proj_speed > 0)
            direct = fire & ~via_proj
            # direct damage: unit targets
            unit_tgt = direct & (sel < MAX_UNITS)
            if bool(unit_tgt.any()):
                b_idx, s_idx = torch.nonzero(unit_tgt, as_tuple=True)
                tgt_slot = sel[b_idx, s_idx]
                s.u_hp[b_idx, tgt_slot] = (s.u_hp[b_idx, tgt_slot] - dmg[b_idx, s_idx]).clamp(min=0.0)
            # direct damage: tower targets
            tw_tgt = direct & (sel >= MAX_UNITS)
            if bool(tw_tgt.any()):
                b_idx, s_idx = torch.nonzero(tw_tgt, as_tuple=True)
                t_idx = sel[b_idx, s_idx] - MAX_UNITS
                s.tower_hp[b_idx, t_idx] = (s.tower_hp[b_idx, t_idx] - dmg[b_idx, s_idx]).clamp(min=0.0)
            # projectiles
            if bool(via_proj.any()):
                for e, i in torch.nonzero(via_proj, as_tuple=False).tolist():
                    self._spawn_projectile(e, i, int(sel[e, i]), float(dmg[e, i]), float(proj_speed[e, i]))
            # finishAttack: currentCooldown = 0 (immediate chaining via windup)
            s.u_attacking = s.u_attacking & ~fire
            s.u_atk = torch.where(fire, torch.zeros_like(s.u_atk), s.u_atk)

    def _spawn_projectile(self, e: int, i: int, target_slot: int, dmg: float, speed: float) -> None:
        s = self.s
        free = (~s.p_active[e]).nonzero(as_tuple=False).flatten()
        if free.numel() == 0:
            return  # overflow: drop (DIVERGENCES #10 policy)
        j = int(free[0])
        s.p_active[e, j] = True
        s.p_x[e, j] = s.u_x[e, i]
        s.p_y[e, j] = s.u_y[e, i]
        s.p_dmg[e, j] = dmg
        s.p_speed[e, j] = speed
        s.p_radius[e, j] = float(self.t.u_proj_radius[s.u_unit[e, i]])
        s.p_target[e, j] = target_slot
        s.p_tx[e, j] = s.p_x[e, j]
        s.p_ty[e, j] = s.p_y[e, j]
        s.p_life[e, j] = 3.0

    def _projectiles(self, dt: float) -> None:
        s, dev = self.s, self.device
        if not bool(s.p_active.any()):
            return
        T = MAX_UNITS + N_TOWERS
        t_alive, t_side, t_x, t_y, t_rad, t_move = self._target_pads()
        idx = s.p_target.clamp(0, T - 1)
        tx = torch.gather(t_x, 1, idx)
        ty = torch.gather(t_y, 1, idx)
        talive = torch.gather(t_alive, 1, idx)
        trad = torch.gather(t_rad, 1, idx)
        # homing: track living targets; frozen targets keep the last known position
        s.p_tx = torch.where(talive & s.p_active, tx, s.p_tx)
        s.p_ty = torch.where(talive & s.p_active, ty, s.p_ty)
        dx = s.p_tx - s.p_x
        dy = s.p_ty - s.p_y
        dist = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
        step = s.p_speed * dt
        s.p_x = s.p_x + torch.where(s.p_active, dx / dist * step, torch.zeros_like(step))
        s.p_y = s.p_y + torch.where(s.p_active, dy / dist * step, torch.zeros_like(step))
        # impact: within combined radius of the (living) target
        dx2 = s.p_tx - s.p_x
        dy2 = s.p_ty - s.p_y
        dist2 = torch.sqrt(dx2 * dx2 + dy2 * dy2)
        hit = s.p_active & talive & (dist2 <= (s.p_radius + trad))
        if bool(hit.any()):
            unit_hits = hit & (idx < MAX_UNITS)
            if bool(unit_hits.any()):
                b_idx, j_idx = torch.nonzero(unit_hits, as_tuple=True)
                tgt_slot = idx[b_idx, j_idx]
                s.u_hp[b_idx, tgt_slot] = (s.u_hp[b_idx, tgt_slot] - s.p_dmg[b_idx, j_idx]).clamp(min=0.0)
            tower_hits = hit & (idx >= MAX_UNITS)
            if bool(tower_hits.any()):
                b_idx, j_idx = torch.nonzero(tower_hits, as_tuple=True)
                t_idx = idx[b_idx, j_idx] - MAX_UNITS
                s.tower_hp[b_idx, t_idx] = (s.tower_hp[b_idx, t_idx] - s.p_dmg[b_idx, j_idx]).clamp(min=0.0)
        # lifetime expiry
        s.p_life = s.p_life - dt
        s.p_active = s.p_active & ~hit & (s.p_life > 0)

    def _tower_attacks(self, dt: float) -> None:
        s, dev = self.s, self.device
        for t_idx in range(N_TOWERS):
            is_king = bool(self.tower_king[t_idx])
            alive = s.tower_alive[:, t_idx] & (s.tower_cd[:, t_idx] <= 0)
            if is_king:
                alive = alive & s.king_active[:, int(self.tower_side[t_idx])]
            if not bool(alive.any()):
                continue
            side = int(self.tower_side[t_idx])
            rng = T_KING_RANGE if is_king else T_PRINCESS_RANGE
            rad = KING_RADIUS if is_king else T_RADIUS
            dmg = T_KING_DMG if is_king else T_PRINCESS_DMG
            cd = T_KING_CD if is_king else T_PRINCESS_CD
            enemy = s.u_active & (s.u_side == 1 - side)
            dx = s.u_x - self.tower_pos[:, t_idx, 0].unsqueeze(1)
            dy = s.u_y - self.tower_pos[:, t_idx, 1].unsqueeze(1)
            d2 = dx * dx + dy * dy
            eff_sq = (rng + rad + self._radius[s.u_unit]) ** 2
            cand = enemy & (d2 <= eff_sq)
            d2m = torch.where(cand, d2, torch.full_like(d2, float("inf")))
            nearest = d2m.argmin(dim=1)
            has = cand.any(dim=1)
            firem = alive & has
            if bool(firem.any()):
                bidx = torch.nonzero(firem, as_tuple=False).flatten()
                tgt = nearest[bidx]
                s.u_hp[bidx, tgt] = (s.u_hp[bidx, tgt] - dmg).clamp(min=0.0)
                s.tower_cd[bidx, t_idx] = cd

    def _movement(self, dt: float) -> None:
        s, dev = self.s, self.device
        T = MAX_UNITS + N_TOWERS
        t_alive, t_side, t_x, t_y, t_rad, t_move = self._target_pads()
        go_move = s.u_active & (s.u_deploy <= 0) & ~s.u_locked
        # buildings don't move
        go_move = go_move & (self.t.u_move_type[s.u_unit] != 2)
        has_target = s.u_tgt >= 0
        moving = go_move & has_target
        targetless = go_move & ~has_target
        if not bool(moving.any()) and not bool(targetless.any()):
            return
        # goal: current target; targetless units advance toward the enemy king
        sel = s.u_tgt.clamp(0, T - 1)
        gx = torch.gather(t_x, 1, sel)
        gy = torch.gather(t_y, 1, sel)
        king_x = torch.full_like(s.u_x, KING_POS["red"][0])
        king_y = torch.full_like(s.u_y, KING_POS["red"][1])
        rk_x = torch.full_like(s.u_x, KING_POS["blue"][0])
        rk_y = torch.full_like(s.u_y, KING_POS["blue"][1])
        adv_x = torch.where(s.u_side == 0, king_x, rk_x)
        adv_y = torch.where(s.u_side == 0, king_y, rk_y)
        gx = torch.where(targetless, adv_x, gx)
        gy = torch.where(targetless, adv_y, gy)
        # BasePathfinder port: ground units route via bridges; air flies straight
        cur_x, cur_y = s.u_x, s.u_y
        air = self.t.u_move_type[s.u_unit] == 1
        north = cur_y > RIVER_Y_MAX_T
        south = cur_y < RIVER_Y_MIN_T
        in_river = ~north & ~south
        cross_n = south & (gy > RIVER_Y_MAX_T)
        cross_s = north & (gy < RIVER_Y_MIN_T)
        bridge_x = torch.where((cur_x - BRIDGE_LX).abs() < (cur_x - BRIDGE_RX).abs(),
                               torch.full_like(cur_x, BRIDGE_LX), torch.full_like(cur_x, BRIDGE_RX))
        bx = torch.where((cur_x - bridge_x).abs() < BRIDGE_ALIGN_T, cur_x, bridge_x)
        approach_y = torch.where(cross_n, torch.full_like(cur_y, RIVER_Y_MIN_T),
                                 torch.full_like(cur_y, RIVER_Y_MAX_T))
        pre = (cross_n & (cur_y < RIVER_Y_MIN_T - APPROACH_TOL_T)) | (cross_s & (cur_y > RIVER_Y_MAX_T + APPROACH_TOL_T))
        wp_y = torch.where(pre, approach_y, torch.full_like(cur_y, RIVER_CENTER_T))
        exit_y = torch.where(gy > cur_y, torch.full_like(cur_y, RIVER_Y_MAX_T + BOUNDARY_EPS_T),
                             torch.full_like(cur_y, RIVER_Y_MIN_T - BOUNDARY_EPS_T))
        wp_y = torch.where(in_river, exit_y, wp_y)
        use_wp = ~air & (cross_n | cross_s | in_river)
        dir_x = torch.where(use_wp, bx - cur_x, gx - cur_x)
        dir_y = torch.where(use_wp, wp_y - cur_y, gy - cur_y)
        dist = torch.sqrt(dir_x * dir_x + dir_y * dir_y).clamp(min=1e-6)
        speed = self.t.u_speed[s.u_unit] / 60.0  # GameUnits.rawSpeedToUnitsPerSecond
        step = speed * dt
        ux = s.u_x + dir_x / dist * step
        uy = s.u_y + dir_y / dist * step
        go = moving | targetless
        s.u_x = torch.where(go, ux, s.u_x)
        s.u_y = torch.where(go, uy, s.u_y)

    def _resolve_deaths(self) -> None:
        s = self.s
        # unit deaths
        dead = s.u_active & (s.u_hp <= 0)
        if bool(dead.any()):
            s.u_active = s.u_active & ~dead
        # tower deaths
        fallen = s.tower_alive & (s.tower_hp <= 0)
        if bool(fallen.any()):
            b_idx, t_idx = torch.nonzero(fallen, as_tuple=True)
            for bb, tt in zip(b_idx.tolist(), t_idx.tolist()):
                side = int(self.tower_side[tt])
                if bool(self.tower_king[tt]):
                    s.crowns[bb, 1 - side] += 3
                    s.game_over[bb] = True
                    s.winner[bb] = 1 - side
                else:
                    s.crowns[bb, 1 - side] += 1
            s.tower_alive = s.tower_alive & ~fallen

    def _time_limit(self) -> None:
        s = self.s
        # regular time end: crowns decide, else overtime
        at_end = (s.time >= MATCH_END_T) & (s.time - 0.0 < MATCH_END_T + 1) & ~s.game_over
        if bool(at_end.any()):
            blue, red = s.crowns[:, 0], s.crowns[:, 1]
            decided = at_end & (blue != red)
            if bool(decided.any()):
                s.game_over = s.game_over | decided
                w = torch.where(blue > red, torch.zeros_like(blue), torch.ones_like(blue))
                s.winner = torch.where(decided, w, s.winner)
        # overtime end: crowns, then total tower hp, then draw
        at_final = (s.time >= MATCH_FINAL_T) & (s.time - 0.0 < MATCH_FINAL_T + 1) & ~s.game_over
        if bool(at_final.any()):
            blue, red = s.crowns[:, 0], s.crowns[:, 1]
            hp_b = torch.where(s.tower_alive, s.tower_hp, torch.zeros_like(s.tower_hp))[:, :3].sum(dim=1)
            hp_r = torch.where(s.tower_alive, s.tower_hp, torch.zeros_like(s.tower_hp))[:, 3:].sum(dim=1)
            w = torch.full_like(blue, 2)
            w = torch.where(blue > red, torch.zeros_like(w), w)
            w = torch.where(red > blue, torch.ones_like(w), w)
            tie = blue == red
            w = torch.where(tie & (hp_b > hp_r), torch.zeros_like(w), w)
            w = torch.where(tie & (hp_r > hp_b), torch.ones_like(w), w)
            s.winner = torch.where(at_final, w, s.winner)
            s.game_over = s.game_over | at_final


def make_sim(data_dir: str, batch_size: int, device: str = "cpu", level: int = 11) -> BatchedCRSim:
    from .cards import load_tables

    return BatchedCRSim(load_tables(data_dir, device=device), batch_size, device=device, level=level)
