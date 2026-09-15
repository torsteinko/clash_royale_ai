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
import math
from dataclasses import dataclass

import torch

from .cards import CardTable, level_multiplier_java, scaled_stat_java

MAX_UNITS = 64           # per env (overflow tracked in DIVERGENCES.md #10)
MAX_PROJ = 128           # per env
MAX_PENDING = 16         # per side: sync-delayed troop/building spawn queue
MAX_CASTS = 4            # per side: sync-delayed spell casts
N_TOWERS = 6             # slot order: [kingB, pB_l, pB_r, kingR, pR_l, pR_r]
TICK_DT = 0.05           # 20 ticks/s (GameEngine.TICKS_PER_SECOND = 20)
ELIXIR_START = 5.0
ELIXIR_MAX = 10.0
ELIXIR_PERIOD = 2.8      # seconds per elixir at rate 1
DOUBLE_ELIXIR_T = 120.0  # activated at match_end - 60s
TRIPLE_ELIXIR_T = 240.0  # overtime + 60s
MATCH_END_T = 180.0
MATCH_FINAL_T = 300.0    # overtime end (regular 180s + OT 120s)

# Java DeploymentSystem server sync delay: troops/buildings spawn, and spells
# cast, 1.0 s after the request (elixir spent + hand cycled immediately).
SYNC_DELAY_T = 1.0
# Troops/buildings become visible one tick later (GameState.processPending at
# the start of the next tick) — a play at record k therefore activates the unit
# during tick k+21 (SYNC + 1 tick). Spells spawn a projectile directly and act
# at tick k+20 (same-tick flight start).
SYNC_TROOP_T = SYNC_DELAY_T + TICK_DT
SYNC_SPELL_T = SYNC_DELAY_T
# Position.MAX_ROUNDING_DISTANCE (sqrt(0.5) game units) — arrival slack
PROJ_HIT_SLACK_T = 0.0007071
# Projectile.DEFAULT_SPEED = 15000 game units/s => 15 tiles/s
DEFAULT_PROJ_SPEED_T = 15.0

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
T_PRINCESS_SIGHT = 9.5    # Tower.PRINCESS_SIGHT_RANGE
T_KING_SIGHT = 7.0        # Tower.CROWN_RANGE (used as sight as well)

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
    u_x: torch.Tensor           # (B, MAX) read position, whole game units (tiles)
    u_y: torch.Tensor
    u_fx: torch.Tensor          # (B, MAX) fixed-point accumulators, float64 sub-units (unit * 65536)
    u_fy: torch.Tensor           # (B, MAX)
    u_deploy: torch.Tensor      # (B, MAX) deploy countdown, seconds
    u_atk: torch.Tensor         # (B, MAX) current cooldown countdown (AttackStateMachine.currentCooldown)
    u_windup: torch.Tensor      # (B, MAX) remaining windup
    u_load: torch.Tensor        # (B, MAX) accumulated load time (capped at loadTime)
    u_attacking: torch.Tensor   # (B, MAX) bool
    u_tgt: torch.Tensor         # (B, MAX) target slot id (-1 = none); >= MAX = tower
    u_locked: torch.Tensor      # (B, MAX) bool

    spawn_pending: torch.Tensor  # (B, 2, MAX_PENDING) long, card idx (-1 empty)
    spawn_timer: torch.Tensor    # (B, 2, MAX_PENDING)
    spawn_pos: torch.Tensor      # (B, 2, MAX_PENDING, 2)

    cast_pending: torch.Tensor   # (B, 2, MAX_CASTS) long, card idx (-1 empty)
    cast_timer: torch.Tensor     # (B, 2, MAX_CASTS)
    cast_pos: torch.Tensor       # (B, 2, MAX_CASTS, 2)

    # tower combat (AttackStateMachine parity: windup machinery + shot projectiles)
    t_windup: torch.Tensor       # (B, N_TOWERS) remaining windup
    t_attacking: torch.Tensor    # (B, N_TOWERS) bool
    t_tgt: torch.Tensor          # (B, N_TOWERS) long, unit slot (-1 = none)
    t_wake: torch.Tensor         # (B, 2) king wake-up timer after activation

    p_active: torch.Tensor       # (B, MAX_PROJ) bool
    p_is_spell: torch.Tensor     # (B, MAX_PROJ) bool — position-targeted spell
    p_side: torch.Tensor         # (B, MAX_PROJ) long — caster/enemy side of the shot
    p_x: torch.Tensor            # (B, MAX_PROJ)
    p_y: torch.Tensor            # (B, MAX_PROJ)
    p_dmg: torch.Tensor          # (B, MAX_PROJ)
    p_speed: torch.Tensor        # (B, MAX_PROJ) tiles/s
    p_radius: torch.Tensor       # (B, MAX_PROJ) AOE radius (spells)
    p_crown: torch.Tensor        # (B, MAX_PROJ) crown-tower damage percent
    p_target: torch.Tensor       # (B, MAX_PROJ) target slot id
    p_tx: torch.Tensor           # (B, MAX_PROJ) last known target pos / destination x
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
        self._m = level_multiplier_java(level)
        # Java-parity float factor (m/100) for the few call sites that scale raw
        # stats directly; scaled values themselves use floor(base * m / 100).
        self.fac = self._m / 100.0
        self._costs = self.t.costs.to(self.device)
        self._hp = scaled_stat_java(self.t.u_health, level).to(self.device)
        self._dmg = scaled_stat_java(self.t.u_damage, level).to(self.device)
        self._radius = self.t.u_radius.to(self.device)
        self.reset()

    def _scaled(self, base: torch.Tensor) -> torch.Tensor:
        """Java `LevelScaling.scaleCard` on a tensor of level-1 base values."""
        return torch.floor(base.float() * self._m / 100.0)

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
            u_fx=torch.zeros(b, MAX_UNITS, dtype=torch.float64, device=dev),
            u_fy=torch.zeros(b, MAX_UNITS, dtype=torch.float64, device=dev),
            u_deploy=torch.zeros(b, MAX_UNITS, device=dev),
            u_atk=torch.zeros(b, MAX_UNITS, device=dev),
            u_windup=torch.zeros(b, MAX_UNITS, device=dev),
            u_load=torch.zeros(b, MAX_UNITS, device=dev),
            u_attacking=torch.zeros(b, MAX_UNITS, dtype=torch.bool, device=dev),
            u_tgt=torch.full((b, MAX_UNITS), -1, dtype=torch.long, device=dev),
            u_locked=torch.zeros(b, MAX_UNITS, dtype=torch.bool, device=dev),
            spawn_pending=torch.full((b, 2, MAX_PENDING), -1, dtype=torch.long, device=dev),
            spawn_timer=torch.zeros(b, 2, MAX_PENDING, device=dev),
            spawn_pos=torch.zeros(b, 2, MAX_PENDING, 2, device=dev),
            cast_pending=torch.full((b, 2, MAX_CASTS), -1, dtype=torch.long, device=dev),
            cast_timer=torch.zeros(b, 2, MAX_CASTS, device=dev),
            cast_pos=torch.zeros(b, 2, MAX_CASTS, 2, device=dev),
            t_windup=torch.zeros(b, N_TOWERS, device=dev),
            t_attacking=torch.zeros(b, N_TOWERS, dtype=torch.bool, device=dev),
            t_tgt=torch.full((b, N_TOWERS), -1, dtype=torch.long, device=dev),
            t_wake=torch.zeros(b, 2, device=dev),
            p_active=torch.zeros(b, MAX_PROJ, dtype=torch.bool, device=dev),
            p_is_spell=torch.zeros(b, MAX_PROJ, dtype=torch.bool, device=dev),
            p_side=torch.zeros(b, MAX_PROJ, dtype=torch.long, device=dev),
            p_x=torch.zeros(b, MAX_PROJ, device=dev),
            p_y=torch.zeros(b, MAX_PROJ, device=dev),
            p_dmg=torch.zeros(b, MAX_PROJ, device=dev),
            p_speed=torch.zeros(b, MAX_PROJ, device=dev),
            p_radius=torch.zeros(b, MAX_PROJ, device=dev),
            p_crown=torch.zeros(b, MAX_PROJ, device=dev),
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
    def _stagger_ticks(self, ci: int) -> int:
        """Java stagger: unit k of a card spawns k * ceil(summonDeployDelay / dt)
        ticks after the synchronised first unit."""
        delay = float(self.t.summon_delay[ci])
        if delay <= 0.0:
            return 0
        return int(math.ceil(delay / TICK_DT - 1e-9))

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
            offs = self.t.formation_offsets[ci]
            interval = self._stagger_ticks(ci) * TICK_DT
            for k in range(count):
                ox, oy = offs[k % len(offs)] if offs else (0.0, 0.0)
                xx, yy = float(x[e]) + ox, float(y[e]) + oy
                # Java sync delay: the first unit exists SYNC_TROOP_T after the
                # request (1.0 s countdown + one tick of pending visibility);
                # staggered units follow at k * stagger interval.
                self._queue_spawn(e, side, ci, xx, yy, SYNC_TROOP_T + k * interval)
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
        # positions live on the whole-game-unit lattice (Java Position: set(int, int))
        xu = float(math.floor(xx * 1000.0 + 0.5))
        yu = float(math.floor(yy * 1000.0 + 0.5))
        s.u_x[e, slot] = xu / 1000.0
        s.u_y[e, slot] = yu / 1000.0
        s.u_fx[e, slot] = xu * 65536.0
        s.u_fy[e, slot] = yu * 65536.0
        s.u_deploy[e, slot] = float(self.t.u_deploy[ui])
        s.u_atk[e, slot] = 0.0  # AttackStateMachine starts with currentCooldown = 0
        s.u_windup[e, slot] = 0.0
        # Troops enter preloaded (Java: initialLoad = loadTime per the RoyaleAPI
        # secret-stats model, DIVERGENCES #18)
        s.u_load[e, slot] = float(self.t.u_loadtime[ui])
        s.u_attacking[e, slot] = False
        s.u_tgt[e, slot] = -1
        s.u_locked[e, slot] = False

    def _queue_spawn(self, e, side, ci, xx, yy, timer):
        s = self.s
        q = (s.spawn_pending[e, side] == -1).nonzero(as_tuple=False).flatten()
        if q.numel() == 0:
            return  # queue full: drop (overflow policy, DIVERGENCES #10)
        j = int(q[0])
        s.spawn_pending[e, side, j] = ci
        s.spawn_timer[e, side, j] = float(timer)
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

        # spells: queue the cast (elixir spent now; Java applies it after the sync delay)
        ok_spell = torch.zeros_like(valid)
        cast_mask = valid & is_spell
        if bool(cast_mask.any()):
            ok_spell = self._queue_cast(side, card, x, y, cast_mask)

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

    # ------------------------------------------------------------------ spells
    def _scaled_scalar(self, base: float) -> float:
        """Java `LevelScaling.scaleCard` for a single base value."""
        return float(math.floor(base * self._m / 100.0))

    def _queue_cast(self, side: int, card: torch.Tensor, x: torch.Tensor,
                    y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Queue a spell cast (Java DeploymentSystem sync delay).

        Elixir is spent immediately. The cast fires 1.0 s later: flying spells
        (projectile data with speed > 0) launch a position-targeted projectile
        from the caster's crown tower and start flying the same tick; direct
        area spells (Zap, Freeze, ...) apply on the next tick (Java spawns an
        AreaEffect entity, which becomes visible through processPending).
        """
        s = self.s
        cclamp = card.clamp(0, len(self.t.names) - 1)
        cost = self._costs[cclamp]
        fire = mask & (s.elixir[:, side] >= cost - 1e-6)
        if not bool(fire.any()):
            return fire
        s.elixir[:, side] = torch.where(fire, s.elixir[:, side] - cost, s.elixir[:, side])
        for e in torch.nonzero(fire, as_tuple=False).flatten().tolist():
            ci = int(card[e])
            free = (s.cast_pending[e, side] == -1).nonzero(as_tuple=False).flatten()
            if free.numel() == 0:
                continue  # cast queue full: drop (overflow policy)
            j = int(free[0])
            flying = float(self.t.spell_speed[ci]) > 0.0
            s.cast_pending[e, side, j] = ci
            s.cast_timer[e, side, j] = SYNC_SPELL_T if flying else SYNC_TROOP_T
            s.cast_pos[e, side, j, 0] = float(x[e])
            s.cast_pos[e, side, j, 1] = float(y[e])
        return fire

    def _cast_spell_now(self, e: int, side: int, ci: int, x: float, y: float) -> None:
        """Fire a queued spell (Java SpellFactory.castSpell)."""
        speed = float(self.t.spell_speed[ci])
        dmg = self._scaled_scalar(float(self.t.spell_damage[ci]))
        rad = float(self.t.spell_radius[ci])
        crown = float(self.t.spell_crown_pct[ci])
        if speed > 0.0:
            self._spawn_spell_projectile(e, side, x, y, dmg, rad, crown, speed)
        else:
            self._apply_spell_aoe(e, side, x, y, dmg, rad, crown)

    def _apply_spell_aoe(self, e: int, side: int, x: float, y: float,
                         dmg: float, rad: float, crown: float) -> None:
        """Java AoeDamageService.applySpellDamage: enemy units within
        radius + collision radius; towers within radius + tower radius with
        crown-tower damage percent (integer arithmetic)."""
        s, dev = self.s, self.device
        rad_u = rad * 1000.0
        enemy = s.u_active[e] & (s.u_side[e] == (1 - side))
        u_rad_u = torch.round(self._radius[s.u_unit[e]].double() * 1000.0)
        dx = torch.round(s.u_x[e].double() * 1000.0) - (math.floor(x * 1000.0 + 0.5))
        dy = torch.round(s.u_y[e].double() * 1000.0) - (math.floor(y * 1000.0 + 0.5))
        d2 = dx * dx + dy * dy
        eff = (rad_u + u_rad_u) ** 2
        hit = enemy & (d2 <= eff)
        if bool(hit.any()):
            s.u_hp[e, hit] = (s.u_hp[e, hit] - dmg).clamp(min=0.0)
        t_rad_u = torch.where(self.tower_king,
                              torch.full((N_TOWERS,), KING_RADIUS * 1000.0, device=dev),
                              torch.full((N_TOWERS,), T_RADIUS * 1000.0, device=dev))
        dx_t = torch.round(self.tower_pos[e, :, 0].double() * 1000.0) - (math.floor(x * 1000.0 + 0.5))
        dy_t = torch.round(self.tower_pos[e, :, 1].double() * 1000.0) - (math.floor(y * 1000.0 + 0.5))
        d2_t = dx_t * dx_t + dy_t * dy_t
        eff_t = (rad_u + t_rad_u) ** 2
        enemy_t = self.tower_side == (1 - side)
        hit_t = s.tower_alive[e] & enemy_t & (d2_t <= eff_t)
        if bool(hit_t.any()):
            # DamageUtil.adjustForCrownTower: (damage * (100 + pct)) / 100, floored, min 1
            dmg_t = dmg if crown == 0.0 else max(1.0, float(math.floor(dmg * (100.0 + crown) / 100.0)))
            s.tower_hp[e, hit_t] = (s.tower_hp[e, hit_t] - dmg_t).clamp(min=0.0)

    def _spawn_spell_projectile(self, e: int, side: int, x: float, y: float,
                                dmg: float, rad: float, crown: float, speed: float) -> None:
        """Position-targeted spell projectile from the caster's crown tower."""
        s = self.s
        free = (~s.p_active[e]).nonzero(as_tuple=False).flatten()
        if free.numel() == 0:
            return  # overflow: drop (DIVERGENCES #10 policy)
        j = int(free[0])
        sx, sy = KING_POS["blue"] if side == 0 else KING_POS["red"]
        s.p_active[e, j] = True
        s.p_is_spell[e, j] = True
        s.p_side[e, j] = side
        s.p_x[e, j] = sx
        s.p_y[e, j] = sy
        s.p_dmg[e, j] = dmg
        s.p_speed[e, j] = speed
        s.p_radius[e, j] = rad
        s.p_crown[e, j] = crown
        s.p_target[e, j] = -1
        s.p_tx[e, j] = math.floor(x * 1000.0 + 0.5) / 1000.0
        s.p_ty[e, j] = math.floor(y * 1000.0 + 0.5) / 1000.0
        s.p_life[e, j] = 10.0

    # ------------------------------------------------------------------ tick
    def tick(self, n: int = 1, dt: float = TICK_DT) -> None:
        for _ in range(n):
            self._tick_once(dt)

    def _tick_once(self, dt: float) -> None:
        s, dev = self.s, self.device
        s.time = s.time + dt

        # elixir regen with double/triple phases. Java semantics (GameEngine.tick):
        # player regen runs at step 2, checkTimeLimit at step 13 of the SAME tick,
        # and the frame counter increments at the end — so `enterDoubleElixir`
        # (frame >= 2400, i.e. 120.0 s elapsed at the check) only affects the
        # NEXT tick's regen: the first x2 regen lands at elapsed 120.10 s
        # (tick 2402). gpusim applies the multiplier to the tick that ENDS at
        # `time`, hence the strict `> T + TICK_DT` (pinned by test_m3).
        rate = (1.0
                + (s.time > DOUBLE_ELIXIR_T + TICK_DT + 1e-4).float()
                + (s.time > TRIPLE_ELIXIR_T + TICK_DT + 1e-4).float())
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

        # sync-delayed spell casts (Java DeploymentSystem: 1.0 s after the request;
        # flying spells launch their projectile and start flying the same tick)
        s.cast_timer = torch.clamp(s.cast_timer - dt, min=0)
        cast_fired = (s.cast_timer <= 0) & (s.cast_pending >= 0)
        if bool(cast_fired.any()):
            for e, side, j in torch.nonzero(cast_fired, as_tuple=False).tolist():
                ci = int(s.cast_pending[e, side, j])
                self._cast_spell_now(e, side, ci,
                                     float(s.cast_pos[e, side, j, 0]),
                                     float(s.cast_pos[e, side, j, 1]))
                s.cast_pending[e, side, j] = -1

        # timers
        s.u_deploy = torch.clamp(s.u_deploy - dt, min=0)
        s.tower_cd = torch.clamp(s.tower_cd - dt, min=0)

        # king activation: damage taken OR a friendly princess destroyed (Tower
        # activation rule). Activation starts a 1.0 s wake-up (Tower.activate()
        # sets activationTimer = 1.0f, decremented in the same tick).
        blue_princess_dead = (~s.tower_alive[:, 1]) | (~s.tower_alive[:, 2])
        red_princess_dead = (~s.tower_alive[:, 4]) | (~s.tower_alive[:, 5])
        now_active = torch.stack([blue_princess_dead, red_princess_dead], dim=1) \
            | (s.tower_hp[:, [0, 3]] < (T_KING_HP - 1e-6))
        newly = now_active & ~s.king_active
        s.king_active = s.king_active | now_active
        s.t_wake = torch.where(newly, torch.full_like(s.t_wake, 1.0), s.t_wake)
        s.t_wake = torch.clamp(s.t_wake - dt, min=0)

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

        # squared distances attacker->target in whole game units (B, MAX, T):
        # positions are lattice points, so the differences and squares are exact
        # integer arithmetic (Java uses long unit geometry — keeps inclusive
        # boundary checks on the same tick)
        u_xu = torch.round(s.u_x.double() * 1000.0)
        u_yu = torch.round(s.u_y.double() * 1000.0)
        t_xu = torch.round(t_x.double() * 1000.0)
        t_yu = torch.round(t_y.double() * 1000.0)
        rad_u = torch.round(self._radius[s.u_unit].double() * 1000.0)
        t_rad_u = torch.round(t_rad.double() * 1000.0)
        rng_u = torch.round(a_rng.double() * 1000.0)
        sight_u = torch.round(a_sight.double() * 1000.0)
        dx = u_xu.unsqueeze(2) - t_xu.unsqueeze(1)
        dy = u_yu.unsqueeze(2) - t_yu.unsqueeze(1)
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
        sight_sq = (sight_u.unsqueeze(2) + rad_u.unsqueeze(2) + t_rad_u.unsqueeze(1)) ** 2
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
        ret = ((sight_u * TARGET_RETENTION).unsqueeze(2) + rad_u.unsqueeze(2) + t_rad_u.unsqueeze(1)) ** 2
        cur_ret_sq = torch.gather(ret, 2, cur_idx.unsqueeze(2)).squeeze(2)
        keep = cur_valid & cur_enemy & cur_alive & cur_typeok & (cur_d2_raw <= cur_ret_sq)
        tbo = a_only_b > 0  # targetOnlyBuildings: always retarget nearest building
        new_tgt = torch.where(keep & ~tbo, cur, torch.where(has_any, nearest, torch.full_like(nearest, -1)))
        s.u_tgt = new_tgt

        # attack range (edge-to-edge inclusive)
        rng_sq = (rng_u.unsqueeze(2) + rad_u.unsqueeze(2) + t_rad_u.unsqueeze(1)) ** 2
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

        # execute attack: melee hits instantly; every ranged unit fires a flying
        # projectile (Java CombatSystem.fireRangedAttack: Projectile with the
        # combat's projectile stats, or the default 15 tiles/s when it has none).
        # Fire condition mirrors Java `isWindingUp() = windup > 0`: no epsilon —
        # the fp32 residue lands the same tick as the reference (#17 revisited).
        fire = s.u_attacking & (s.u_windup <= 0) & in_range & a_active
        if bool(fire.any()):
            dmg = self._dmg[s.u_unit]
            proj_speed = self.t.u_proj_speed[s.u_unit]
            is_ranged = a_rng >= RANGED_THRESHOLD
            via_proj = fire & is_ranged
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
            # projectiles (speed 0 -> Java default 15 tiles/s)
            if bool(via_proj.any()):
                for e, i in torch.nonzero(via_proj, as_tuple=False).tolist():
                    sp = float(proj_speed[e, i])
                    if sp <= 0.0:
                        sp = DEFAULT_PROJ_SPEED_T
                    self._spawn_projectile(e, i, int(sel[e, i]), float(dmg[e, i]), sp)
            # finishAttack: currentCooldown = 0 (immediate chaining via windup)
            s.u_attacking = s.u_attacking & ~fire
            s.u_atk = torch.where(fire, torch.zeros_like(s.u_atk), s.u_atk)

    def _spawn_projectile(self, e: int, i: int, target_slot: int, dmg: float, speed: float) -> None:
        """Entity-targeted attack projectile (Java ProjectileFactory): starts at the
        attacker's centre, homes on the target, hits within ``speed*dt`` +
        rounding slack or the target's collision radius."""
        s = self.s
        free = (~s.p_active[e]).nonzero(as_tuple=False).flatten()
        if free.numel() == 0:
            return  # overflow: drop (DIVERGENCES #10 policy)
        j = int(free[0])
        s.p_active[e, j] = True
        s.p_is_spell[e, j] = False
        s.p_side[e, j] = int(s.u_side[e, i])
        s.p_x[e, j] = math.floor(float(s.u_x[e, i]) * 1000.0 + 0.5) / 1000.0
        s.p_y[e, j] = math.floor(float(s.u_y[e, i]) * 1000.0 + 0.5) / 1000.0
        s.p_dmg[e, j] = dmg
        s.p_speed[e, j] = speed
        s.p_radius[e, j] = 0.0
        s.p_crown[e, j] = 0.0
        s.p_target[e, j] = target_slot
        s.p_tx[e, j] = s.p_x[e, j]
        s.p_ty[e, j] = s.p_y[e, j]
        s.p_life[e, j] = 5.0

    def _spawn_shot(self, e: int, ti: int, target_slot: int, dmg: float) -> None:
        """Tower shot (Java: tower Combat has no projectile stats -> default-speed
        (15 tiles/s) homing projectile from the tower centre)."""
        s = self.s
        free = (~s.p_active[e]).nonzero(as_tuple=False).flatten()
        if free.numel() == 0:
            return  # overflow: drop (DIVERGENCES #10 policy)
        j = int(free[0])
        s.p_active[e, j] = True
        s.p_is_spell[e, j] = False
        s.p_side[e, j] = int(self.tower_side[ti])
        s.p_x[e, j] = float(self.tower_pos[e, ti, 0])
        s.p_y[e, j] = float(self.tower_pos[e, ti, 1])
        s.p_dmg[e, j] = dmg
        s.p_speed[e, j] = DEFAULT_PROJ_SPEED_T
        s.p_radius[e, j] = 0.0
        s.p_crown[e, j] = 0.0
        s.p_target[e, j] = target_slot
        s.p_tx[e, j] = math.floor(float(s.u_x[e, target_slot]) * 1000.0 + 0.5) / 1000.0
        s.p_ty[e, j] = math.floor(float(s.u_y[e, target_slot]) * 1000.0 + 0.5) / 1000.0
        s.p_life[e, j] = 5.0

    def _projectiles(self, dt: float) -> None:
        """Projectile lifecycle, Java semantics (ProjectileSystem.update).

        Entity-targeted shots (unit attacks, tower shots): homing; the shot
        disappears when the target dies; impact when the gap is within
        ``speed*dt + Position.MAX_ROUNDING_DISTANCE`` or the target's collision
        radius. Position-targeted spells fly to the cast point and detonate
        there (AOE), snapping to it on arrival.
        """
        s = self.s
        if not bool(s.p_active.any()):
            return
        for e, j in torch.nonzero(s.p_active, as_tuple=False).tolist():
            if not bool(s.p_active[e, j]):
                continue  # deactivated earlier in this loop
            step = float(s.p_speed[e, j]) * dt
            if bool(s.p_is_spell[e, j]):
                self._update_spell_projectile(e, j, step)
            else:
                self._update_entity_projectile(e, j, step)
            if not bool(s.p_active[e, j]):
                continue
            s.p_life[e, j] = float(s.p_life[e, j]) - dt
            if float(s.p_life[e, j]) <= 0:
                s.p_active[e, j] = False

    def _update_spell_projectile(self, e: int, j: int, step: float) -> None:
        s = self.s
        tx, ty = float(s.p_tx[e, j]), float(s.p_ty[e, j])
        px, py = float(s.p_x[e, j]), float(s.p_y[e, j])
        dx = tx - px
        dy = ty - py
        dist = math.sqrt(dx * dx + dy * dy)
        # arrival check in game units (Java: moveDistance + MAX_ROUNDING_DISTANCE)
        dist_u = dist * 1000.0
        step_u = step * 1000.0
        if dist_u <= step_u + 0.70710678:
            s.p_x[e, j] = tx
            s.p_y[e, j] = ty
            self._apply_spell_aoe(e, int(s.p_side[e, j]), tx, ty,
                                  float(s.p_dmg[e, j]), float(s.p_radius[e, j]),
                                  float(s.p_crown[e, j]))
            s.p_active[e, j] = False
        else:
            r = step / dist
            s.p_x[e, j] = px + dx * r
            s.p_y[e, j] = py + dy * r

    def _update_entity_projectile(self, e: int, j: int, step: float) -> None:
        s = self.s
        tgt = int(s.p_target[e, j])
        if tgt < 0 or tgt >= MAX_UNITS + N_TOWERS:
            s.p_active[e, j] = False
            return
        if tgt < MAX_UNITS:
            alive = bool(s.u_active[e, tgt])
            tx, ty = float(s.u_x[e, tgt]), float(s.u_y[e, tgt])
            trad = float(self._radius[int(s.u_unit[e, tgt])])
        else:
            ti = tgt - MAX_UNITS
            alive = bool(s.tower_alive[e, ti])
            tx, ty = float(self.tower_pos[e, ti, 0]), float(self.tower_pos[e, ti, 1])
            trad = KING_RADIUS if bool(self.tower_king[ti]) else T_RADIUS
        if not alive:
            s.p_active[e, j] = False  # homing shot disappears with its target
            return
        dx = tx - float(s.p_x[e, j])
        dy = ty - float(s.p_y[e, j])
        dist = math.sqrt(dx * dx + dy * dy)
        dist_u = dist * 1000.0
        step_u = step * 1000.0
        trad_u = trad * 1000.0
        # Java: hit when within moveDistance + MAX_ROUNDING_DISTANCE or the
        # target's collision radius (both in game units)
        if dist_u <= step_u + 0.70710678 or dist_u <= trad_u:
            dmg = float(s.p_dmg[e, j])
            radius = float(s.p_radius[e, j])
            if radius > 0.0:
                # AOE-on-impact projectiles detonate at the target position
                self._apply_spell_aoe(e, int(s.p_side[e, j]), tx, ty, dmg, radius,
                                      float(s.p_crown[e, j]))
            elif tgt < MAX_UNITS:
                s.u_hp[e, tgt] = max(0.0, float(s.u_hp[e, tgt]) - dmg)
            else:
                ti = tgt - MAX_UNITS
                crown = float(s.p_crown[e, j])
                d = dmg if crown == 0.0 else max(1.0, float(math.floor(dmg * (100.0 + crown) / 100.0)))
                s.tower_hp[e, ti] = max(0.0, float(s.tower_hp[e, ti]) - d)
            s.p_active[e, j] = False
        else:
            r = step / dist
            s.p_x[e, j] = float(s.p_x[e, j]) + dx * r
            s.p_y[e, j] = float(s.p_y[e, j]) + dy * r

    def _tower_attacks(self, dt: float) -> None:
        """Tower combat, Java-parity: AttackStateMachine windup (attackCooldown
        stat consumed by load, loadTime 0 -> full cooldown each cycle), shot
        projectiles, no epsilon (the fp32 residue lands the same tick as Java)."""
        s, dev = self.s, self.device
        u_rad = self._radius[s.u_unit]
        for ti in range(N_TOWERS):
            is_king = bool(self.tower_king[ti])
            side = int(self.tower_side[ti])
            sight = T_KING_SIGHT if is_king else T_PRINCESS_SIGHT
            rng = T_KING_RANGE if is_king else T_PRINCESS_RANGE
            rad = KING_RADIUS if is_king else T_RADIUS
            cd_stat = T_KING_CD if is_king else T_PRINCESS_CD
            dmg = T_KING_DMG if is_king else T_PRINCESS_DMG
            alive = s.tower_alive[:, ti]
            if is_king:
                alive = alive & s.king_active[:, side] & (s.t_wake[:, side] <= 0)

            # combat timers (Java updateCombatTimers): windup counts down while attacking
            s.t_windup[:, ti] = torch.where(s.t_attacking[:, ti],
                                            s.t_windup[:, ti] - dt, s.t_windup[:, ti])

            # targeting: nearest enemy within sight + radii; keep the current target
            # while it stays within the retention radius (sight * 1.5) — all in
            # whole game units (exact integer geometry, as in the reference)
            enemy = s.u_active & (s.u_side == (1 - side))
            u_rad_u = torch.round(u_rad.double() * 1000.0)
            txu = float(math.floor(float(self.tower_pos[0, ti, 0]) * 1000.0 + 0.5))
            tyu = float(math.floor(float(self.tower_pos[0, ti, 1]) * 1000.0 + 0.5))
            dx = torch.round(s.u_x.double() * 1000.0) - txu
            dy = torch.round(s.u_y.double() * 1000.0) - tyu
            d2 = dx * dx + dy * dy
            sight_sq = (sight * 1000.0 + rad * 1000.0 + u_rad_u) ** 2
            cand = enemy & (d2 <= sight_sq)
            cur = s.t_tgt[:, ti]
            cur_idx = cur.clamp(min=0, max=MAX_UNITS - 1)
            cur_valid = (cur >= 0) & (cur < MAX_UNITS)
            ret_sq = (sight * 1000.0 * TARGET_RETENTION + rad * 1000.0 + u_rad_u) ** 2
            keep_ok = torch.gather((d2 <= ret_sq) & enemy, 1, cur_idx.unsqueeze(1)).squeeze(1)
            keep = cur_valid & keep_ok
            d2m = torch.where(cand, d2, torch.full_like(d2, float("inf")))
            nearest = d2m.argmin(dim=1)
            has = cand.any(dim=1)
            new_tgt = torch.where(keep, cur, torch.where(has, nearest, torch.full_like(nearest, -1)))
            s.t_tgt[:, ti] = new_tgt

            sel = new_tgt.clamp(0, MAX_UNITS - 1)
            d2_sel = torch.gather(d2, 1, sel.unsqueeze(1)).squeeze(1)
            rng_sq = (rng * 1000.0 + rad * 1000.0 + u_rad_u) ** 2
            in_rng = (new_tgt >= 0) & (d2_sel <= torch.gather(rng_sq, 1, sel.unsqueeze(1)).squeeze(1))

            # cancel an in-progress attack when the target leaves range
            cancel = s.t_attacking[:, ti] & ~in_rng
            s.t_windup[:, ti] = torch.where(cancel, torch.zeros_like(s.t_windup[:, ti]),
                                            s.t_windup[:, ti])
            s.t_attacking[:, ti] = s.t_attacking[:, ti] & ~cancel
            # start a new attack sequence (load consumed; towers have loadTime 0)
            start = alive & in_rng & ~s.t_attacking[:, ti]
            s.t_windup[:, ti] = torch.where(start, torch.full_like(s.t_windup[:, ti], cd_stat),
                                            s.t_windup[:, ti])
            s.t_attacking[:, ti] = s.t_attacking[:, ti] | start
            # fire when the windup expires (Java isWindingUp() = windup > 0, no epsilon)
            fire = s.t_attacking[:, ti] & (s.t_windup[:, ti] <= 0) & in_rng & alive
            if bool(fire.any()):
                for e in torch.nonzero(fire, as_tuple=False).flatten().tolist():
                    self._spawn_shot(e, ti, int(new_tgt[e]), float(dmg))
                s.t_attacking[:, ti] = s.t_attacking[:, ti] & ~fire

    def _movement(self, dt: float) -> None:
        """Movement in Java game-unit geometry (BasePathfinder routing + Position's
        float64 sub-unit accumulator, 1/65536 unit carry) — keeps the whole-unit
        lattice exactly like the reference, so range checks land on the same ticks."""
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
        # whole game-unit coordinates (float64, all values integer-valued)
        cur_x = s.u_x.double() * 1000.0
        cur_y = s.u_y.double() * 1000.0
        t_xu = torch.round(t_x.double() * 1000.0)
        t_yu = torch.round(t_y.double() * 1000.0)
        # goal: current target; targetless units advance toward the enemy king
        sel = s.u_tgt.clamp(0, T - 1)
        gx = torch.gather(t_xu, 1, sel)
        gy = torch.gather(t_yu, 1, sel)
        king_x = torch.full_like(cur_x, KING_POS["red"][0] * 1000.0)
        king_y = torch.full_like(cur_y, KING_POS["red"][1] * 1000.0)
        rk_x = torch.full_like(cur_x, KING_POS["blue"][0] * 1000.0)
        rk_y = torch.full_like(cur_y, KING_POS["blue"][1] * 1000.0)
        adv_x = torch.where(s.u_side == 0, king_x, rk_x)
        adv_y = torch.where(s.u_side == 0, king_y, rk_y)
        gx = torch.where(targetless, adv_x, gx)
        gy = torch.where(targetless, adv_y, gy)
        # BasePathfinder port: ground units route via bridges; air flies straight
        air = self.t.u_move_type[s.u_unit] == 1
        north = cur_y > RIVER_Y_MAX_T * 1000.0
        south = cur_y < RIVER_Y_MIN_T * 1000.0
        in_river = ~north & ~south
        cross_n = south & (gy > RIVER_Y_MAX_T * 1000.0)
        cross_s = north & (gy < RIVER_Y_MIN_T * 1000.0)
        bx_l = torch.full_like(cur_x, BRIDGE_LX * 1000.0)
        bx_r = torch.full_like(cur_x, BRIDGE_RX * 1000.0)
        bridge_x = torch.where((cur_x - bx_l).abs() < (cur_x - bx_r).abs(), bx_l, bx_r)
        bx = torch.where((cur_x - bridge_x).abs() < BRIDGE_ALIGN_T * 1000.0, cur_x, bridge_x)
        approach_y = torch.where(cross_n, torch.full_like(cur_y, RIVER_Y_MIN_T * 1000.0),
                                 torch.full_like(cur_y, RIVER_Y_MAX_T * 1000.0))
        pre = (cross_n & (cur_y < (RIVER_Y_MIN_T - APPROACH_TOL_T) * 1000.0)) | \
              (cross_s & (cur_y > (RIVER_Y_MAX_T + APPROACH_TOL_T) * 1000.0))
        wp_y = torch.where(pre, approach_y, torch.full_like(cur_y, RIVER_CENTER_T * 1000.0))
        exit_y = torch.where(gy > cur_y,
                             torch.full_like(cur_y, (RIVER_Y_MAX_T + BOUNDARY_EPS_T) * 1000.0),
                             torch.full_like(cur_y, (RIVER_Y_MIN_T - BOUNDARY_EPS_T) * 1000.0))
        wp_y = torch.where(in_river, exit_y, wp_y)
        use_wp = ~air & (cross_n | cross_s | in_river)
        dir_x = torch.where(use_wp, bx - cur_x, gx - cur_x)
        dir_y = torch.where(use_wp, wp_y - cur_y, gy - cur_y)
        dist = torch.sqrt(dir_x * dir_x + dir_y * dir_y).clamp(min=1e-9)
        # Java GameUnits.rawSpeedToUnitsPerSecond: raw * 1000/60 units/s
        speed_u = self.t.u_speed[s.u_unit].double() * (1000.0 / 60.0)
        step_u = speed_u * dt
        ratio = step_u / dist
        dxu = dir_x * ratio
        dyu = dir_y * ratio
        go = moving | targetless
        sub = 65536.0
        s.u_fx = torch.where(go, s.u_fx + torch.round(dxu * sub), s.u_fx)
        s.u_fy = torch.where(go, s.u_fy + torch.round(dyu * sub), s.u_fy)
        # read: nearest whole unit (floor((f + half) / 65536)), exposed in tiles
        s.u_x = torch.floor((s.u_fx + 32768.0) / sub) / 1000.0
        s.u_y = torch.floor((s.u_fy + 32768.0) / sub) / 1000.0

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
        """Match timing exactly as Java `GameEngine.checkTimeLimit` sees it.

        `checkTimeLimit` runs at step 13 of tick N (frame counter incremented at
        the end), so the regular-time-end check that first reads
        `frame >= 3600` happens during tick 3601 — i.e. at elapsed time
        180.05 s, not 180.00 s. gpusim checks at the END of the tick (time = the
        tick's end time), so the boundary is shifted by half a tick to land on
        the same tick: first trigger at `time >= 180.025` -> 180.05 ✓ (the
        half-tick margin keeps it float-safe). Same for the OT end at 300.05 s.
        """
        s = self.s
        # regular time end: crowns decide, else overtime
        at_end = (s.time >= MATCH_END_T + TICK_DT / 2) & ~s.game_over
        if bool(at_end.any()):
            blue, red = s.crowns[:, 0], s.crowns[:, 1]
            decided = at_end & (blue != red)
            if bool(decided.any()):
                s.game_over = s.game_over | decided
                w = torch.where(blue > red, torch.zeros_like(blue), torch.ones_like(blue))
                s.winner = torch.where(decided, w, s.winner)
        # overtime end: crowns, then total tower hp, then draw
        at_final = (s.time >= MATCH_FINAL_T + TICK_DT / 2) & ~s.game_over
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
