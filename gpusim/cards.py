"""Card/unit data tables for the batched GPU sim.

Loads the patched crforge data (fidelity/patched/{cards,units}.json) into
device tensors. Data provenance: crforge base + noff.gg 2026-09 corrections
(see fidelity/) — this is the frozen spec from milestone M0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Official per-level scale factors, derived from the reference arrays
# (fidelity: giant hitpoints_per_level / 1930). Index 0 = level 1.
LEVEL_FACTORS = [
    1.0, 1.1, 1.2098, 1.3295, 1.4596, 1.6, 1.7596, 1.9295, 2.1197,
    2.3295, 2.5596, 2.8098, 3.0896, 3.3896, 3.7197, 4.0896, 4.5,
]


def level_factor(level: int) -> float:
    return LEVEL_FACTORS[max(0, min(level - 1, len(LEVEL_FACTORS) - 1))]


def level_multiplier_java(level: int) -> int:
    """The Java reference's integer-hundredths card multiplier (LevelScaling.java).

    m starts at 100; advance per level: ``m = floor(m * 1.10)``; the scaled stat is
    ``floor(base * m / 100)`` (integer math). Level 11 -> m = 256 (exactly 2.56).
    """
    m = 100
    for _ in range(1, max(1, level)):
        m = int(m * 1.10)
    return m


def scaled_stat_java(base: torch.Tensor, level: int) -> torch.Tensor:
    """Java `LevelScaling.scaleCard`: integer floor(base * m / 100) with m per level."""
    m = level_multiplier_java(level)
    return torch.floor(base.float() * m / 100.0)


@dataclass
class CardTable:
    """Per-card and per-unit tensors. C = cards, U = units."""

    names: list[str] = field(default_factory=list)
    costs: torch.Tensor | None = None            # (C,)
    unit_of_card: torch.Tensor | None = None     # (C,) -> unit index
    spawn_count: torch.Tensor | None = None      # (C,)
    summon_delay: torch.Tensor | None = None     # (C,) seconds between staggered spawns
    formation_offsets: list = field(default_factory=list)  # per card: list[(dx, dy)]

    unit_names: list[str] = field(default_factory=list)
    u_health: torch.Tensor | None = None         # (U,) base (level-1) values
    u_damage: torch.Tensor | None = None
    u_cooldown: torch.Tensor | None = None       # seconds per attack
    u_speed: torch.Tensor | None = None
    u_range: torch.Tensor | None = None
    u_sight: torch.Tensor | None = None
    u_radius: torch.Tensor | None = None         # collision radius, tiles
    u_deploy: torch.Tensor | None = None         # deploy time, seconds
    u_target_type: torch.Tensor | None = None    # 0=ALL 1=GROUND 2=AIR (air-hitting dimension)
    u_only_buildings: torch.Tensor | None = None # true = ignores non-buildings (Giant, Hog, Balloon)
    u_move_type: torch.Tensor | None = None      # 0=GROUND 1=AIR 2=BUILDING
    u_proj_speed: torch.Tensor | None = None     # projectile speed, tiles/s (0 = no projectile)
    u_proj_radius: torch.Tensor | None = None    # projectile hit radius, tiles
    u_loadtime: torch.Tensor | None = None       # hidden loadTime stat (windup pre-charge cap)

    card_types: torch.Tensor | None = None      # (C,) 0=TROOP 1=SPELL 2=BUILDING 3=HERO
    card_spell_as_deploy: torch.Tensor | None = None  # (C,) 1/0: SpellFactory-as-deploy (Log)
    spell_radius: torch.Tensor | None = None     # (C,) tiles (0 for non-spells)
    spell_damage: torch.Tensor | None = None     # (C,) base damage (level 1)
    spell_crown_pct: torch.Tensor | None = None  # (C,) e.g. -70 = deals 30% to crown towers
    spell_hits_air: torch.Tensor | None = None   # (C,) 1/0
    spell_hits_ground: torch.Tensor | None = None
    spell_speed: torch.Tensor | None = None      # (C,) tiles/s; > 0 = flying spell (projectile)
    # ticking area-effect zones (poison/earthquake/tornado; M3.5)
    spell_life: torch.Tensor | None = None       # (C,) zone lifeDuration (s; 0 = one-shot)
    spell_hitspeed: torch.Tensor | None = None   # (C,) zone tick interval (s; 0 = one-shot)
    spell_tick_dmg_base: torch.Tensor | None = None  # (C,) per-tick damage (level-1 base)
    spell_building_pct: torch.Tensor | None = None   # (C,) zone building-damage bonus (350 = x4.5)

    card_index: dict = field(default_factory=dict)   # norm name -> card idx
    unit_index: dict = field(default_factory=dict)   # norm name -> unit idx

    def scaled(self, tensor: torch.Tensor, level: int) -> torch.Tensor:
        return tensor * level_factor(level)

    def __len__(self) -> int:
        return len(self.names)


def _norm(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_tables(data_dir: str | Path, device: str = "cpu") -> CardTable:
    """Load cards.json + units.json (+ projectiles.json) from the patched data dir."""
    data_dir = Path(data_dir)
    cards_raw = json.loads((data_dir / "cards.json").read_text())
    units_raw = json.loads((data_dir / "units.json").read_text())
    cards = cards_raw if isinstance(cards_raw, list) else list(cards_raw.values())
    units = units_raw if isinstance(units_raw, list) else list(units_raw.values())
    proj_path = data_dir / "projectiles.json"
    projectiles = json.loads(proj_path.read_text()) if proj_path.exists() else {}
    # buff definitions (for area-effect zones: the reference derives zone tick
    # damage from the buff's damagePerSecond — see AreaEffectFactory.java)
    buff_path = data_dir / "buffs.json"
    buffs = json.loads(buff_path.read_text()) if buff_path.exists() else {}

    t = CardTable()
    t.unit_names = [u.get("name") or u.get("id") for u in units]
    t.unit_index = {_norm(n): i for i, n in enumerate(t.unit_names)}

    for u in units:
        t.u_health = _cat(t.u_health, float(u.get("health", 0.0)), device)
        t.u_damage = _cat(t.u_damage, float(u.get("damage", 0.0)), device)
        t.u_cooldown = _cat(t.u_cooldown, float(u.get("attackCooldown", 1.0)), device)
        t.u_speed = _cat(t.u_speed, float(u.get("speed", 0.0)), device)
        t.u_range = _cat(t.u_range, float(u.get("range", 1.0)), device)
        t.u_sight = _cat(t.u_sight, float(u.get("sightRange", 5.5)), device)
        t.u_radius = _cat(t.u_radius, float(u.get("collisionRadius", 0.5)), device)
        # deploy duration = deployTime + deployDelay (Java TroopFactory sets
        # `deployTimer = stats.getDeployTime() + stats.getDeployDelay()`; 40
        # units carry a deployDelay, e.g. Musketeer 0.3, Archer 0.4)
        t.u_deploy = _cat(t.u_deploy,
                          float(u.get("deployTime", 1.0)) + float(u.get("deployDelay", 0.0)),
                          device)
        tt = str(u.get("targetType", "ALL")).upper()
        t.u_target_type = _cat(t.u_target_type, {"ALL": 0, "GROUND": 1, "AIR": 2}.get(tt, 0), device)
        t.u_only_buildings = _cat(t.u_only_buildings, float(bool(u.get("targetOnlyBuildings", False))), device)
        mt = str(u.get("movementType", "GROUND")).upper()
        t.u_move_type = _cat(t.u_move_type, {"GROUND": 0, "AIR": 1, "BUILDING": 2}.get(mt, 0), device)
        # projectile mapping: raw speed is on the same scale as unit raw speeds (x1000/60)
        pname = u.get("projectile")
        pdata = projectiles.get(pname, {}) if pname else {}
        praw = float(pdata.get("speed", 0.0)) if pdata else 0.0
        t.u_proj_speed = _cat(t.u_proj_speed, praw * (1000.0 / 60.0) / 1000.0, device)  # tiles/s
        t.u_proj_radius = _cat(t.u_proj_radius, float(pdata.get("projectileRadius", 0.5)) if pdata else 0.0, device)
        t.u_loadtime = _cat(t.u_loadtime, float(u.get("loadTime", 0.0)), device)

    for c in cards:
        t.names.append(c.get("name"))
        t.card_index[_norm(c.get("name"))] = len(t.names) - 1
        t.costs = _cat(t.costs, float(c.get("cost", 0.0)), device)
        unit_key = _norm(c.get("unit", ""))
        t.unit_of_card = _cat(t.unit_of_card, float(t.unit_index.get(unit_key, 0)), device)
        t.spawn_count = _cat(t.spawn_count, float(c.get("count", 1)), device)
        t.summon_delay = _cat(t.summon_delay, float(c.get("summonDeployDelay", 0.0)), device)
        offs = c.get("formationOffsets") or [[0.0, 0.0]]
        t.formation_offsets.append([(float(x), float(y)) for x, y in offs])
        # card type + spell fields
        ctype = {"TROOP": 0, "SPELL": 1, "BUILDING": 2, "HERO": 3}.get(str(c.get("type", "TROOP")).upper(), 0)
        t.card_types = _cat(t.card_types, ctype, device)
        t.card_spell_as_deploy = _cat(
            t.card_spell_as_deploy, float(bool(c.get("spellAsDeploy", False))), device)
        ae = c.get("areaEffect") or {}
        pdata = projectiles.get(c.get("projectile"), {}) if c.get("projectile") else {}
        radius = c.get("radius", ae.get("radius", pdata.get("radius", 0.0)))
        damage = ae.get("damage", pdata.get("damage", 0.0))
        crown = ae.get("crownTowerDamagePercent", pdata.get("crownTowerDamagePercent", 0.0))
        # M3.5 zone fields. Damage derivation ported 1:1 from
        # AreaEffectFactory.deployAreaEffect: when the area effect carries no
        # damage of its own, the per-tick damage is
        # round(buff.damagePerSecond * hitSpeed), scaled by card level at use.
        # Crown percent falls back to the buff's (e.g. Poison -75 -> 25%).
        life = float(ae.get("lifeDuration", 0.0) or 0.0)
        hitspeed = float(ae.get("hitSpeed", 0.0) or 0.0)
        bdef = buffs.get(str(ae.get("buff", "") or ""), {}) or {}
        bdps = float(bdef.get("damagePerSecond", 0.0) or 0.0)
        if float(crown or 0.0) == 0.0 and bdef.get("crownTowerDamagePercent"):
            crown = float(bdef["crownTowerDamagePercent"])
        if float(damage or 0.0) > 0.0:
            tick_base = float(damage)
        elif bdps > 0.0 and hitspeed > 0.0:
            tick_base = float(math.floor(bdps * hitspeed + 0.5))  # Java Math.round
        else:
            tick_base = 0.0
        t.spell_life = _cat(t.spell_life, life, device)
        t.spell_hitspeed = _cat(t.spell_hitspeed, hitspeed, device)
        t.spell_tick_dmg_base = _cat(t.spell_tick_dmg_base, tick_base, device)
        t.spell_building_pct = _cat(
            t.spell_building_pct, float(bdef.get("buildingDamagePercent", 0.0) or 0.0), device)
        t.spell_radius = _cat(t.spell_radius, float(radius or 0.0), device)
        t.spell_damage = _cat(t.spell_damage, float(damage or 0.0), device)
        t.spell_crown_pct = _cat(t.spell_crown_pct, float(crown or 0.0), device)
        t.spell_hits_air = _cat(t.spell_hits_air, float(ae.get("hitsAir", True)), device)
        t.spell_hits_ground = _cat(t.spell_hits_ground, float(ae.get("hitsGround", True)), device)
        # Flying spells (projectile-based): speed in tiles/s (raw value 60 = 1 tile/s)
        praw = float(pdata.get("speed", 0.0)) if pdata else 0.0
        t.spell_speed = _cat(t.spell_speed, praw / 60.0, device)

    assert t.costs is not None and t.unit_of_card is not None and t.spawn_count is not None and t.summon_delay is not None
    t.costs = t.costs.to(torch.float32)
    t.unit_of_card = t.unit_of_card.to(torch.long)
    t.spawn_count = t.spawn_count.to(torch.long)
    t.summon_delay = t.summon_delay.to(torch.float32)
    for name in ("u_health", "u_damage", "u_cooldown", "u_speed", "u_range", "u_sight",
                 "u_radius", "u_deploy", "u_target_type", "u_only_buildings", "u_move_type",
                 "u_proj_speed", "u_proj_radius", "u_loadtime",
                 "card_types", "card_spell_as_deploy", "spell_radius", "spell_damage",
                 "spell_crown_pct", "spell_hits_air", "spell_hits_ground", "spell_speed",
                 "spell_life", "spell_hitspeed", "spell_tick_dmg_base", "spell_building_pct"):
        tensor = getattr(t, name)
        assert tensor is not None
        setattr(t, name, tensor.to(torch.float32))
    return t


def _cat(existing, value: float, device: str) -> torch.Tensor:
    v = torch.tensor([value], dtype=torch.float32, device=device)
    return v if existing is None else torch.cat([existing, v])
