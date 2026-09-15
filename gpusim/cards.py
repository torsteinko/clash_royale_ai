"""Card/unit data tables for the batched GPU sim.

Loads the patched crforge data (fidelity/patched/{cards,units}.json) into
device tensors. Data provenance: crforge base + noff.gg 2026-09 corrections
(see fidelity/) — this is the frozen spec from milestone M0.
"""
from __future__ import annotations

import json
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
        t.u_deploy = _cat(t.u_deploy, float(u.get("deployTime", 1.0)), device)
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

    assert t.costs is not None and t.unit_of_card is not None and t.spawn_count is not None and t.summon_delay is not None
    t.costs = t.costs.to(torch.float32)
    t.unit_of_card = t.unit_of_card.to(torch.long)
    t.spawn_count = t.spawn_count.to(torch.long)
    t.summon_delay = t.summon_delay.to(torch.float32)
    for name in ("u_health", "u_damage", "u_cooldown", "u_speed", "u_range", "u_sight",
                 "u_radius", "u_deploy", "u_target_type", "u_only_buildings", "u_move_type",
                 "u_proj_speed", "u_proj_radius", "u_loadtime"):
        tensor = getattr(t, name)
        assert tensor is not None
        setattr(t, name, tensor.to(torch.float32))
    return t


def _cat(existing, value: float, device: str) -> torch.Tensor:
    v = torch.tensor([value], dtype=torch.float32, device=device)
    return v if existing is None else torch.cat([existing, v])
