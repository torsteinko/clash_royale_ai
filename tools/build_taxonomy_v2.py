#!/usr/bin/env python
"""Build the v2 (side-less) taxonomy from data.yaml.

Why: the current 296-class taxonomy duplicates every unit into `ally_` and
`enemy_` classes for the same visual object and carries ~13 junk/meta
classes.  V2 keeps ONE class per unit -- the team side is derived at inference
time from the rendered blue/red tint (game_state/team_classifier.py) -- and
drops the junk classes.  This matches how the reference project (KataCR)
structures its classes.

Outputs (written to config/):
  units_v2.yaml       -- nc + names list for the retrained detector
  taxonomy_map.json   -- old class -> new class (null = dropped), plus
                         old-index -> new-index map for migrating label files

Usage (repo root):
    ~/venvs/clash/bin/python tools/build_taxonomy_v2.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.game_config import JUNK_CLASS_EXACT, JUNK_CLASS_SUBSTRINGS  # noqa: E402

REPO = Path(__file__).parent.parent

# Explicit renames / merges where the old taxonomy had duplicates or
# inconsistent naming.  Tune freely -- the tool is data-driven otherwise.
ALIASES = {
    "building_tower": "princess_tower",      # duplicate class on the same towers
    "king": "king_tower",                    # king unit sits on the king tower
    "furnace_rework": "furnace",
    "firespirit": "fire_spirit",
    "royalgiant_evolution": "royal_giant_evolution",
    "baby_goblin_dagger_troop": "baby_goblin",
    "cannoneer_troop": "cannoneer",
    "goblin_queen_troop": "goblin_queen",
    "knives_thrower_troop": "knives_thrower",
    "movingcannon": "moving_cannon",
    "giant_buffing": "giant",
    "elixir_golem_big": "elixir_golem",
    "elixir_golem_mid": "elixir_golem",
    "elixir_golem_small": "elixir_golem",
}
PREFIX_STRIPS = ("building_",)
SUFFIX_STRIPS = ("_troop",)


def is_junk(name: str) -> bool:
    n = (name or "").lower()
    if n in JUNK_CLASS_EXACT:
        return True
    return any(s in n for s in JUNK_CLASS_SUBSTRINGS)


def to_base(old: str) -> str:
    base = old
    for pre in ("ally_", "enemy_"):
        if base.startswith(pre):
            base = base[len(pre):]
            break
    for pre in PREFIX_STRIPS:
        if base.startswith(pre) and base != "building_tower":
            base = base[len(pre):]
            break
    for suf in SUFFIX_STRIPS:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return ALIASES.get(base, base)


def main():
    import yaml

    data = yaml.safe_load(open(REPO / "data.yaml"))
    names = list(data["names"])
    print(f"loaded {len(names)} classes from data.yaml")

    old_to_new = {}
    new_names = []
    seen = set()
    dropped = []
    merged = 0
    for old in names:
        if is_junk(old):
            old_to_new[old] = None
            dropped.append(old)
            continue
        base = to_base(old)
        old_to_new[old] = base
        if base in seen:
            merged += 1
        else:
            seen.add(base)
            new_names.append(base)

    old_index_to_new = {}
    for i, old in enumerate(names):
        nb = old_to_new[old]
        old_index_to_new[str(i)] = new_names.index(nb) if nb else None

    out_cfg = REPO / "config"
    out_cfg.mkdir(exist_ok=True)

    units_path = out_cfg / "units_v2.yaml"
    with open(units_path, "w") as f:
        f.write(f"# Taxonomy v2 (side-less, junk removed) - generated {date.today()}\n")
        f.write(f"# {len(names)} old classes -> {len(new_names)} classes\n")
        f.write(f"nc: {len(new_names)}\nnames:\n")
        for n in new_names:
            f.write(f"- {n}\n")

    map_path = out_cfg / "taxonomy_map.json"
    with open(map_path, "w") as f:
        json.dump(
            {
                "version": 2,
                "created": str(date.today()),
                "old_count": len(names),
                "new_count": len(new_names),
                "junk_dropped": dropped,
                "map": old_to_new,
                "old_index_to_new_index": old_index_to_new,
                "new_names": new_names,
            },
            f,
            indent=2,
        )

    print(f"\n✅ {len(names)} -> {len(new_names)} classes "
          f"({len(dropped)} junk dropped, {merged} duplicates merged)")
    print(f"   wrote {units_path.relative_to(REPO)} "
          f"and {map_path.relative_to(REPO)}")
    print("\nnew class list:")
    for i, n in enumerate(new_names):
        print(f"  {i:3d}  {n}")


if __name__ == "__main__":
    main()
