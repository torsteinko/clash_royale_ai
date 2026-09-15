#!/usr/bin/env python3
"""L1 constant parity check: crforge card data vs a reference table (card level).

Card-level comparison: crforge cards.json (base cards only; _hero/_EV1 variants
excluded) with stats from units.json, versus:

- noff.gg live dump (2026): sections troops/buildings/spells/towers (fetch via fetch_noff.py)
- cr-api-data (2023): cards_stats_characters.json in a directory

Usage:
    python3 check_constants.py --crforge DIR --reference PATH_OR_DIR [--out REPORT.md]
"""
import argparse
import json
import re
from pathlib import Path

# cr-api reference values are in milli-units (durations ms, range/sight milli-tiles)
CRAPI_SCALE = {
    "hitpoints": 1.0,
    "damage": 1.0,
    "speed": 1.0,
    "sight_range": 0.001,
    "range": 0.001,
    "hit_speed": 0.001,
    "load_time": 0.001,
    "deploy_time": 0.001,
}

SPEED_MAP = {"slow": 45.0, "medium": 60.0, "fast": 90.0, "very fast": 120.0, "veryfast": 120.0}
MELEE_MAP = {"melee: short": 0.8, "melee: medium": 1.2, "melee: long": 1.6}

# crforge internal name -> noff card (renames / internal codenames).
# Candidates marked (?) are verified from the mismatch output on first run.
ALIASES = {
    "axeman": "executioner",
    "blowdartgoblin": "dartgoblin",
    "darkwitch": "nightwitch",
    "witchmother": "motherwitch",
    "ghost": "royalghost",
    "mergemaiden": "spiritempress",
    "giantbuffer": "runegiant",
    "movingcannon": "cannoncart",
    "goblinhutrework": "goblinhut",
    "firespirits": "firespirit",
    "icespirits": "icespirit",
    "darkmagic": "void",
    "minisparkys": "zappies",
    "skeletoncontainernew": "skeletonbarrel",  # (?)
    "barblog": "barbarianbarrel",  # (?)
    "goblinpartyhut": "goblinhut",  # (?) party variant of hut
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_crforge(d: Path):
    """Returns (cards_base, units, variant_count)."""
    cards = json.load(open(d / "cards.json"))
    cards = cards if isinstance(cards, list) else list(cards.values())
    units_raw = json.load(open(d / "units.json"))
    units_items = units_raw if isinstance(units_raw, list) else list(units_raw.values())
    units = {norm(u.get("name") or u.get("id")): u for u in units_items}

    base = []
    variants = 0
    for c in cards:
        nm = c.get("name", "")
        if nm.endswith("_hero") or nm.endswith("_EV1") or nm.endswith("_ev1"):
            variants += 1
            continue
        base.append(c)
    return base, units, variants


def load_noff(path: Path):
    data = json.load(open(path))
    out = {}
    for section in data.values():
        if isinstance(section, dict):
            for slug, card in section.items():
                if isinstance(card, dict) and card.get("name"):
                    out[norm(card["name"])] = card
    return out


def load_crapi(d: Path):
    data = json.load(open(d / "cards_stats_characters.json"))
    return {norm(c["name"]): c for c in data if isinstance(c, dict) and c.get("name")}


def find_match(key: str, ref: dict):
    candidates = [key]
    if key in ALIASES:
        candidates.insert(0, ALIASES[key])
    if key.endswith("s"):
        candidates.append(key[:-1])
    else:
        candidates.append(key + "s")
    for cand in candidates:
        if cand in ref:
            return cand
    return None


def parse_hit_speed(v):
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"([\d.]+)\s*sec", str(v))
    return float(m.group(1)) if m else None


def noff_stats(card: dict):
    s = card.get("stats", {})
    out = {
        "health": s.get("hitpoints"),
        "damage": s.get("damage", s.get("ranged_damage")),
        "attackCooldown": parse_hit_speed(s.get("hit_speed")),
    }
    sp = s.get("speed")
    out["speed"] = SPEED_MAP.get(sp.strip().lower()) if isinstance(sp, str) else sp
    rg = s.get("range")
    out["range"] = rg if not isinstance(rg, str) else None  # Melee:* -> informational only
    return out


def crapi_stats(card: dict):
    out = {}
    for f, scale in CRAPI_SCALE.items():
        v = card.get(f)
        if v is None:
            v = card.get(f + "_per_level", [None])[0]
        if v is not None:
            try:
                out[f] = float(v) * scale
            except (TypeError, ValueError):
                pass
    return out


def num(v):
    """Float if v is a plain number (not dict/str/None), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crforge", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_path = Path(args.reference)
    is_noff = ref_path.is_file()
    ref = load_noff(ref_path) if is_noff else load_crapi(ref_path if ref_path.is_dir() else ref_path.parent)
    kind = "noff.gg live dump (2026-09)" if is_noff else "cr-api-data (Oct 2023)"

    base_cards, units, variant_count = load_crforge(Path(args.crforge))
    print(f"crforge: {len(base_cards)} base cards (+{variant_count} hero/evo variants) | "
          f"reference ({kind}): {len(ref)}")

    matched, no_match = {}, []
    for c in base_cards:
        key = norm(c["name"])
        m = find_match(key, ref)
        if m:
            matched[key] = (c, ref[m])
        else:
            no_match.append(c["name"])

    ref_only = sorted(set(ref) - {find_match(norm(c["name"]), ref) for c in base_cards} - {None})
    print(f"card-matched: {len(matched)} | crforge-only: {len(no_match)} | ref-only: {len(ref_only)}")

    stat_rows, cost_rows = [], []
    for key, (c, r) in sorted(matched.items()):
        # cost
        cost = c.get("cost")
        elx = r.get("elixirs")
        if cost is not None and isinstance(elx, (int, float)) and float(cost) != float(elx):
            cost_rows.append((c["name"], float(cost), float(elx)))

        # stats (troop/building with unit linkage)
        unit = units.get(norm(c.get("unit", "")))
        if not unit:
            continue
        if is_noff:
            rs = noff_stats(r)
            for f in ("health", "damage", "attackCooldown", "speed", "range"):
                cv, rv = num(unit.get(f)), num(rs.get(f))
                if cv is None or rv is None:
                    continue
                if abs(cv - rv) > 1e-6:
                    stat_rows.append((c["name"], f, cv, rv))
        else:
            rs = crapi_stats(r)
            for rf, rv in rs.items():
                cf = {"hitpoints": "health"}.get(rf, rf)
                cv = num(unit.get(cf))
                rv = num(rv)
                if cv is None or rv is None or rv == 0:
                    continue
                if abs(cv - rv) > 1e-6:
                    stat_rows.append((c["name"], cf, cv, rv))

    lines = [f"# L1 fidelity report (card level): crforge vs {kind}", ""]
    lines.append(f"- crforge base cards: {len(base_cards)} (+{variant_count} hero/evo variants, excluded)")
    lines.append(f"- reference cards: {len(ref)} | card-matched: {len(matched)}")
    lines.append(f"- stat mismatches: **{len(stat_rows)}** | elixir-cost mismatches: **{len(cost_rows)}**")
    lines.append("")
    lines.append("## Stat mismatches")
    lines.append("")
    lines.append("| card | field | crforge | reference | delta |")
    lines.append("|---|---|---|---|---|")
    for nm, f, cv, rv in stat_rows:
        lines.append(f"| {nm} | {f} | {cv:g} | {rv:g} | {cv - rv:+g} |")
    lines.append("")
    lines.append("## Elixir cost mismatches")
    lines.append("")
    lines.append("| card | crforge | reference |")
    lines.append("|---|---|---|")
    for nm, cv, rv in cost_rows:
        lines.append(f"| {nm} | {cv:g} | {rv:g} |")
    lines.append("")
    lines.append(f"## In reference but missing from crforge ({len(ref_only)})")
    lines.append("")
    lines.append(", ".join(names_of(ref, ref_only)))
    lines.append("")
    lines.append(f"## In crforge but not in reference ({len(no_match)})")
    lines.append("")
    lines.append(", ".join(sorted(no_match)))
    report = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(report)
        print("report ->", args.out)
    print()
    print(report[:3200])


def names_of(ref: dict, keys):
    return [ref[k].get("name", k) for k in keys]


if __name__ == "__main__":
    main()
