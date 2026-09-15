#!/usr/bin/env python3
"""L1 constant parity check: crforge unit data vs reference tables (cr-api-data).

Usage:
    python3 check_constants.py --crforge DIR --reference DIR [--out REPORT.md]

--crforge DIR    directory with units.json / cards.json (from crforge/data)
--reference DIR  directory with cards_stats_characters.json (cr-api-data json docs)
"""
import argparse
import json
import re
from pathlib import Path

# crforge unit field -> reference field + unit conversion factor
# (reference values are in milli-units: HP/damage raw, durations in ms,
#  ranges/sight in milli-tiles; crforge: raw hp/damage, seconds, tiles)
FIELD_MAP = [
    ("health", "hitpoints", 1.0),
    ("damage", "damage", 1.0),
    ("speed", "speed", 1.0),
    ("sightRange", "sight_range", 0.001),
    ("range", "range", 0.001),
    ("attackCooldown", "hit_speed", 0.001),
    ("loadTime", "load_time", 0.001),
    ("deployTime", "deploy_time", 0.001),
]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_crforge_units(d: Path) -> dict:
    units = json.load(open(d / "units.json"))
    items = units if isinstance(units, list) else list(units.values())
    out = {}
    for u in items:
        nm = u.get("name") or u.get("id")
        if nm:
            out[norm(nm)] = u
    return out


def load_reference_chars(d: Path) -> dict:
    chars = json.load(open(d / "cards_stats_characters.json"))
    out = {}
    for c in chars:
        nm = c.get("name")
        if nm:
            out[norm(nm)] = c
    return out


def ref_value(ch: dict, field: str) -> float | None:
    """Base value: prefer the scalar; fall back to per-level[0]."""
    v = ch.get(field)
    if v is None:
        v = ch.get(field + "_per_level", [None])[0]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crforge", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    crf = load_crforge_units(Path(args.crforge))
    ref = load_reference_chars(Path(args.reference))
    print(f"crforge units: {len(crf)} | reference characters: {len(ref)}")

    matched = sorted(set(crf) & set(ref))
    only_crf = sorted(set(crf) - set(ref))
    only_ref = sorted(set(ref) - set(crf))
    print(f"name-matched: {len(matched)} | only crforge: {len(only_crf)} | only ref: {len(only_ref)}")

    rows = []
    proj_dmg = []
    for key in matched:
        u, r = crf[key], ref[key]
        for crf_f, ref_f, scale in FIELD_MAP:
            cv = u.get(crf_f)
            rv = ref_value(r, ref_f)
            if cv is None or rv is None:
                continue
            if ref_f == "damage" and float(rv) == 0:
                # Reference damage lives on the projectile, not the character
                # (archers, wizards, ...). Not comparable here; listed separately.
                proj_dmg.append(key)
                continue
            expect = float(rv) * scale
            cv = float(cv)
            ok = abs(cv - expect) < 1e-6
            if not ok:
                rows.append((key, crf_f, cv, expect, float(rv)))

    lines = ["# L1 fidelity report: crforge vs cr-api-data (Oct 2023 baseline)", ""]
    lines.append(f"- crforge units: {len(crf)} | reference characters: {len(ref)} | name-matched: {len(matched)}")
    lines.append(f"- mismatched field values: **{len(rows)}**")
    lines.append(f"- damage-on-projectile (not comparable here): {len(set(proj_dmg))} units")
    lines.append(f"- only in crforge: {', '.join(only_crf[:30])}")
    lines.append(f"- only in reference: {', '.join(only_ref[:30])}")
    lines.append("")
    lines.append("| unit | field | crforge | reference | ref raw |")
    lines.append("|---|---|---|---|---|")
    for key, f, cv, expect, raw in rows:
        lines.append(f"| {key} | {f} | {cv:g} | {expect:g} | {raw:g} |")
    report = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(report)
        print("report ->", args.out)
    print()
    print(report[:4000])


if __name__ == "__main__":
    main()
