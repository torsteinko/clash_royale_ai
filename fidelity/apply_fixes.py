#!/usr/bin/env python3
"""Apply the L1 triage fixes to crforge's data files, producing patched data.

Reads:  crforge data dir (units.json, cards.json)
        noff reference dump
        corrections.json (verdict overlay)
        check_constants alias map
Writes: <out>/units.json, <out>/cards.json, <out>/changes.json (audit trail)

Only applies mechanical stat updates (bucket A) + verified verdicts.
Semantics-watch cards (xbow) and unverified cost rows are left untouched.

Usage: python3 apply_fixes.py --crforge DIR --reference noff.json --out DIR
"""
import argparse
import json
from pathlib import Path

import check_constants as cc

# Cost fixes applied only when verified (Olsen verdict or obvious bug).
COST_FIXES = {
    # card norm -> new cost, reason
    "skeletoncontainernew": (3, "placeholder 0; verified 3 (Olsen + reference)"),
}

SKIP_FIELDS_WATCH = {  # card norm -> fields left for manual review
    "xbow": {"damage", "attackCooldown"},  # tick-damage semantics differ; verify in L2
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crforge", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crf_dir = Path(args.crforge)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = json.load(open(crf_dir / "units.json"))
    cards = json.load(open(crf_dir / "cards.json"))
    units_items = units if isinstance(units, list) else list(units.values())
    cards_items = cards if isinstance(cards, list) else list(cards.values())

    ref = cc.load_noff(Path(args.reference))
    corr = cc.load_corrections(Path(__file__).parent / "reference" / "corrections.json")
    base_cards, units_by_key, _ = cc.load_crforge(crf_dir)

    changes = []
    conflicts = []

    # 1) stat updates via unit stats (apply on the unit object in-place by name)
    units_index = {}
    for u in units_items:
        nm = cc.norm(u.get("name") or u.get("id"))
        if nm:
            units_index[nm] = u

    def writer_priority(card_name: str, ukey: str) -> int:
        """Canonical card (name == unit key, singular/plural forms) beats others."""
        n = cc.norm(card_name)
        if n == ukey or n.rstrip("s") == ukey or n == ukey + "s":
            return 2
        return 1

    unit_writer = {}  # ukey -> (priority, card name)

    for c in base_cards:
        key = cc.norm(c["name"])
        m = cc.find_match(key, ref)
        if not m:
            continue
        r = ref[m]
        unit = units_by_key.get(cc.norm(c.get("unit", "")))
        if not unit:
            continue
        rs = cc.noff_stats(r)
        ukey = cc.norm(c.get("unit", ""))
        watch = SKIP_FIELDS_WATCH.get(key, set())
        prio = writer_priority(c["name"], ukey)
        prev = unit_writer.get(ukey)
        if prev is not None and prev[0] >= prio:
            conflicts.append({"unit": c.get("unit"), "skipped_card": c["name"],
                              "kept_writer": prev[1], "reason": "shared unit; canonical/first writer kept"})
            continue
        for f in ("health", "damage", "attackCooldown", "speed", "range"):
            if f in watch:
                continue
            cv, rv = cc.num(unit.get(f)), cc.num(rs.get(f))
            if cv is None or rv is None:
                continue
            if f in corr.get(key, {}):
                rv = float(corr[key][f])
            if abs(cv - rv) > 1e-6:
                units_index[ukey][f] = rv
                changes.append({"card": c["name"], "unit": c.get("unit"), "unit_field": f, "from": cv, "to": rv})
        unit_writer[ukey] = (prio, c["name"])

    # 2) cost fixes (cards.json)
    cards_index = {cc.norm(c["name"]): c for c in cards_items}
    for ckey, (new_cost, why) in COST_FIXES.items():
        card = cards_index.get(ckey)
        if card is None:
            continue
        old = card.get("cost")
        if old != new_cost:
            card["cost"] = new_cost
            changes.append({"card": card["name"], "card_field": "cost", "from": old, "to": new_cost, "reason": why})

    # sanity: knight/giant spot check
    g = units_index.get("giant") or {}
    k = units_index.get("knight") or {}
    print("spot: giant health ->", g.get("health"), "| knight health ->", k.get("health"))

    json.dump(units, open(out_dir / "units.json", "w"), indent=1)
    json.dump(cards, open(out_dir / "cards.json", "w"), indent=1)
    json.dump(
        {
            "source": "crforge data patched against noff.gg 2026-09 dump",
            "note": "xbow tick-damage fields left for L2; unverified cost rows untouched",
            "changes": changes,
            "conflicts": conflicts,
        },
        open(out_dir / "changes.json", "w"),
        indent=1,
    )
    print(f"applied {len(changes)} changes, skipped {len(conflicts)} conflicting writers -> {out_dir}")


if __name__ == "__main__":
    main()
