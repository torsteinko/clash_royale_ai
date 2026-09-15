#!/usr/bin/env python3
"""Triage the L1 mismatches into actionable buckets.

Outputs fidelity/triage_l1_<date>.md:
  A. Mechanical data updates (stat rows: crforge value -> reference value)
  B. Manual review (elixir costs, semantics-sensitive rows)
  C. Alias resolution (existing aliases + fingerprint suggestions for unmatched cards)
  D. Missing cards (in reference, absent from crforge)

Usage: python3 triage.py --crforge DIR --reference noff.json [--out FILE]
"""
import argparse
import json
import re
from pathlib import Path

import check_constants as cc

SEMANTIC_WATCH = {"xbow"}  # fields where noff semantics may differ (tick model etc.)


def fingerprint_ref(card):
    s = card.get("stats", {})
    return {
        "cost": card.get("elixirs"),
        "health": s.get("hitpoints"),
        "damage": s.get("damage", s.get("ranged_damage")),
        "speed": cc.SPEED_MAP.get(s.get("speed", "").lower()) if isinstance(s.get("speed"), str) else s.get("speed"),
    }


def fingerprint_crf(card, units):
    u = units.get(cc.norm(card.get("unit", ""))) or {}
    return {
        "cost": card.get("cost"),
        "health": u.get("health"),
        "damage": u.get("damage"),
        "speed": u.get("speed"),
    }


def dist(a, b):
    """Normalized distance over shared numeric fields."""
    keys = [k for k in ("health", "damage", "speed") if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))]
    if not keys:
        return None
    d = 0.0
    for k in keys:
        m = max(abs(a[k]), abs(b[k]), 1.0)
        d += abs(a[k] - b[k]) / m
    return d / len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crforge", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", default="triage_l1.md")
    args = ap.parse_args()

    base_cards, units, _ = cc.load_crforge(Path(args.crforge))
    ref = cc.load_noff(Path(args.reference))
    corr = cc.load_corrections(Path(__file__).parent / "reference" / "corrections.json")

    matched, unmatched_crf = {}, []
    for c in base_cards:
        m = cc.find_match(cc.norm(c["name"]), ref)
        if m:
            matched[cc.norm(c["name"])] = (c, ref[m])
        else:
            unmatched_crf.append(c)
    matched_ref_keys = {cc.find_match(cc.norm(c["name"]), ref) for c in base_cards} - {None}
    ref_only = sorted(set(ref) - matched_ref_keys)

    # Buckets A/B: stat rows and cost rows
    stat_rows, cost_rows = [], []
    for key, (c, r) in sorted(matched.items()):
        corr_here = corr.get(cc.norm(c["name"]), {})
        elx = r.get("elixirs")
        if isinstance(elx, (int, float)):
            if "cost" in corr_here:
                elx = corr_here["cost"]
            if c.get("cost") is not None and float(c["cost"]) != float(elx):
                cost_rows.append((c["name"], float(c["cost"]), float(elx)))
        unit = units.get(cc.norm(c.get("unit", "")))
        if not unit:
            continue
        rs = cc.noff_stats(r)
        for f in ("health", "damage", "attackCooldown", "speed", "range"):
            cv, rv = cc.num(unit.get(f)), cc.num(rs.get(f))
            if cv is None or rv is None:
                continue
            if f in corr_here:
                rv = float(corr_here[f])
            if abs(cv - rv) > 1e-6:
                stat_rows.append((c["name"], cc.norm(c["name"]), f, cv, rv))

    # Bucket C: fingerprint suggestions for unmatched pairs
    suggestions = []
    for rk in ref_only:
        rc = ref[rk]
        rf = fingerprint_ref(rc)
        if rf["health"] is None and rf["damage"] is None:
            continue  # towers/buildings without numeric stats
        best = []
        for c in unmatched_crf:
            cf = fingerprint_crf(c, units)
            d = dist(rf, cf)
            if d is not None:
                best.append((d, c["name"], cf))
        best.sort()
        best = [b for b in best if b[0] < 0.35][:3]
        if best:
            suggestions.append((rc.get("name", rk), rf, best))

    # Write
    L = []
    L.append("# L1 triage — crforge vs current data (2026-09-15)")
    L.append("")
    L.append(f"- matched: {len(matched)} | crforge-only: {len(unmatched_crf)} | ref-only: {len(ref_only)}")
    L.append(f"- bucket A (mechanical stat updates): **{len(stat_rows)}** rows")
    L.append(f"- bucket B (manual review): **{len(cost_rows)}** cost rows (+semantics watch: {', '.join(sorted(SEMANTIC_WATCH))})")
    L.append("")
    L.append("## A. Mechanical stat updates (crforge value -> reference)")
    L.append("")
    L.append("| card | field | from | to |")
    L.append("|---|---|---|---|")
    for nm, key, f, cv, rv in stat_rows:
        L.append(f"| {nm} | {f} | {cv:g} | {rv:g} |")
    L.append("")
    L.append("## B. Manual review")
    L.append("")
    L.append("| card | crforge | reference | note |")
    L.append("|---|---|---|---|")
    for nm, cv, rv in cost_rows:
        L.append(f"| {nm} | {cv:g} | {rv:g} | elixir cost — verify semantics |")
    L.append("")
    L.append("## C. Alias suggestions (ref-only vs crforge-only, by stat fingerprint)")
    L.append("")
    for nm, rf, best in suggestions:
        fp = " ".join(f"{k}={v:g}" for k, v in rf.items() if isinstance(v, (int, float)))
        cand = "; ".join(f"{n} (d={d:.3f})" for d, n, _ in best)
        L.append(f"- **{nm}** [{fp}] -> {cand}")
    L.append("")
    L.append("## D. Missing from crforge (no plausible match)")
    L.append("")
    resolved = {nm for nm, _, _ in suggestions}
    for rk in ref_only:
        nm = ref[rk].get("name", rk)
        if nm not in resolved:
            L.append(f"- {nm}")

    out = Path(args.out)
    out.write_text("\n".join(L))
    print("\n".join(L[:60]))
    print(f"\n... full report -> {out}")


if __name__ == "__main__":
    main()
