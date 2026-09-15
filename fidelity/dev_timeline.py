#!/usr/bin/env python3
"""Compact per-tick timeline of one trace (for human diffing side by side)."""
import json
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    p = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    only = sys.argv[4] if len(sys.argv) > 4 else None  # card name filter
    prev = None
    for rec in load(p)[start:end]:
        us = rec.get("u", [])
        if only:
            us = [u for u in us if u["c"] == only]
        sig = (
            tuple(rec["el"]),
            tuple(rec["th"]),
            tuple(rec["ta"]),
            tuple(rec["cr"]),
            rec["go"],
            rec["w"],
            tuple((u["s"], u["c"], u["x"], u["y"], u["hp"]) for u in us),
        )
        if sig != prev:
            print(f"i={rec['i']} el={rec['el']} th={rec['th']} cr={rec['cr']} go={rec['go']} w={rec['w']}")
            for u in us:
                print(f"    s{u['s']} {u['c']:<12} x={u['x']:<7} y={u['y']:<7} hp={u['hp']}")
            prev = sig


if __name__ == "__main__":
    main()
