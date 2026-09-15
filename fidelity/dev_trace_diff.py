#!/usr/bin/env python3
"""Per-tick diff inspection for m4-trace-v1 traces (dev tool for M4.2b/4.3)."""
import json
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def fmt(rec):
    us = rec.get("u", [])
    us_s = ";".join(f"{u['c']}@{u['x']},{u['y']}hp{u['hp']}" for u in us)
    return (f"i={rec['i']} el={rec['el']} th={rec['th']} cr={rec['cr']} "
            f"go={rec['go']} w={rec['w']} u=[{us_s}]")


def main():
    a, b = sys.argv[1], sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else start + 10
    ra, rb = load(a), load(b)
    for i in range(start, min(end, len(ra), len(rb))):
        x, y = ra[i], rb[i]
        if json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True):
            continue
        print(f"--- tick {i}")
        for k in ("el", "th", "ta", "cr", "go", "w"):
            if x.get(k) != y.get(k):
                print(f"  {k}: java={x.get(k)}  gpu={y.get(k)}")
        ua, ub = x.get("u", []), y.get("u", [])
        ja = {(u["s"], u["c"]): (u["x"], u["y"], u["hp"]) for u in ua}
        jb = {(u["s"], u["c"]): (u["x"], u["y"], u["hp"]) for u in ub}
        for key in sorted(set(ja) | set(jb)):
            if ja.get(key) != jb.get(key):
                print(f"  u[{key[0]}:{key[1]}]: java={ja.get(key)}  gpu={jb.get(key)}")


if __name__ == "__main__":
    main()
