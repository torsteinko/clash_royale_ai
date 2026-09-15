"""Print unit positions over time from two traces side by side."""
import json
import sys


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pos(r, s, c):
    for u in r.get("u", []):
        if u["s"] == s and u["c"] == c:
            return (u["x"], u["y"], u["hp"])
    return None


def main():
    a, b = sys.argv[1], sys.argv[2]
    s, c = int(sys.argv[3]), sys.argv[4]
    step = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    ra, rb = rows(a), rows(b)
    for i in range(0, min(len(ra), len(rb)), step):
        pa, pb = pos(ra[i], s, c), pos(rb[i], s, c)
        mark = "" if pa == pb else "   <-- DIFF"
        print(f"i={i:<5} java={pa}  gpu={pb}{mark}")


if __name__ == "__main__":
    main()
