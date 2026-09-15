"""List damage events (tower hp deltas) and unit hp deltas from a trace."""
import json
import sys


def events(path, slot=None, unit=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    prev_th = None
    prev_hp = {}
    out = []
    for r in rows:
        if slot is not None:
            cur = r["th"][slot]
            if prev_th is not None and cur != prev_th:
                out.append((r["i"], "th%d" % slot, round(prev_th - cur, 1)))
            prev_th = cur
        if unit is not None:
            for u in r.get("u", []):
                key = (u["s"], u["c"])
                if u["c"] != unit:
                    continue
                ph = prev_hp.get(key)
                if ph is not None and u["hp"] != ph:
                    out.append((r["i"], "%s%d" % ("b" if key[0] == 0 else "r", key[0]), round(ph - u["hp"], 1)))
                prev_hp[key] = u["hp"]
    return out


if __name__ == "__main__":
    p = sys.argv[1]
    slot = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    unit = sys.argv[3] if len(sys.argv) > 3 else None
    for e in events(p, slot, unit):
        print(e)
