#!/usr/bin/env python3
"""Fetch the full current Clash Royale card database from noff.gg.

The site embeds the complete card DB as `const cards = {...}` in every card page.
We fetch one page, extract the JSON via raw_decode (string-safe), and save it.

Usage: python3 fetch_noff.py [--out DIR]
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
BASE = "https://www.noff.gg"
PAGE = f"{BASE}/clash-royale/card/giant"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


def extract_db(page: str) -> dict:
    marker = "const cards = "
    i = page.find(marker)
    if i < 0:
        raise SystemExit("marker not found in page")
    start = page.index("{", i)
    obj, end = json.JSONDecoder().raw_decode(page, start)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    page = fetch(PAGE)
    print(f"page: {len(page)} bytes")
    db = extract_db(page)

    sections = {k: len(v) for k, v in db.items() if isinstance(v, dict)}
    print("sections:", sections)

    total = sum(sections.values())
    print("total cards:", total)

    raw = json.dumps(db, ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(raw.encode()).hexdigest()[:16]

    db_path = f"{args.out}/noff_cards_2026-09.json"
    with open(db_path, "w") as fh:
        json.dump(db, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"wrote {db_path} ({len(raw)} bytes, sha256:{sha})")

    # quick spot checks
    for section in db.values():
        for slug, card in section.items():
            if isinstance(card, dict) and card.get("name") in ("Giant", "Knight", "Minion Giant"):
                print(f"  {slug}: {json.dumps({k: card.get(k) for k in ('name', 'elixirs', 'stats', 'level_stats')}, ensure_ascii=False)[:300]}")


if __name__ == "__main__":
    sys.exit(main())
