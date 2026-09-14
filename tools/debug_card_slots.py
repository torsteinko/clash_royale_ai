#!/usr/bin/env python
"""Debug card-slot template matching for one frame.

Prints the top-N template matches for each hand slot and saves crops for
visual inspection:
    slotN_art.png      -- the cropped artwork used for matching
    slotN_compare.png  -- artwork | top1 | top2 | top3 templates for eyeballing

Usage (repo root):
    ~/venvs/clash/bin/python tools/debug_card_slots.py <frame.jpg> [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config.game_config import CARD_SLOTS  # noqa: E402
from detection.card_detector_simple import CardDetectorSimple  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame", help="Path to a gameplay frame")
    ap.add_argument("--out", default="/tmp/card_debug", help="Output directory")
    ap.add_argument("--top", type=int, default=6, help="Top-N matches to print")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cd = CardDetectorSimple()
    f = cv2.imread(args.frame)
    if f is None:
        print(f"❌ could not read {args.frame}")
        return
    print(f"frame: {args.frame} {f.shape}")

    for idx, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
        region = f[y1:y2, x1:x2]
        h, w = region.shape[:2]
        ct = int(h * cd.CROP_TOP)
        cb = int(h * (1 - cd.CROP_BOTTOM))
        cs = int(w * cd.CROP_SIDE)
        art = region[ct:cb, cs:w - cs]
        if art.size == 0:
            print(f"slot {idx + 1}: empty crop")
            continue

        cv2.imwrite(str(out / f"slot{idx + 1}_art.png"), art)

        scores = []
        for name, tpl in cd.templates.items():
            t = cv2.resize(
                tpl, (art.shape[1], art.shape[0]), interpolation=cv2.INTER_AREA
            )
            s = cd._match_template_cpu(art, t)
            scores.append((s, name))
        scores.sort(reverse=True, key=lambda x: x[0])

        top = [(round(s, 3), n) for s, n in scores[: args.top]]
        print(f"slot {idx + 1} (art {art.shape[1]}x{art.shape[0]}): {top}")

        tiles = [art]
        for s, n in scores[:3]:
            t = cv2.resize(
                cd.templates[n], (art.shape[1], art.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            tiles.append(t)
        cv2.imwrite(str(out / f"slot{idx + 1}_compare.png"), np.hstack(tiles))

    print(f"saved crops to {out}")


if __name__ == "__main__":
    main()
