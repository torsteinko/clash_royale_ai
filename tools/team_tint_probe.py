"""Color probe v2 — team tint measurement, background-robust.

Metric: central crop of each detection (15% pad), count strongly blue-dominant
vs red-dominant pixels:  blue = B > R+15 and B > G+8 ; red = R > B+15 and R > G+8
(grass is green-dominant -> excluded; gray -> excluded).
frac = (n_blue - n_red) / (n_blue + n_red)   ->  +1 = clearly blue, -1 = clearly red.

Run from repo root:  ~/venvs/clash/bin/python /tmp/color_probe2.py
"""
import glob
import os
import time

import cv2
import numpy as np
import torch

torch.set_num_threads(2)
from ultralytics import YOLO  # noqa: E402

JUNK_SUBSTR = (
    "tower_bar", "king_tower_bar", "bar-level", "big-text", "clock", "elixir",
    "emote", "evolution_symbol", "background-items", "miner_dirt",
    "skeleton_king_bar", "dagger_duchess_tower_bar", "goblin_bush", "snowman",
)
JUNK_EXACT = {"bar"}


def is_junk(cls):
    return cls in JUNK_EXACT or any(s in cls for s in JUNK_SUBSTR)


def counts(img, xyxy, pad=0.15):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 6:
        return None
    x1 += int(w * pad)
    x2 -= int(w * pad)
    y1 += int(h * pad)
    y2 -= int(h * pad)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    crop = img[y1:y2, x1:x2].astype(np.int16)
    if crop.size < 30:
        return None
    B, G, R = crop[..., 0], crop[..., 1], crop[..., 2]
    nb = int(((B > R + 15) & (B > G + 8)).sum())
    nr = int(((R > B + 15) & (R > G + 8)).sum())
    frac = (nb - nr) / max(1, nb + nr)
    return nb, nr, frac


def save_crop(r, tag, expand=0.0, outdir="/tmp/tint_crops"):
    img = cv2.imread(r["f"])
    x1, y1, x2, y2 = r["box"]
    if expand:
        w, h = x2 - x1, y2 - y1
        x1 -= int(w * expand)
        x2 += int(w * expand)
        y1 -= int(h * expand)
        y2 += int(h * expand)
    pad = 6
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    crop = img[y1:y2, x1:x2]
    stem = os.path.basename(r["f"]).rsplit(".", 1)[0]
    name = os.path.join(
        outdir, f"{tag}_{stem}_{r['cls']}_bf{r['frac']:+.2f}_c{r['conf']:.2f}.png"
    )
    cv2.imwrite(name, crop)
    return name


os.makedirs("/tmp/tint_crops", exist_ok=True)
frames = sorted(glob.glob("recordings/20251203_162744/*.jpg"))
sel = frames[:: max(1, len(frames) // 10)][:10]
model = YOLO("best.pt")

rows = []
t_all = time.time()
for f in sel:
    img = cv2.imread(f)
    res = model(img, verbose=False)[0]
    for b in res.boxes:
        cls = model.names[int(b.cls)]
        conf = float(b.conf)
        c = counts(img, b.xyxy[0].tolist())
        if c is None:
            continue
        nb, nr, frac = c
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        rows.append(
            dict(f=f, cls=cls, conf=conf, nb=nb, nr=nr, frac=frac,
                 yc=(y1 + y2) / 2, box=(x1, y1, x2, y2))
        )

print(f"total detections: {len(rows)} in {time.time() - t_all:.0f}s")
print("junk detections:", sum(1 for r in rows if is_junk(r["cls"])))
real = [r for r in rows if not is_junk(r["cls"])]


def agg(rs, label):
    if not rs:
        print(f"{label:<12} n=0")
        return
    fr = np.array([r["frac"] for r in rs])
    sb = int((fr > 0.25).sum())
    sr = int((fr < -0.25).sum())
    amb = len(fr) - sb - sr
    print(
        f"{label:<12} n={len(rs):3d}  blue_frac mean={fr.mean():+.2f} "
        f"std={fr.std():.2f}  strongBlue={sb:3d} strongRed={sr:3d} ambiguous={amb:3d}"
    )


print("\n--- blue_frac by model-side prefix (non-junk only) ---")
agg([r for r in real if r["cls"].startswith("ally_")], "ALLY model")
agg([r for r in real if r["cls"].startswith("enemy_")], "ENEMY model")
agg(real, "ALL")

mis_ally_red = [r for r in real if r["cls"].startswith("ally_") and r["frac"] < -0.25]
mis_enemy_blue = [r for r in real if r["cls"].startswith("enemy_") and r["frac"] > 0.25]
print(
    f"\nmodel=ally but color STRONG RED: {len(mis_ally_red)}   "
    f"model=enemy but color STRONG BLUE: {len(mis_enemy_blue)}"
)
for r in (mis_ally_red + mis_enemy_blue)[:12]:
    print(
        f"   {os.path.basename(r['f'])} {r['cls']:28s} conf={r['conf']:.2f} "
        f"frac={r['frac']:+.2f} yc={int(r['yc'])}"
    )

knights = [r for r in real if "knight" in r["cls"]]
print("\nknight crops (tight / context):")
for r in knights:
    p1 = save_crop(r, "k")
    p2 = save_crop(r, "kctx", expand=1.5)
    print(f"  {p1}\n  {p2}  {r['cls']} conf={r['conf']:.2f} frac={r['frac']:+.2f} yc={int(r['yc'])}")

ally_tr = sorted([r for r in real if r["cls"].startswith("ally_") and "knight" not in r["cls"]], key=lambda r: -abs(r["frac"]))[:3]
enemy_tr = sorted([r for r in real if r["cls"].startswith("enemy_") and "knight" not in r["cls"]], key=lambda r: -abs(r["frac"]))[:3]
print("\nstrongest ally non-knight crops:")
for r in ally_tr:
    print("  ", save_crop(r, "a", expand=0.6), r["cls"], f"conf={r['conf']:.2f} frac={r['frac']:+.2f} yc={int(r['yc'])}")
print("\nstrongest enemy non-knight crops:")
for r in enemy_tr:
    print("  ", save_crop(r, "e", expand=0.6), r["cls"], f"conf={r['conf']:.2f} frac={r['frac']:+.2f} yc={int(r['yc'])}")

if knights:
    r = knights[0]
    img = cv2.imread(r["f"])
    for rr in [x for x in real if x["f"] == r["f"]]:
        x1, y1, x2, y2 = rr["box"]
        col = (255, 0, 0) if rr["frac"] > 0.25 else (0, 0, 255) if rr["frac"] < -0.25 else (128, 128, 128)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{rr['cls'].replace('_', '')[:14]} {rr['frac']:+.2f}",
                    (max(4, x1 - 40), max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    out = "/tmp/tint_crops/annotated_check.png"
    cv2.imwrite(out, img)
    print("\nannotated:", out, "(blue box = measured blue, red box = measured red, gray = ambiguous)")
