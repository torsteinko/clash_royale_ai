# Training Guide (detector + offline RL)

Target machine: **Windows desktop with RTX 5070 Ti** (Blackwell, sm_120).
The agent VM (agent-studio-01) is CPU-only — use it for capture, analysis and
dataset prep, not training.

## 1. Environment on the 5070 Ti machine

```powershell
python -m venv .venv
.venv\Scripts\activate
# Blackwell needs the CUDA 12.8 build of PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics tensorboard
python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

## 2. Detector retraining

Current state (Sept 2026): `best.pt` is a yolo11l trained **only on synthetic
data** — it misfires on junk classes, misses units at the wrong inference
resolution and is asymmetric between sides. Priority: retrain on real frames.

### 2a. Taxonomy

Use the v2 taxonomy — one class per unit, side derived at inference time from
the rendered blue/red tint (`game_state/team_classifier.py`):

```bash
python tools/build_taxonomy_v2.py     # regenerates config/units_v2.yaml
```

296 old classes -> 166 v2 classes (17 junk dropped, 113 duplicates merged).

### 2b. Data sources (verified Sept 14 2026)

| Source | What it is | Size / license | Use for |
|---|---|---|---|
| `wty-yy/Clash-Royale-Dataset` (`images/part2`) | 7,380 real arena frames 568x896, YOLO `.txt` + LabelMe `.json`, ~6,940 annotated | ~576 MB, **MIT** | primary labeled corpus |
| Own recordings + ADB captures | 720x1280 MEmu frames (exact domain) | local | fine-tune + val (pre-label with current model + fix manually) |
| `chrisrca/clash-royale-tv-replays` (HF) | 2,540 replays, 540x960 @10fps, **frames only, no labels** | 1.88 TB total (pull single replays ~80–500 MB), **MIT** | unlabeled domain adaptation |

Note: KataCR class names are hyphen-style (`king-tower`) vs our
underscore-style registry — a label-mapping script is still needed
(GitHub issue: "KataCR label -> taxonomy v2 conversion").

### 2c. Train

```powershell
# from repo root, dataset prepared as datasets/v2/{train,val}/{images,labels}
yolo detect train model=yolo11s.pt data=datasets/v2/data_v2.yaml imgsz=640 batch=16 epochs=200 device=0
```

- Start from `yolo11s` (iteration speed) before going back to `yolo11l`.
- `imgsz=640` — **must match at inference time** (the old pipeline ran
  inference at 1280 on a 640-trained model and silently lost units).
- After training: copy `best.pt` to the repo root (or point
  `yolo_model_path` at it) and verify with
  `python scripts/analyze_recording.py recordings/<dir> --save-debug 20`.

## 3. Offline RL (policy)

```bash
# on the training machine, repo root:
# Data: wty-yy replay episodes (state/action/reward .npy.xz), or the
# MIT-licensed 211 MB subset wty-yy/Clash-Royale-Dataset/replay_data/golem_ai
python tools/patch_replay.py            # local slot ids -> global card ids (now handles .npy.xz)
python policy/offline/train.py --replay-dir replay_data
```

- The dataset format is: `state[t]` = {time, unit_infos[xy,cls,bel], cards
  (hand), elixir}; `action[t]` = {xy placement, card_id (0=no-op / 1-4=hand
  slot)}; `reward[t]` float. Verified length-aligned.
- The slot-vs-global card-id contract is documented in the code but was
  inconsistent between collector and dataset — see the GitHub issue before
  collecting your own data.

## 4. Capture loop (once the emulator is reachable)

```bash
# VM side (over Tailscale):
adb connect <pc-tailnet-ip>:<port>
python scripts/collect_replay_data.py --mode live --duration 300 --serial <serial>
python scripts/analyze_recording.py recordings/<dir> --no-levels
```
