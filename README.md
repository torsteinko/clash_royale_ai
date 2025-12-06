# Clash Royale AI — Synthetic Dataset & YOLO Training

Repository to build a synthetic Clash Royale object-detection dataset and train YOLO detectors for game-state perception (troops, buildings, spells, UI elements). This README summarizes the real project layout, how to reproduce dataset generation and training, and next tasks.

---

## Quick summary

- Dataset builder: `dataset/sprites_dataset/build_synthetic_dataset.py`  
  - Generates images at native background size (568×896). Arena rectangle: top-left (0,80) → bottom-right (565,825).
  - Supports `--use-cuda` (PyTorch/CUDA) fast compositing and threaded CPU fallback.
  - Outputs `dataset/<output_dir>/` with `train/ val/ test/`, `classes.json`, `data.yaml`.

- Training scripts:
  - `scripts/train_yolo_synthetic.py` — Ultralytics YOLO training (supports two-stage pretrain→fine tune, rect mode, auto device).
  - `scripts/train_yolo.py` — minimal entry point.
  - Trained runs saved under `runs/` (see `runs/synthetic` and `runs/detect`).

- Detection / game-state code:
  - `detection/` — troop/tower/card detectors and OCR.
  - `game_state/` — state extractor, elixir & deck trackers.
  - `utils/` — screen capture, mouse control helpers.
  - `tools/`, `tests/`, `scripts/` — helpers, benchmarks and test utilities.

---

## Getting started (assumes Windows, venv)

1. Create & activate venv, install dependencies (adjust CUDA wheel as needed):
   - Example:
     pip install -r requirements.txt
     pip install ultralytics
   - For CUDA PyTorch (recommended for training & builder `--use-cuda`):
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

2. Quick smoke dataset generation (verify output)
```powershell
cd f:\clash_royale_ai\dataset\sprites_dataset
python build_synthetic_dataset.py --output ..\synthetic_dataset_test --train 50 --val 10 --test 10 --min-sprites 5 --max-sprites 12 --seed 42 --use-cuda
```

3. Two-stage training (recommended)
```powershell
cd f:\clash_royale_ai
python .\scripts\train_yolo_synthetic.py --mode train --two-stage --model yolo11n.pt --imgsz 1280 --epochs 150 --stage1-imgsz 640 --stage1-epochs 40 --device cuda
```
- If the script selects the wrong dataset folder it will prompt; or pass `--data "F:\clash_royale_ai\dataset\synthetic_dataset\data.yaml"`.

---

## Important file locations

- Dataset builder: `dataset/sprites_dataset/build_synthetic_dataset.py`
- Backgrounds & sprite sources: `dataset/sprites_dataset/backgrounds`, `dataset/sprites_dataset/github_dataset`, `dataset/sprites_dataset/sprites_dataset/`
- Generated dataset: `dataset/synthetic_dataset/` (contains `train/`, `val/`, `test/`, `classes.json`, `data.yaml`)
- Training: `scripts/train_yolo_synthetic.py`
- Runs: `runs/synthetic/` and `runs/detect/`
- Detection runtime: `detection/` (troop_detector.py, tower_detector.py, ocr_reader.py)
- Game state extraction: `game_state/` (state_extractor.py, elixir_manager.py, deck_tracker.py)
- Utilities: `utils/` (screen capture, mouse control)

---

## Notes & recommendations

- Keep generator output at native background resolution (568×896). Use rect training to preserve aspect for portrait inference (720×1280) or resize later.
- Recommended dataset scale: generate in stages (10k → inspect → 50k+) to validate annotation quality.
- Average sprites/image and sprites-per-image control data density; default ranges are configurable in the builder.
- Training strategy: pretrain at 640 then fine-tune at 1280 (rect True) on your RTX 5070 Ti.

---

## Troubleshooting

- "Could not find data.yaml": point `--data` to the correct YAML or run `scripts/fix_data_yaml_synthetic.py`.
- "images not found" / wrong dataset: the trainer lists dataset folders and can prompt to pick one; pass `--data` to disambiguate.
- Malformed `data.yaml` (nested keys): run `scripts/fix_data_yaml_synthetic.py` to rewrite a compatible YAML from `classes.json`.

---

## TODOs / Next tasks

- [ ] Finish full synthetic generation (50k+ train images) in chunks and validate class balance.
- [ ] Add per-class size multipliers and rarity-aware sampling in the builder.
- [ ] Improve generator diagnostics (visual preview, sample viewer).
- [ ] Integration: export detector to runtime model and integrate into bot pipeline (720×1280 inference).
- [ ] Add detection post-processing (tracking, tower-health parsing, bar OCR).
- [ ] Add CI smoke-tests for generator + a short training test.
- [ ] Document runtime API for detectors and state extractor.

---

## Contact / Iteration

Open issues or request patches for:
- different generator sizing rules per-class,
- improved GPU-backed generation (torch compositing),
- alternate training schedules.