# tests/batch_analyze_simple.py
"""Batch analyze using the simplified state extractor with synthetic model (ally/enemy prefixed classes)
+ OCR debug overlay showing detected elixir and per-card elixir cost
+ Tower HP/level OCR debug overlay
"""
import sys
import shutil
from pathlib import Path

# Ensure project root is on sys.path so 'game_state' package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from game_state.state_extractor import GameStateExtractor
from detection.ocr_reader import OCRReader
from config.game_config import CARD_SLOTS


def list_recordings():
    recordings_dir = Path(__file__).parent.parent / "recordings"
    if not recordings_dir.exists():
        return []
    recordings = [d for d in recordings_dir.iterdir() if d.is_dir()]
    return sorted(recordings, reverse=True)


def _draw_ocr_debug(vis: cv2.Mat, frame: cv2.Mat, ocr: OCRReader):
    """Draw OCR debug info onto vis image (in-place) - shows elixir bar reading."""
    h, w = vis.shape[:2]

    # Read current elixir using bar detection
    current_elixir = ocr.read_elixir(frame)

    # Draw elixir value in the middle of the elixir bar
    # Bar is roughly from x=195 to x=695, y=1234 to y=1250
    BAR_CENTER_X = 445  # Middle of bar (195 + 695) / 2
    BAR_CENTER_Y = 1242  # Middle of bar height

    if current_elixir is not None:
        elixir_text = f"{current_elixir:.1f}"
    else:
        elixir_text = "??"

    # Draw background box for readability
    text_size = cv2.getTextSize(elixir_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = BAR_CENTER_X - text_size[0] // 2
    text_y = BAR_CENTER_Y + text_size[1] // 2

    # Background rectangle
    padding = 4
    cv2.rectangle(
        vis,
        (text_x - padding, text_y - text_size[1] - padding),
        (text_x + text_size[0] + padding, text_y + padding),
        (0, 0, 0),
        -1,
    )

    # Elixir text (white with purple tint)
    cv2.putText(
        vis,
        elixir_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 100, 255),  # Purple-ish
        2,
    )

    # Also draw "ELIXIR" label above for clarity
    label = "ELIXIR"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    label_x = BAR_CENTER_X - label_size[0] // 2
    label_y = BAR_CENTER_Y - text_size[1] - 8

    cv2.putText(
        vis,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )


def main():
    print("\n" + "=" * 80)
    print(
        "📊 BATCH ANALYZER - Process frames with synthetic model (ally/enemy detection)"
    )
    print("=" * 80)

    recordings = list_recordings()
    if not recordings:
        print("❌ No recordings found in 'recordings/'")
        return

    print(f"\n✅ Found {len(recordings)} recording(s):")
    for i, r in enumerate(recordings, 1):
        print(f"   {i}. {r.name}")

    # Choose latest by default
    selected = recordings[0] if len(recordings) == 1 else recordings[0]

    print(f"\n📂 Processing: {selected.name}")
    frames = sorted(selected.glob("frame_*.jpg"))
    if not frames:
        print("❌ No frames found (expected files named frame_*.jpg)")
        return

    print(f"📊 Total frames: {len(frames)}")

    debug_dir = selected / "debug_output_simple"
    debug_dir.mkdir(exist_ok=True)
    # If directory exists, clear it
    if debug_dir.exists():
        try:
            shutil.rmtree(debug_dir)
            debug_dir.mkdir(exist_ok=True)
        except Exception as e:
            print(f"⚠️  Failed to clear existing debug_output_simple: {e}")

    skip_input = input("Save every Nth frame (1=all) [default: 1]: ").strip()
    try:
        skip = int(skip_input) if skip_input else 1
        skip = max(1, skip)
    except:
        skip = 1

    # Ask if user wants OCR debug overlay (slow)
    ocr_debug_input = (
        input("Enable OCR debug overlay? (y/N) [default: N]: ").strip().lower()
    )
    enable_ocr_debug = ocr_debug_input == "y"

    my_deck = [
        "archer",
        "fireball",
        "giant",
        "minions",
        "spear_goblins",
        "arrows",
        "ram_rider",
        "musketeer",
    ]

    extractor = GameStateExtractor()
    ocr = OCRReader() if enable_ocr_debug else None

    import time

    start_time = time.time()
    frames_processed = 0

    total = len(frames)
    for idx, frame_path in enumerate(frames, 1):
        if (idx - 1) % skip != 0:
            continue

        frame_start = time.perf_counter()
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Could not read frame: {frame_path.name}")
            continue

        state = extractor.extract_state(frame, debug=True)
        vis = extractor.visualize_state(frame, state)

        # Draw OCR debug overlay (only if enabled - it's slow!)
        if enable_ocr_debug and ocr is not None:
            _draw_ocr_debug(vis, frame, ocr)

        out_name = f"debug_simple_{idx:05d}.jpg"
        out_path = debug_dir / out_name
        cv2.imwrite(str(out_path), vis)

        frames_processed += 1
        cards = state.get("cards_in_hand", ["unknown"] * 4)

        frame_elapsed = time.perf_counter() - frame_start

        # Show speed stats every 50 frames
        if frames_processed % 50 == 0:
            elapsed = time.time() - start_time
            fps = frames_processed / elapsed
            print(
                f"   ✅ Frame {idx}/{total} → {out_name} | Cards: {', '.join(cards[:2])}... | {fps:.1f} FPS | Frame time: {frame_elapsed:.3f}s"
            )
        else:
            print(
                f"   ✅ Frame {idx}/{total} → {out_name} | Cards: {', '.join(cards)} | Frame time: {frame_elapsed:.3f}s"
            )

    elapsed = time.time() - start_time
    fps = frames_processed / elapsed if elapsed > 0 else 0
    print(
        f"\n✅ Done. Processed {frames_processed} frames in {elapsed:.1f}s ({fps:.1f} FPS)"
    )
    print(f"   Debug images saved to: {debug_dir.resolve()}")


if __name__ == "__main__":
    main()
