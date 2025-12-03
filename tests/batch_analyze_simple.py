# tests/batch_analyze_simple.py
"""Batch analyze using the simple card detector (card_detector_simple.py)"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'game_state' package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from game_state.state_extractor import GameStateExtractor


def list_recordings():
    recordings_dir = Path(__file__).parent.parent / "recordings"
    if not recordings_dir.exists():
        return []
    recordings = [d for d in recordings_dir.iterdir() if d.is_dir()]
    return sorted(recordings, reverse=True)


def main():
    print("\n" + "=" * 80)
    print("📊 BATCH ANALYZER (SIMPLE) - Process frames and save debug images")
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
    # remove previous output quietly
    if debug_dir.exists():
        # don't delete, just reuse — create if missing
        pass
    debug_dir.mkdir(exist_ok=True)

    skip_input = input("Save every Nth frame (1=all) [default: 1]: ").strip()
    try:
        skip = int(skip_input) if skip_input else 1
        skip = max(1, skip)
    except:
        skip = 1

    extractor = (
        GameStateExtractor()
    )  # uses card_detector_simple via state_extractor import

    total = len(frames)
    for idx, frame_path in enumerate(frames, 1):
        if (idx - 1) % skip != 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Could not read frame: {frame_path.name}")
            continue

        state = extractor.extract_state(frame, debug=True)
        vis = extractor.visualize_state(frame, state)

        out_name = f"debug_simple_{idx:05d}.jpg"
        out_path = debug_dir / out_name
        cv2.imwrite(str(out_path), vis)

        cards = state.get("cards_in_hand", ["unknown"] * 4)
        print(f"   ✅ Frame {idx}/{total} → {out_name} | Cards: {', '.join(cards)}")

    print("\n✅ Done. Debug images saved to:", debug_dir.resolve())


if __name__ == "__main__":
    main()
