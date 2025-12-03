# tests/replay_recording.py
"""Replay recorded gameplay with detection for debugging"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
from game_state.state_extractor import GameStateExtractor


def list_recordings():
    """List available recordings"""
    recordings_dir = Path(__file__).parent.parent / "recordings"

    if not recordings_dir.exists():
        return []

    recordings = [d for d in recordings_dir.iterdir() if d.is_dir()]
    return sorted(recordings, reverse=True)  # Newest first


def main():
    print("\n" + "=" * 80)
    print("🎬 REPLAY DEBUGGER - Test detection on saved recordings")
    print("=" * 80)

    # List available recordings
    recordings = list_recordings()

    if not recordings:
        print("❌ No recordings found!")
        print("   Run 'python tests/record_gameplay.py' first to record gameplay")
        return

    print(f"\n✅ Found {len(recordings)} recording(s):\n")
    for idx, recording in enumerate(recordings, 1):
        frame_count = len(list(recording.glob("*.jpg")))
        print(f"   [{idx}] {recording.name} - {frame_count} frames")

    # Select recording
    if len(recordings) == 1:
        selected = recordings[0]
        print(f"\n✅ Auto-selected: {selected.name}")
    else:
        choice = input(f"\nSelect recording (1-{len(recordings)}): ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(recordings):
                selected = recordings[idx]
            else:
                selected = recordings[0]
        except:
            selected = recordings[0]

    # Load frames
    frames = sorted(selected.glob("*.jpg"))

    if not frames:
        print("❌ No frames found in recording!")
        return

    print(f"\n📂 Loading: {selected.name}")
    print(f"📊 Frames: {len(frames)}")

    # Initialize detector
    print("\n" + "=" * 80)
    extractor = GameStateExtractor()
    print("=" * 80)

    # Playback options
    print("\n📋 Playback Controls:")
    print("  • SPACE: Pause/Resume")
    print("  • RIGHT ARROW: Next frame (when paused)")
    print("  • LEFT ARROW: Previous frame (when paused)")
    print("  • S: Save debug screenshot")
    print("  • Q/ESC: Quit")
    print("\n" + "=" * 80)
    input("\nPress Enter to start playback...")

    # Playback state
    current_frame_idx = 0
    paused = False
    playback_speed = 0.033  # 30 FPS

    # Debug output directory
    debug_dir = selected / "debug_output"
    debug_dir.mkdir(exist_ok=True)

    print(f"\n▶️  PLAYBACK STARTED\n")
    print(f"💾 Debug screenshots: {debug_dir}\n")

    while current_frame_idx < len(frames):
        # Load frame
        frame_path = frames[current_frame_idx]
        frame = cv2.imread(str(frame_path))

        if frame is None:
            current_frame_idx += 1
            continue

        # Extract state with DEBUG MODE
        state = extractor.extract_state(frame, debug=True)

        # Visualize
        vis_frame = extractor.visualize_state(frame, state)

        # Add frame info
        h, w = vis_frame.shape[:2]
        info_text = f"Frame {current_frame_idx + 1}/{len(frames)} | " + (
            f"Ally: {state['troops']['total_ally']} | Enemy: {state['troops']['total_enemy']}"
        )
        cv2.putText(
            vis_frame,
            info_text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if paused:
            cv2.putText(
                vis_frame,
                "PAUSED",
                (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # Display
        cv2.imshow("Replay Debug", vis_frame)

        # Print state
        print(
            f"\rFrame {current_frame_idx + 1:05d}/{len(frames)} | "
            f"Cards: {', '.join(state['cards_in_hand'][:2])}... | "
            f"Ally: {state['troops']['total_ally']} | "
            f"Enemy: {state['troops']['total_enemy']}",
            end="",
        )

        # Handle keyboard
        wait_time = 1 if paused else int(playback_speed * 1000)
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord(" "):  # SPACE
            paused = not paused
            print(f"\n{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
        elif key == 83:  # RIGHT ARROW
            if paused:
                current_frame_idx = min(current_frame_idx + 1, len(frames) - 1)
        elif key == 81:  # LEFT ARROW
            if paused:
                current_frame_idx = max(current_frame_idx - 1, 0)
        elif key == ord("s"):  # S - Save debug screenshot
            debug_filename = debug_dir / f"debug_{current_frame_idx + 1:05d}.jpg"
            cv2.imwrite(str(debug_filename), vis_frame)
            print(f"\n💾 Saved: {debug_filename.name}")

        if not paused:
            current_frame_idx += 1

    cv2.destroyAllWindows()

    print("\n\n" + "=" * 80)
    print("✅ PLAYBACK COMPLETE")
    print("=" * 80)
    print(f"📂 Debug screenshots: {debug_dir.absolute()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
