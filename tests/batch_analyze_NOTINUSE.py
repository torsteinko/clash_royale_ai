# tests/batch_analyze.py
"""Batch analyze recorded gameplay and save debug images"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import json
from game_state.state_extractor import GameStateExtractor


def list_recordings():
    """List available recordings"""
    recordings_dir = Path(__file__).parent.parent / "recordings"

    if not recordings_dir.exists():
        return []

    recordings = [d for d in recordings_dir.iterdir() if d.is_dir()]
    return sorted(recordings, reverse=True)


def main():
    print("\n" + "=" * 80)
    print("📊 BATCH ANALYZER - Process all frames and save debug images")
    print("=" * 80)

    # Select recording
    recordings = list_recordings()

    if not recordings:
        print("❌ No recordings found!")
        print("   Run 'python tests/record_gameplay.py' first")
        return

    print(f"\n✅ Found {len(recordings)} recording(s):\n")
    for idx, recording in enumerate(recordings, 1):
        frame_count = len(list(recording.glob("*.jpg")))
        print(f"   [{idx}] {recording.name} - {frame_count} frames")

    # Select
    if len(recordings) == 1:
        selected = recordings[0]
        print(f"\n✅ Auto-selected: {selected.name}")
    else:
        choice = input(f"\nSelect recording (1-{len(recordings)}): ")
        try:
            idx = int(choice) - 1
            selected = recordings[idx] if 0 <= idx < len(recordings) else recordings[0]
        except:
            selected = recordings[0]

    # Load frames
    frames = sorted(selected.glob("frame_*.jpg"))

    if not frames:
        print("❌ No frames found!")
        return

    print(f"\n📂 Processing: {selected.name}")
    print(f"📊 Total frames: {len(frames)}")

    # Create debug output directory
    debug_dir = selected / "debug_output"

    # Clear previous debug output
    import shutil

    if debug_dir.exists():
        try:
            shutil.rmtree(debug_dir)
        except Exception as e:
            print(f"⚠️  Failed to clear existing debug_output: {e}")

    debug_dir.mkdir(exist_ok=True)

    print(f"💾 Debug output: {debug_dir}")

    # Ask for frame skip
    print("\n⚙️  Options:")
    skip = input(
        "   Save every Nth frame (1=all, 5=every 5th, 10=every 10th) [default: 1]: "
    ).strip()

    try:
        frame_skip = int(skip) if skip else 1
    except:
        frame_skip = 1

    frame_skip = max(1, frame_skip)

    # Initialize
    print("\n" + "=" * 80)
    extractor = GameStateExtractor()
    print("=" * 80)

    # Analysis results
    results = {
        "recording": selected.name,
        "total_frames": len(frames),
        "processed_frames": 0,
        "frames": [],
    }

    # Process all frames
    print(f"\n🔄 Processing frames (saving every {frame_skip} frame(s))...\n")

    saved_count = 0

    for idx, frame_path in enumerate(frames, 1):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            continue

        # Extract state with DEBUG MODE
        state = extractor.extract_state(frame, debug=True)

        # Store analysis data
        frame_data = {
            "frame_num": idx,
            "filename": frame_path.name,
            "cards": state["cards_in_hand"],
            "ally_count": state["troops"]["total_ally"],
            "enemy_count": state["troops"]["total_enemy"],
            "ally_troops": [t["type"] for t in state["troops"]["ally"]],
            "enemy_troops": [t["type"] for t in state["troops"]["enemy"]],
            "elixir": state.get("elixir"),
            "match_time": state.get("match_time"),
            "towers_ally": state["towers"].get("ally_towers_alive", 0),
            "towers_enemy": state["towers"].get("enemy_towers_alive", 0),
        }

        results["frames"].append(frame_data)

        # Save debug image (every Nth frame)
        if idx % frame_skip == 0:
            vis_frame = extractor.visualize_state(frame, state)

            # Add frame info overlay
            h, w = vis_frame.shape[:2]
            info_text = f"Frame {idx}/{len(frames)} | Ally: {state['troops']['total_ally']} | Enemy: {state['troops']['total_enemy']}"
            cv2.putText(
                vis_frame,
                info_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Save
            debug_filename = debug_dir / f"debug_{idx:05d}.jpg"
            cv2.imwrite(str(debug_filename), vis_frame)
            saved_count += 1

            print(
                f"   ✅ Frame {idx:05d}/{len(frames)} → {debug_filename.name} | "
                f"Cards: {', '.join(state['cards_in_hand'][:2])}... | "
                f"Ally: {state['troops']['total_ally']} | Enemy: {state['troops']['total_enemy']}"
            )
        else:
            # Just print progress without saving
            if idx % 10 == 0:
                print(f"   ⏭️  Frame {idx:05d}/{len(frames)} (skipped)")

    results["processed_frames"] = len(results["frames"])
    results["saved_images"] = saved_count

    # Save JSON report
    report_path = selected / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("✅ BATCH ANALYSIS COMPLETE")
    print("=" * 80)

    # Calculate stats
    if results["frames"]:
        total_ally = sum(f["ally_count"] for f in results["frames"])
        total_enemy = sum(f["enemy_count"] for f in results["frames"])
        avg_ally = total_ally / len(results["frames"])
        avg_enemy = total_enemy / len(results["frames"])

        # Most common troops
        all_ally_troops = []
        all_enemy_troops = []
        for f in results["frames"]:
            all_ally_troops.extend(f["ally_troops"])
            all_enemy_troops.extend(f["enemy_troops"])

        from collections import Counter

        ally_counter = Counter(all_ally_troops)
        enemy_counter = Counter(all_enemy_troops)

        print(f"\n📊 Statistics:")
        print(f"   • Total frames analyzed: {len(results['frames'])}")
        print(f"   • Debug images saved: {saved_count}")
        print(f"   • Average ally troops: {avg_ally:.1f}")
        print(f"   • Average enemy troops: {avg_enemy:.1f}")

        if ally_counter:
            top_ally = ally_counter.most_common(3)
            print(
                f"   • Top ally troops: {', '.join([f'{t[0]} ({t[1]}x)' for t in top_ally])}"
            )

        if enemy_counter:
            top_enemy = enemy_counter.most_common(3)
            print(
                f"   • Top enemy troops: {', '.join([f'{t[0]} ({t[1]}x)' for t in top_enemy])}"
            )

    print(f"\n💾 Files saved:")
    print(f"   • Debug images: {debug_dir.absolute()}")
    print(f"   • JSON report:  {report_path.absolute()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
