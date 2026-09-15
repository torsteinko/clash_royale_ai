#!/usr/bin/env python
"""Analyze a recorded gameplay session and write analysis_report.json.

Non-interactive replacement for the old tests/batch_analyze.py (lost to
history).  Uses the fixed state-extraction pipeline: arena/UI filter, team
from rendered tint, elixir from the purple bar, timer via OCR, deck-cycle
tracking.

Usage (repo root):
    ~/venvs/clash/bin/python scripts/analyze_recording.py recordings/20251203_162744
    ... --every 2            # process every 2nd frame
    ... --no-levels          # skip per-unit level OCR (faster)
    ... --save-debug 10      # save a debug image every 10th processed frame

Report per frame: cards, elixir, match_time, troop names by team, tower
status, deck-cycle snapshot.  Written to <recording>/analysis_report.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402
from game_state.state_extractor import GameStateExtractor  # noqa: E402


def analyze(
    recording_dir: Path,
    every: int = 1,
    save_debug: int = 0,
    read_levels: bool = True,
):
    frames = sorted(recording_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"❌ No frames found in {recording_dir}")
        return None

    print(f"\n{'=' * 80}")
    print(f"📊 ANALYZING {recording_dir.name} — {len(frames)} frames")
    print(f"{'=' * 80}")

    extractor = GameStateExtractor(
        read_levels=read_levels, frame_based_tracking=True
    )

    results = {
        "recording": recording_dir.name,
        "total_frames": len(frames),
        "processed_frames": 0,
        "saved_images": 0,
        "frames": [],
    }

    debug_dir = None
    if save_debug > 0:
        debug_dir = recording_dir / "debug_fixed"
        debug_dir.mkdir(exist_ok=True)

    start = time.time()
    processed = 0
    for idx, frame_path in enumerate(frames, 1):
        if (idx - 1) % every != 0:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️  Could not read {frame_path.name}")
            continue

        state = extractor.extract_state(frame, debug=False)
        entry = {
            "frame_num": idx,
            "filename": frame_path.name,
            "cards": state["cards_in_hand"],
            "elixir": state["elixir"],
            "match_time": state["match_time"],
            "match_time_str": state["match_time_str"],
            "ally_count": state["troops"]["total_ally"],
            "enemy_count": state["troops"]["total_enemy"],
            "ally_troops": [t["type"] for t in state["troops"]["ally"]],
            "enemy_troops": [t["type"] for t in state["troops"]["enemy"]],
            "towers_ally": state["towers"].get("ally_towers_alive", 0),
            "towers_enemy": state["towers"].get("enemy_towers_alive", 0),
            "deck": state.get("deck"),
        }
        results["frames"].append(entry)
        processed += 1

        if debug_dir is not None and (processed % save_debug == 0):
            vis = extractor.visualize_state(frame, state)
            cv2.imwrite(str(debug_dir / f"debug_{idx:05d}.jpg"), vis)
            results["saved_images"] += 1

        if processed % 25 == 0:
            elapsed = time.time() - start
            print(
                f"   … {processed} frames | {processed / elapsed:.1f} fps | "
                f"cards={entry['cards']} elixir={entry['elixir']}",
                flush=True,
            )

    results["processed_frames"] = processed
    report_path = recording_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # -------- summary --------
    from collections import Counter

    ally_c = Counter()
    enemy_c = Counter()
    for e in results["frames"]:
        ally_c.update(e["ally_troops"])
        enemy_c.update(e["enemy_troops"])
    elixirs = [e["elixir"] for e in results["frames"] if e["elixir"] is not None]
    times = [e["match_time"] for e in results["frames"] if e["match_time"] is not None]
    cards_seen = Counter()
    for e in results["frames"]:
        for c in e["cards"]:
            if c != "unknown":
                cards_seen[c] += 1

    elapsed = time.time() - start
    print(f"\n✅ DONE — {processed} frames in {elapsed:.0f}s ({processed / elapsed:.1f} fps)")
    print(f"   elixir reads: {len(elixirs)}/{processed}"
          f" (range {min(elixirs):.1f}-{max(elixirs):.1f})" if elixirs else
          "   elixir reads: 0")
    print(f"   timer reads:  {len(times)}/{processed}")
    print(f"   frames with troops: "
          f"{sum(1 for e in results['frames'] if e['ally_count'] + e['enemy_count'])}")
    print(f"   top ally troops:    {ally_c.most_common(8)}")
    print(f"   top enemy troops:   {enemy_c.most_common(8)}")
    print(f"   top hand cards:     {cards_seen.most_common(8)}")
    print(f"   report: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Recording directory (frame_*.jpg)")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--save-debug", type=int, default=0,
                        help="Save debug image every Nth processed frame (0=off)")
    parser.add_argument("--no-levels", action="store_true",
                        help="Skip unit level OCR (faster)")
    args = parser.parse_args()

    rec = Path(args.recording)
    if not rec.exists():
        print(f"❌ Not found: {rec}")
        sys.exit(1)
    analyze(
        rec,
        every=max(1, args.every),
        save_debug=max(0, args.save_debug),
        read_levels=not args.no_levels,
    )


if __name__ == "__main__":
    main()
