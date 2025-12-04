"""Profile detection pipeline to find bottlenecks"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_state.state_extractor import GameStateExtractor
from detection.card_detector_simple import CardDetectorSimple
from detection.ocr_reader import OCRReader
from config.game_config import CARD_SLOTS


def profile_components(frame: np.ndarray, num_iterations: int = 10):
    """Profile individual components"""

    print("\n" + "=" * 60)
    print("PROFILING DETECTION COMPONENTS")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")
    ocr = OCRReader()
    card_detector = CardDetectorSimple()

    results = {}

    # 1. Profile elixir bar reading (should be fast)
    print("\n[1] Elixir Bar Reading (color-based, no OCR)...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        elixir = ocr.read_elixir(frame)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["elixir_bar"] = avg
    print(f"    Average: {avg:.2f} ms | Result: {elixir}")

    # 2. Profile card elixir OCR (expected to be slow)
    print("\n[2] Card Elixir OCR (pytesseract)...")
    slot = CARD_SLOTS[0]
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        cost = ocr.read_card_elixir_cost(frame, slot)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["card_elixir_ocr"] = avg
    print(f"    Average: {avg:.2f} ms | Result: {cost}")

    # 3. Profile gray card detection
    print("\n[3] Gray Card Detection (saturation-based)...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        is_gray = ocr.detect_gray_card(frame, slot)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["gray_detection"] = avg
    print(f"    Average: {avg:.2f} ms | Result: {is_gray}")

    # 4. Profile histogram shortlist
    print("\n[4] Histogram Shortlist (pre-filtering)...")
    x1, y1, x2, y2 = slot
    card_region = frame[y1:y2, x1:x2]
    # Crop like detector does
    h, w = card_region.shape[:2]
    crop_top = int(h * 0.17)
    crop_bottom = int(h * 0.28)
    crop_side = int(w * 0.12)
    card_artwork = card_region[crop_top : h - crop_bottom, crop_side : w - crop_side]

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        shortlist = card_detector._get_shortlist_by_histogram(card_artwork, top_k=10)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["histogram_shortlist"] = avg
    print(f"    Average: {avg:.2f} ms | Top matches: {shortlist[:3]}")

    # 5. Profile template matching (single template)
    print("\n[5] Template Matching (single template, GPU if available)...")
    template_name = list(card_detector.templates.keys())[0]
    template = card_detector.templates[template_name]
    # Resize to match
    resized_template = cv2.resize(
        template, (card_artwork.shape[1], card_artwork.shape[0])
    )

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        score = card_detector._match_template_gpu(card_artwork, resized_template)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["template_match_single"] = avg
    print(f"    Average: {avg:.2f} ms | Score: {score:.3f}")

    # 6. Profile full card detection (single slot)
    print("\n[6] Full Card Detection (1 slot, includes histogram + matching)...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = card_detector._detect_card_at_position(
            frame, x1, y1, x2, y2, use_gray=False
        )
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["detect_single_slot"] = avg
    print(
        f"    Average: {avg:.2f} ms | Result: {result['name']} ({result['score']:.3f})"
    )

    # 7. Profile full detect_cards_with_debug (4 slots) - FORCES fresh detection each time
    print("\n[7] Full detect_cards_with_debug (4 slots, unlocked)...")
    times = []
    for _ in range(num_iterations):
        # Reset state EACH iteration to force fresh detection (no locking benefit)
        card_detector._locked_cards = ["unknown"] * 4
        card_detector._card_confidence = [0] * 4  # Reset confidence too
        card_detector._waiting_state = [
            {"active": False, "frame_count": 0, "detected_card": None},
            {"active": False, "frame_count": 0, "detected_card": None},
            {"active": False, "frame_count": 0, "detected_card": None},
            {"active": False, "frame_count": 0, "detected_card": None},
        ]
        card_detector._slot_elixir_cache = [
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
        ]
        start = time.perf_counter()
        cards = card_detector.detect_cards_with_debug(frame)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["detect_all_slots_no_ocr"] = avg
    print(f"    Average: {avg:.2f} ms (fresh detection, cost via lookup)")

    # 8. Profile WITHOUT cost lookup - forces OCR fallback (WORST CASE)
    print("\n[8] Full detect_cards_with_debug (4 slots, FORCE OCR fallback)...")
    # Temporarily disable cost lookup to force OCR
    original_card_costs = card_detector._card_costs.copy()
    card_detector._card_costs = {}  # Clear costs to force OCR fallback
    times = []
    for _ in range(num_iterations):
        # Reset state EACH iteration (forces OCR every time)
        card_detector._locked_cards = ["unknown"] * 4
        card_detector._card_confidence = [0] * 4
        card_detector._waiting_state = [
            {"active": False, "frame_count": 0, "detected_card": None},
            {"active": False, "frame_count": 0, "detected_card": None},
            {"active": False, "frame_count": 0, "detected_card": None},
            {"active": False, "frame_count": 0, "detected_card": None},
        ]
        card_detector._slot_elixir_cache = [
            {"cost": None, "needs_read": True, "pending": False, "wait_frames": 100},
            {"cost": None, "needs_read": True, "pending": False, "wait_frames": 100},
            {"cost": None, "needs_read": True, "pending": False, "wait_frames": 100},
            {"cost": None, "needs_read": True, "pending": False, "wait_frames": 100},
        ]
        start = time.perf_counter()
        cards = card_detector.detect_cards_with_debug(frame)
        times.append(time.perf_counter() - start)
    card_detector._card_costs = original_card_costs  # Restore costs
    avg = sum(times) / len(times) * 1000
    results["detect_all_slots_with_ocr"] = avg
    print(f"    Average: {avg:.2f} ms (OCR fallback x4 - worst case)")

    # 9. Profile with LOCKED cards (fast path - skips detection entirely!)
    print("\n[9] Full detect_cards_with_debug (4 slots, ALL LOCKED - fast path)...")
    # Simulate locked cards with known costs
    card_detector._locked_cards = ["hog_rider", "fireball", "ice_spirit", "cannon"]
    card_detector._waiting_state = [
        {"active": False, "frame_count": 0, "detected_card": None},
        {"active": False, "frame_count": 0, "detected_card": None},
        {"active": False, "frame_count": 0, "detected_card": None},
        {"active": False, "frame_count": 0, "detected_card": None},
    ]
    card_detector._slot_elixir_cache = [
        {
            "cost": 4,
            "needs_read": False,
            "pending": False,
            "wait_frames": 0,
        },  # hog_rider
        {
            "cost": 4,
            "needs_read": False,
            "pending": False,
            "wait_frames": 0,
        },  # fireball
        {
            "cost": 1,
            "needs_read": False,
            "pending": False,
            "wait_frames": 0,
        },  # ice_spirit
        {"cost": 3, "needs_read": False, "pending": False, "wait_frames": 0},  # cannon
    ]
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        cards = card_detector.detect_cards_with_debug(frame)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    results["detect_all_slots_locked"] = avg
    print(f"    Average: {avg:.2f} ms (LOCKED CARDS - skips template matching!)")
    print(
        f"    → This is {results['detect_all_slots_no_ocr']/avg:.0f}x faster than unlocked detection"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Time per operation (ms)")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, ms in sorted_results:
        bar = "█" * int(ms / 10)
        print(f"  {name:30} {ms:8.2f} ms  {bar}")

    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    ocr_time = results.get("card_elixir_ocr", 0)
    no_ocr_time = results.get("detect_all_slots_no_ocr", 0)
    with_ocr_time = results.get("detect_all_slots_with_ocr", 0)

    print(f"\n  OCR overhead per slot: ~{ocr_time:.0f} ms")
    print(f"  4-slot detection WITHOUT OCR: ~{no_ocr_time:.0f} ms")
    print(f"  4-slot detection WITH OCR: ~{with_ocr_time:.0f} ms")
    print(
        f"  OCR adds: ~{with_ocr_time - no_ocr_time:.0f} ms ({(with_ocr_time - no_ocr_time) / with_ocr_time * 100:.1f}% of total)"
    )

    target_fps = 10
    target_ms = 1000 / target_fps
    locked_time = results.get("detect_all_slots_locked", 0)

    print(f"\n  Target for {target_fps} FPS: <{target_ms:.0f} ms per frame")
    print(
        f"  Current (no OCR): {no_ocr_time:.0f} ms → {1000/no_ocr_time:.1f} FPS possible"
    )
    if locked_time > 0:
        print(
            f"  Current (LOCKED): {locked_time:.2f} ms → {1000/locked_time:.0f} FPS possible 🚀"
        )

    print("\n" + "-" * 60)
    print("OPTIMIZATION SUMMARY")
    print("-" * 60)
    print(
        f"  ❌ Worst case (OCR every frame):  {with_ocr_time:.0f} ms ({1000/with_ocr_time:.1f} FPS)"
    )
    print(
        f"  ⚡ With cost lookup (no OCR):     {no_ocr_time:.0f} ms ({1000/no_ocr_time:.1f} FPS)"
    )
    if locked_time > 0:
        print(
            f"  🚀 With card locking:             {locked_time:.2f} ms ({1000/locked_time:.0f} FPS)"
        )
        print(
            f"\n  → Card locking is {no_ocr_time/locked_time:.0f}x faster than fresh detection!"
        )

    return results


def main():
    # Find a recording with frames
    recordings_dir = Path("recordings")
    if not recordings_dir.exists():
        print("❌ No recordings directory found")
        return

    recordings = sorted(
        [d for d in recordings_dir.iterdir() if d.is_dir()], reverse=True
    )
    if not recordings:
        print("❌ No recordings found")
        return

    # Use first recording
    recording = recordings[0]
    frames = sorted(recording.glob("frame_*.jpg"))
    if not frames:
        print(f"❌ No frames found in {recording}")
        return

    print(f"📂 Using recording: {recording.name}")
    print(f"📊 Found {len(frames)} frames")

    # Load a frame from the middle (more likely to have cards visible)
    frame_idx = len(frames) // 2
    frame_path = frames[frame_idx]
    print(f"📷 Loading frame: {frame_path.name}")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"❌ Could not read frame")
        return

    print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Run profiling
    profile_components(frame, num_iterations=5)


if __name__ == "__main__":
    main()
