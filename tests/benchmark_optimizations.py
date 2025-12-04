"""
Benchmark different optimization strategies for card detection.
Tests GPU batch matching, reduced templates, resolution scaling, etc.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.card_detector_simple import CardDetectorSimple
from config.game_config import CARD_SLOTS

# Try to import torch for GPU benchmarks
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, GPU benchmarks will be skipped")


def load_test_frame():
    """Load a test frame from recordings"""
    recordings_dir = Path("recordings")
    if not recordings_dir.exists():
        return None

    recordings = sorted(
        [d for d in recordings_dir.iterdir() if d.is_dir()], reverse=True
    )
    if not recordings:
        return None

    recording = recordings[0]
    frames = sorted(recording.glob("frame_*.jpg"))
    if not frames:
        return None

    # Use frame from middle (more likely to have cards)
    frame_path = frames[len(frames) // 2]
    return cv2.imread(str(frame_path))


def benchmark_current_implementation(frame, detector, iterations=20):
    """Benchmark the current implementation"""
    print("\n" + "=" * 60)
    print("CURRENT IMPLEMENTATION")
    print("=" * 60)

    # Reset detector state
    detector._locked_cards = ["unknown"] * 4
    detector._slot_elixir_cache = [
        {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0}
        for _ in range(4)
    ]

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        cards = detector.detect_cards_with_debug(frame)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

        if i == 0:
            print(f"   First detection: {[c['name'].split('/')[-1] for c in cards]}")

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    print(f"   Average: {avg:.2f} ms")
    print(f"   Min/Max: {min_t:.2f} / {max_t:.2f} ms")
    print(f"   FPS: {1000/avg:.1f}")

    return avg


def benchmark_single_slot_detection(frame, detector, iterations=50):
    """Benchmark single slot detection breakdown"""
    print("\n" + "=" * 60)
    print("SINGLE SLOT DETECTION BREAKDOWN")
    print("=" * 60)

    x1, y1, x2, y2 = CARD_SLOTS[0]
    card_region = frame[y1:y2, x1:x2]

    # Crop like detector does
    h, w = card_region.shape[:2]
    crop_top = int(h * detector.CROP_TOP)
    crop_bottom = int(h * (1 - detector.CROP_BOTTOM))
    crop_left = int(w * detector.CROP_SIDE)
    crop_right = int(w * (1 - detector.CROP_SIDE))
    card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]

    results = {}

    # 1. Histogram shortlist
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        shortlist = detector._get_shortlist_by_histogram(card_artwork, top_k=10)
        times.append((time.perf_counter() - start) * 1000)
    results["histogram_shortlist"] = sum(times) / len(times)
    print(f"   Histogram shortlist (10): {results['histogram_shortlist']:.3f} ms")

    # 2. Template resize + match (single)
    template_name = list(detector.templates.keys())[0]
    template = detector.templates[template_name]

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        resized = cv2.resize(template, (card_artwork.shape[1], card_artwork.shape[0]))
        result = cv2.matchTemplate(card_artwork, resized, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(result))
        times.append((time.perf_counter() - start) * 1000)
    results["single_template_cpu"] = sum(times) / len(times)
    print(f"   Single template (CPU): {results['single_template_cpu']:.3f} ms")

    # 3. Match 10 templates (shortlist)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for name in shortlist[:10]:
            tpl = detector.templates.get(name)
            if tpl is not None:
                resized = cv2.resize(
                    tpl, (card_artwork.shape[1], card_artwork.shape[0])
                )
                result = cv2.matchTemplate(card_artwork, resized, cv2.TM_CCOEFF_NORMED)
                score = float(np.max(result))
        times.append((time.perf_counter() - start) * 1000)
    results["10_templates_cpu"] = sum(times) / len(times)
    print(f"   10 templates (CPU): {results['10_templates_cpu']:.3f} ms")

    return results


def benchmark_gpu_batch_matching(frame, detector, iterations=20):
    """Benchmark GPU batch template matching"""
    if not TORCH_AVAILABLE:
        print("\n⚠️  Skipping GPU batch benchmark (CUDA not available)")
        return None

    print("\n" + "=" * 60)
    print("GPU BATCH MATCHING EXPERIMENTS")
    print("=" * 60)

    x1, y1, x2, y2 = CARD_SLOTS[0]
    card_region = frame[y1:y2, x1:x2]
    h, w = card_region.shape[:2]
    crop_top = int(h * detector.CROP_TOP)
    crop_bottom = int(h * (1 - detector.CROP_BOTTOM))
    crop_left = int(w * detector.CROP_SIDE)
    crop_right = int(w * (1 - detector.CROP_SIDE))
    card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]

    target_h, target_w = card_artwork.shape[:2]
    device = torch.device("cuda")

    results = {}

    # Get shortlist templates
    shortlist = detector._get_shortlist_by_histogram(card_artwork, top_k=10)

    # Prepare batch of templates on GPU
    templates_batch = []
    for name in shortlist[:10]:
        tpl = detector.templates.get(name)
        if tpl is not None:
            resized = cv2.resize(tpl, (target_w, target_h))
            templates_batch.append(resized)

    if not templates_batch:
        print("   No templates to test")
        return None

    # Convert to GPU tensors
    templates_np = np.stack(templates_batch, axis=0)  # (N, H, W, C)
    templates_tensor = torch.from_numpy(templates_np).float().to(device)
    templates_tensor = templates_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)

    query_tensor = torch.from_numpy(card_artwork).float().to(device)
    query_tensor = query_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Warm up GPU
    for _ in range(5):
        _ = torch.sum(templates_tensor)
    torch.cuda.synchronize()

    # 1. Batch normalized cross-correlation using convolution
    print("\n   Method 1: Batch Cosine Similarity")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Flatten and normalize tensors
        query_flat = query_tensor.reshape(1, -1)  # (1, C*H*W)
        templates_flat = templates_tensor.reshape(
            len(templates_batch), -1
        )  # (N, C*H*W)

        query_norm = query_flat / (query_flat.norm(dim=1, keepdim=True) + 1e-8)
        templates_norm = templates_flat / (
            templates_flat.norm(dim=1, keepdim=True) + 1e-8
        )

        # Compute similarity as dot product
        similarities = torch.mm(templates_norm, query_norm.t()).squeeze()
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    results["gpu_batch_cosine"] = sum(times) / len(times)
    print(
        f"      Average: {results['gpu_batch_cosine']:.3f} ms for {len(templates_batch)} templates"
    )

    # 2. Batch L2 distance
    print("\n   Method 2: Batch L2 Distance")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Flatten tensors
        query_flat = query_tensor.reshape(1, -1)  # (1, C*H*W)
        templates_flat = templates_tensor.reshape(
            len(templates_batch), -1
        )  # (N, C*H*W)

        # Compute L2 distance
        diff = query_flat - templates_flat
        l2_dist = diff.pow(2).sum(dim=1).sqrt()

        best_idx = l2_dist.argmin().item()
        best_score = -l2_dist[best_idx].item()  # Negative because lower is better

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    results["gpu_batch_l2"] = sum(times) / len(times)
    print(
        f"      Average: {results['gpu_batch_l2']:.3f} ms for {len(templates_batch)} templates"
    )

    # 3. Batch histogram comparison on GPU
    print("\n   Method 3: GPU Histogram Comparison")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Simple color histogram - mean of each channel
        query_hist = query_tensor.mean(dim=(2, 3))  # (1, 3)
        templates_hist = templates_tensor.mean(dim=(2, 3))  # (N, 3)

        # L2 distance between histograms
        diff = query_hist - templates_hist
        distances = diff.pow(2).sum(dim=1).sqrt()

        best_idx = distances.argmin().item()

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    results["gpu_histogram"] = sum(times) / len(times)
    print(
        f"      Average: {results['gpu_histogram']:.3f} ms for {len(templates_batch)} templates"
    )

    return results


def benchmark_resolution_scaling(frame, detector, iterations=20):
    """Benchmark detection at different resolutions"""
    print("\n" + "=" * 60)
    print("RESOLUTION SCALING EXPERIMENTS")
    print("=" * 60)

    x1, y1, x2, y2 = CARD_SLOTS[0]
    card_region = frame[y1:y2, x1:x2]
    h, w = card_region.shape[:2]
    crop_top = int(h * detector.CROP_TOP)
    crop_bottom = int(h * (1 - detector.CROP_BOTTOM))
    crop_left = int(w * detector.CROP_SIDE)
    crop_right = int(w * (1 - detector.CROP_SIDE))
    card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]

    original_h, original_w = card_artwork.shape[:2]
    print(f"   Original size: {original_w}x{original_h}")

    results = {}
    scales = [1.0, 0.75, 0.5, 0.25]

    for scale in scales:
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        if new_w < 10 or new_h < 10:
            continue

        # Resize card artwork
        scaled_artwork = cv2.resize(card_artwork, (new_w, new_h))

        # Get first 10 templates and resize them
        template_names = list(detector.templates.keys())[:10]

        times = []
        for _ in range(iterations):
            start = time.perf_counter()

            best_score = 0
            best_name = None

            for name in template_names:
                tpl = detector.templates[name]
                # Resize template to match scaled artwork
                scaled_tpl = cv2.resize(tpl, (new_w, new_h))

                result = cv2.matchTemplate(
                    scaled_artwork, scaled_tpl, cv2.TM_CCOEFF_NORMED
                )
                score = float(np.max(result))

                if score > best_score:
                    best_score = score
                    best_name = name

            times.append((time.perf_counter() - start) * 1000)

        avg = sum(times) / len(times)
        results[f"scale_{scale}"] = avg
        print(f"   Scale {scale:.2f} ({new_w}x{new_h}): {avg:.3f} ms for 10 templates")

    return results


def benchmark_deck_confirmed_mode(frame, detector, iterations=20):
    """Benchmark with deck confirmed (reduced template set)"""
    print("\n" + "=" * 60)
    print("DECK CONFIRMED MODE (Reduced Templates)")
    print("=" * 60)

    # Simulate deck confirmation with 8 cards
    fake_deck = [
        "archers",
        "knight",
        "minions",
        "fireball",
        "arrows",
        "giant",
        "musketeer",
        "hog_rider",
    ]

    # Build allowed templates
    allowed = set()
    for tpl_name in detector.templates.keys():
        base = tpl_name.split("/")[-1]
        if base in fake_deck:
            allowed.add(tpl_name)
        elif any(card in tpl_name for card in fake_deck):
            allowed.add(tpl_name)

    # Add waiting_for_card
    for name in detector.WAITING_CARD_NAMES:
        if name in detector.templates:
            allowed.add(name)

    allowed_list = list(allowed)

    print(f"   Full template count: {len(detector.templates)}")
    print(f"   Deck-restricted count: {len(allowed_list)}")
    print(f"   Templates: {allowed_list}")

    # Benchmark with restricted set
    x1, y1, x2, y2 = CARD_SLOTS[0]
    card_region = frame[y1:y2, x1:x2]
    h, w = card_region.shape[:2]
    crop_top = int(h * detector.CROP_TOP)
    crop_bottom = int(h * (1 - detector.CROP_BOTTOM))
    crop_left = int(w * detector.CROP_SIDE)
    crop_right = int(w * (1 - detector.CROP_SIDE))
    card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]

    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        best_score = 0
        best_name = None

        for name in allowed_list:
            tpl = detector.templates.get(name)
            if tpl is None:
                continue
            resized = cv2.resize(tpl, (card_artwork.shape[1], card_artwork.shape[0]))
            result = cv2.matchTemplate(card_artwork, resized, cv2.TM_CCOEFF_NORMED)
            score = float(np.max(result))

            if score > best_score:
                best_score = score
                best_name = name

        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"\n   Average: {avg:.3f} ms for {len(allowed_list)} templates")
    print(f"   Best match: {best_name} ({best_score:.3f})")

    # Extrapolate to 4 slots
    estimated_4_slots = avg * 4
    print(
        f"   Estimated 4 slots: {estimated_4_slots:.2f} ms ({1000/estimated_4_slots:.1f} FPS)"
    )

    return avg


def benchmark_skip_locked_cards(frame, detector, iterations=20):
    """Benchmark with locked cards skipped (no re-detection)"""
    print("\n" + "=" * 60)
    print("LOCKED CARDS SKIP (Persistence Optimization)")
    print("=" * 60)

    # First pass: detect and lock all cards
    detector._locked_cards = ["unknown"] * 4
    detector._slot_elixir_cache = [
        {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0}
        for _ in range(4)
    ]

    # Run detection 10 times to lock cards
    for _ in range(10):
        detector.detect_cards_with_debug(frame)

    locked = detector._locked_cards.copy()
    print(f"   Locked cards: {locked}")

    # Now benchmark - should be much faster with locked cards
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cards = detector.detect_cards_with_debug(frame)
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"   Average with locked cards: {avg:.2f} ms ({1000/avg:.1f} FPS)")

    return avg


def main():
    print("=" * 70)
    print("CARD DETECTION OPTIMIZATION BENCHMARKS")
    print("=" * 70)

    # Load test frame
    frame = load_test_frame()
    if frame is None:
        print("❌ No test frame found. Please run a recording first.")
        return

    print(f"\n📷 Test frame: {frame.shape[1]}x{frame.shape[0]}")

    # Initialize detector
    print("\n🔧 Initializing detector...")
    detector = CardDetectorSimple()

    # Run benchmarks
    results = {}

    # 1. Current implementation
    results["current"] = benchmark_current_implementation(frame, detector)

    # 2. Single slot breakdown
    results["breakdown"] = benchmark_single_slot_detection(frame, detector)

    # 3. GPU batch matching
    gpu_results = benchmark_gpu_batch_matching(frame, detector)
    if gpu_results:
        results["gpu"] = gpu_results

    # 4. Resolution scaling
    results["resolution"] = benchmark_resolution_scaling(frame, detector)

    # 5. Deck confirmed mode
    results["deck_confirmed"] = benchmark_deck_confirmed_mode(frame, detector)

    # 6. Locked cards skip
    results["locked_skip"] = benchmark_skip_locked_cards(frame, detector)

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    print(
        f"\n📊 Current 4-slot detection: {results['current']:.2f} ms ({1000/results['current']:.1f} FPS)"
    )

    print("\n🎯 Potential optimizations:")

    if "deck_confirmed" in results:
        improvement = (
            (results["current"] - results["deck_confirmed"] * 4)
            / results["current"]
            * 100
        )
        print(
            f"   • Deck confirmed mode: ~{results['deck_confirmed'] * 4:.1f} ms ({improvement:.0f}% faster)"
        )

    if "locked_skip" in results:
        improvement = (
            (results["current"] - results["locked_skip"]) / results["current"] * 100
        )
        print(
            f"   • With locked cards: {results['locked_skip']:.1f} ms ({improvement:.0f}% faster)"
        )

    if gpu_results and "gpu_batch_cosine" in gpu_results:
        # GPU is per 10 templates, estimate for 4 slots
        gpu_time = gpu_results["gpu_batch_cosine"] * 4
        print(f"   • GPU batch matching: ~{gpu_time:.1f} ms (estimated)")

    print("\n💡 Recommendations:")
    print(
        "   1. Enable deck confirmation after seeing 8 cards (reduces templates 166→~16)"
    )
    print("   2. Skip re-detection for locked cards (persistence)")
    print("   3. Consider GPU batch matching for further speedup")
    print("   4. Resolution scaling has diminishing returns (accuracy tradeoff)")


if __name__ == "__main__":
    main()
