# tools/compare_extraction_vs_templates.py
"""Show side-by-side comparison of extracted cards vs templates"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.game_config import CARD_SLOTS
from detection.card_detector import CardDetector


def compare_slot_with_template(
    screenshot_path: str,
    slot_idx: int,
    template_name: str,
    output_dir: str = "tools/check",
):
    """
    Compare a specific slot with a specific template

    Args:
        screenshot_path: Path to screenshot
        slot_idx: Slot index (0-3)
        template_name: Template name (e.g., 'other/waiting_for_card', 'base/minions')
        output_dir: Where to save comparison images
    """

    frame = cv2.imread(screenshot_path)

    if frame is None:
        print(f"❌ Could not load: {screenshot_path}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🔍 SLOT {slot_idx + 1} vs TEMPLATE '{template_name}'")
    print("=" * 80)

    detector = CardDetector()

    # Get slot coordinates
    x1, y1, x2, y2 = CARD_SLOTS[slot_idx]
    h, w = frame.shape[:2]

    if x2 > w or y2 > h:
        print("❌ Slot out of bounds")
        return

    # ===== EXTRACT FROM SCREENSHOT =====
    card_region = frame[y1:y2, x1:x2]
    card_height = card_region.shape[0]
    card_width = card_region.shape[1]

    # Apply same cropping as detector
    crop_top = int(card_height * detector.CROP_TOP)
    crop_bottom = int(card_height * (1 - detector.CROP_BOTTOM))
    crop_left = int(card_width * detector.CROP_SIDE)
    crop_right = int(card_width * (1 - detector.CROP_SIDE))

    extracted = card_region[crop_top:crop_bottom, crop_left:crop_right]

    print(f"\n📤 EXTRACTED FROM SLOT {slot_idx + 1}:")
    print(f"   Full slot size: {card_width}x{card_height}")
    print(f"   Cropped size: {extracted.shape[1]}x{extracted.shape[0]}")
    print(
        f"   Crop ratios: top={detector.CROP_TOP}, bottom={detector.CROP_BOTTOM}, sides={detector.CROP_SIDE}"
    )

    # ===== ANALYZE COLOR PROFILE =====
    hsv = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    avg_hue = np.mean(h_channel)
    avg_saturation = np.mean(s_channel)
    avg_value = np.mean(v_channel)

    dark_blue_mask = (s_channel < 80) & (v_channel > 30) & (v_channel < 180)
    dark_ratio = np.count_nonzero(dark_blue_mask) / dark_blue_mask.size

    print(f"\n📊 COLOR ANALYSIS:")
    print(f"   Avg Hue: {avg_hue:.1f} (blue is ~100-130)")
    print(f"   Avg Saturation: {avg_saturation:.1f} (waiting_for_card < 70)")
    print(f"   Avg Value: {avg_value:.1f} (waiting_for_card 40-170)")
    print(f"   Dark/desaturated ratio: {dark_ratio:.2%}")

    is_likely_waiting = (
        avg_saturation < 70 and 40 < avg_value < 170 and dark_ratio > 0.5
    )
    print(
        f"\n   {'✅ LOOKS LIKE waiting_for_card' if is_likely_waiting else '❌ Does NOT look like waiting_for_card'}"
    )

    # ===== FIND TEMPLATE =====
    template = None
    for name, tmpl in detector.templates.items():
        if template_name in name or name in template_name:
            template = tmpl
            template_name = name
            break

    if template is None:
        print(f"\n❌ Template '{template_name}' not found!")
        print(f"   Available templates: {list(detector.templates.keys())}")
        return

    print(f"\n📥 TEMPLATE: {template_name}")
    print(f"   Size: {template.shape[1]}x{template.shape[0]}")

    # ===== RESIZE FOR COMPARISON =====
    template_resized = cv2.resize(template, (extracted.shape[1], extracted.shape[0]))

    # ===== CALCULATE MATCH SCORES =====
    extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(extracted_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    score_gray = result.max()

    # Also try color matching
    result_color = cv2.matchTemplate(extracted, template_resized, cv2.TM_CCOEFF_NORMED)
    score_color = result_color.max()

    print(f"\n📊 MATCH SCORES:")
    print(f"   Grayscale: {score_gray:.4f}")
    print(f"   Color: {score_color:.4f}")

    # ===== COMPARE WITH OTHER TEMPLATES =====
    print(f"\n📊 COMPARISON WITH ALL TEMPLATES:")
    scores = []
    for name, tmpl in detector.templates.items():
        try:
            tmpl_resized = cv2.resize(tmpl, (extracted.shape[1], extracted.shape[0]))
            tmpl_gray = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(extracted_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            scores.append((name, res.max()))
        except:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n   TOP 10 MATCHES:")
    for i, (name, score) in enumerate(scores[:10]):
        marker = "👈 TARGET" if template_name in name else ""
        print(f"   {i+1}. {name}: {score:.4f} {marker}")

    # ===== SAVE IMAGES =====
    # Save extracted
    cv2.imwrite(str(output_path / f"slot{slot_idx+1}_extracted.png"), extracted)

    # Save template
    cv2.imwrite(
        str(
            output_path
            / f"slot{slot_idx+1}_template_{template_name.replace('/', '_')}.png"
        ),
        template_resized,
    )

    # Save comparison
    comparison = np.hstack([extracted, template_resized])
    cv2.putText(
        comparison, "EXTRACTED", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
    )
    cv2.putText(
        comparison,
        "TEMPLATE",
        (extracted.shape[1] + 5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        comparison,
        f"Score: {score_gray:.3f}",
        (5, comparison.shape[0] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.imwrite(str(output_path / f"slot{slot_idx+1}_comparison.png"), comparison)

    # Save grayscale comparison
    comparison_gray = np.hstack([extracted_gray, template_gray])
    cv2.imwrite(
        str(output_path / f"slot{slot_idx+1}_comparison_gray.png"), comparison_gray
    )

    # Save color analysis visualization
    saturation_vis = cv2.applyColorMap(s_channel, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path / f"slot{slot_idx+1}_saturation.png"), saturation_vis)

    print(f"\n💾 Saved images to: {output_path.absolute()}")
    print("=" * 80)


def compare_extraction_vs_templates(
    screenshot_path: str, card_names: list, output_dir: str = "tools/check"
):
    """
    Compare what's extracted from screenshot vs what's in templates

    Args:
        screenshot_path: Path to screenshot
        card_names: List of 4 card names (e.g., ['archer', 'knight', 'minions', 'fireball'])
        output_dir: Where to save comparison images
    """

    frame = cv2.imread(screenshot_path)

    if frame is None:
        print(f"❌ Could not load: {screenshot_path}")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🔍 EXTRACTION vs TEMPLATE COMPARISON")
    print("=" * 80)
    print(f"\n📸 Screenshot: {frame.shape[1]}x{frame.shape[0]}")
    print(f"🎴 Cards: {', '.join(card_names)}")
    print(f"💾 Output: {output_path.absolute()}\n")

    detector = CardDetector()

    for idx, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
        if idx >= len(card_names):
            break

        card_name = card_names[idx]

        print(f"\n{'='*80}")
        print(f"CARD {idx + 1}: {card_name.upper()}")
        print(f"{'='*80}")

        h, w = frame.shape[:2]

        if x2 > w or y2 > h:
            print("❌ Out of bounds")
            continue

        # ===== EXTRACT FROM SCREENSHOT =====
        card_region = frame[y1:y2, x1:x2]

        card_height = card_region.shape[0]
        card_width = card_region.shape[1]

        # Apply same cropping as detector
        top_crop = detector.CROP_TOP
        bottom_crop = detector.CROP_BOTTOM
        side_crop = detector.CROP_SIDE

        crop_top = int(card_height * top_crop)
        crop_bottom = int(card_height * (1 - bottom_crop))
        crop_left = int(card_width * side_crop)
        crop_right = int(card_width * (1 - side_crop))

        extracted = card_region[crop_top:crop_bottom, crop_left:crop_right]

        if extracted.size == 0:
            print("❌ Extracted region is empty")
            continue

        print(f"\n📤 EXTRACTED FROM SCREENSHOT:")
        print(f"   Size: {extracted.shape[1]}x{extracted.shape[0]}")
        print(f"   Crop: top={top_crop}, bottom={bottom_crop}, sides={side_crop}")

        # ===== FIND MATCHING TEMPLATE =====
        # Look for card_name in templates (could be base/card_name, evolution/card_name, etc.)
        matching_templates = []
        for template_name in detector.templates.keys():
            if card_name in template_name.lower():
                matching_templates.append(template_name)

        if not matching_templates:
            print(f"\n❌ NO TEMPLATE FOUND for '{card_name}'")
            print(f"   Available templates: {list(detector.templates.keys())[:10]}...")
            continue

        # Use first match (or you can be more specific)
        template_name = matching_templates[0]
        template = detector.templates[template_name]

        print(f"\n📥 TEMPLATE: {template_name}")
        print(f"   Size: {template.shape[1]}x{template.shape[0]}")

        # ===== RESIZE TEMPLATE TO MATCH EXTRACTED =====
        try:
            template_resized = cv2.resize(
                template, (extracted.shape[1], extracted.shape[0])
            )
        except Exception as e:
            print(f"❌ Could not resize template: {e}")
            continue

        # ===== CONVERT TO GRAYSCALE FOR COMPARISON =====
        extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

        # ===== CALCULATE MATCH SCORE =====
        result = cv2.matchTemplate(extracted_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        score = result.max()

        print(f"\n📊 MATCH SCORE: {score:.4f}")
        if score >= 0.60:
            print(f"   ✅ WOULD BE DETECTED (threshold: 0.60)")
        elif score >= 0.55:
            print(f"   ⚠️  BARELY DETECTED (threshold: 0.55)")
        else:
            print(f"   ❌ NOT DETECTED (too low)")

        # ===== SAVE INDIVIDUAL IMAGES =====
        # Save extracted
        extracted_file = output_path / f"{idx+1}_{card_name}_extracted.png"
        cv2.imwrite(str(extracted_file), extracted)
        print(f"\n💾 Saved: {extracted_file.name}")

        # Save template (resized)
        template_file = output_path / f"{idx+1}_{card_name}_template.png"
        cv2.imwrite(str(template_file), template_resized)
        print(f"💾 Saved: {template_file.name}")

        # ===== CREATE SIDE-BY-SIDE COMPARISON =====
        # Make both same size for comparison
        h_max = max(extracted.shape[0], template_resized.shape[0])

        # Pad extracted if needed
        if extracted.shape[0] < h_max:
            padding = h_max - extracted.shape[0]
            extracted_padded = cv2.copyMakeBorder(
                extracted, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            extracted_padded = extracted

        # Pad template if needed
        if template_resized.shape[0] < h_max:
            padding = h_max - template_resized.shape[0]
            template_padded = cv2.copyMakeBorder(
                template_resized, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            template_padded = template_resized

        # Create comparison image (side by side)
        comparison = np.hstack([extracted_padded, template_padded])

        # Add labels
        cv2.putText(
            comparison,
            "EXTRACTED",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            comparison,
            "TEMPLATE",
            (extracted_padded.shape[1] + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            f"Score: {score:.3f}",
            (10, comparison.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Save comparison
        comparison_file = output_path / f"{idx+1}_{card_name}_comparison.png"
        cv2.imwrite(str(comparison_file), comparison)
        print(f"💾 Saved: {comparison_file.name}")

        # Create grayscale comparison
        extracted_gray_padded = (
            cv2.cvtColor(extracted_padded, cv2.COLOR_BGR2GRAY)
            if len(extracted_padded.shape) == 3
            else extracted_padded
        )
        template_gray_padded = (
            cv2.cvtColor(template_padded, cv2.COLOR_BGR2GRAY)
            if len(template_padded.shape) == 3
            else template_padded
        )

        comparison_gray = np.hstack([extracted_gray_padded, template_gray_padded])
        cv2.putText(
            comparison_gray,
            "EXTRACTED",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            comparison_gray,
            "TEMPLATE",
            (extracted_gray_padded.shape[1] + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        # Save grayscale comparison
        comparison_gray_file = output_path / f"{idx+1}_{card_name}_comparison_gray.png"
        cv2.imwrite(str(comparison_gray_file), comparison_gray)
        print(f"💾 Saved: {comparison_gray_file.name}")

    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\n📁 All images saved to: {output_path.absolute()}")
    print("\n📊 Files created for each card:")
    print("   • {N}_{card}_extracted.png - What's extracted from game")
    print("   • {N}_{card}_template.png - What's in templates")
    print("   • {N}_{card}_comparison.png - Side-by-side (color)")
    print("   • {N}_{card}_comparison_gray.png - Side-by-side (grayscale)")
    print("\n💡 Compare extracted vs template to see why matching fails!")
    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 80)
    print("🔍 SLOT vs WAITING_FOR_CARD COMPARISON")
    print("=" * 80)

    # Use check.jpg in tools folder
    screenshot_path = Path(__file__).parent / "check.jpg"

    if not screenshot_path.exists():
        print(f"\n❌ Screenshot not found: {screenshot_path}")
        print("   Please place your screenshot as 'tools/check.jpg'")
        return

    print(f"\n📸 Using screenshot: {screenshot_path}")

    # Compare slot 3 (index 2) with waiting_for_card
    compare_slot_with_template(
        str(screenshot_path),
        slot_idx=2,  # Slot 3 (0-indexed)
        template_name="waiting_for_card",
    )


if __name__ == "__main__":
    main()
