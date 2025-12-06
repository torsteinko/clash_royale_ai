"""
Debug script to check waiting_for_card detection
"""

import cv2
import numpy as np
from pathlib import Path

# Card slot positions (from game_config)
CARD_SLOTS = [
    (169, 1069, 283, 1215),  # Card 1
    (305, 1069, 418, 1215),  # Card 2
    (440, 1069, 553, 1215),  # Card 3
    (575, 1069, 688, 1215),  # Card 4
]

# Cropping settings (from CardDetectorSimple)
CROP_TOP = 0.17  # Remove top border
CROP_BOTTOM = 0.28  # Remove elixir + bottom border
CROP_SIDE = 0.12  # Remove side borders


def crop_to_artwork(slot_roi):
    """Crop slot ROI to artwork area (same as template extraction)"""
    card_height, card_width = slot_roi.shape[:2]
    crop_top = int(card_height * CROP_TOP)
    crop_bottom = int(card_height * (1 - CROP_BOTTOM))
    crop_left = int(card_width * CROP_SIDE)
    crop_right = int(card_width * (1 - CROP_SIDE))
    return slot_roi[crop_top:crop_bottom, crop_left:crop_right]


def main():
    # Load the RAW frame where waiting_for_card should be detected (slot 1 = knight)
    # Frame 45 from the screenshot shows a waiting_for_card on slot 1
    frame_path = Path("recordings/20251203_162744/frame_00045.jpg")

    if not frame_path.exists():
        print(f"Frame not found at {frame_path}")
        return

    frame = cv2.imread(str(frame_path))
    print(f"Loaded frame: {frame_path}")
    print(f"Frame shape: {frame.shape}")

    # Load all waiting templates
    templates_dir = Path("data/card_templates/other")
    waiting_templates = {}
    for i in range(1, 5):
        tpl_path = templates_dir / f"waiting_for_card_{i}.png"
        if tpl_path.exists():
            tpl = cv2.imread(str(tpl_path))
            waiting_templates[i] = tpl
            print(f"Loaded template {i}: {tpl.shape}")
        else:
            print(f"Template {i} NOT FOUND at {tpl_path}")

    # Check each slot
    print("\n=== SLOT ANALYSIS (with artwork cropping) ===")
    for slot_idx, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
        slot_num = slot_idx + 1
        print(f"\n--- Slot {slot_idx} (template waiting_for_card_{slot_num}) ---")

        slot_roi = frame[y1:y2, x1:x2]
        print(f"Full slot ROI shape: {slot_roi.shape}")

        # Crop to artwork area (same as template extraction)
        card_artwork = crop_to_artwork(slot_roi)
        print(f"Cropped artwork shape: {card_artwork.shape}")

        # Save the cropped artwork for visual inspection
        cv2.imwrite(f"slot_{slot_idx}_artwork.png", card_artwork)
        print(f"Saved slot_{slot_idx}_artwork.png")

        # Get the template for this slot
        if slot_num in waiting_templates:
            template = waiting_templates[slot_num]
            print(f"Template shape: {template.shape}")

            # Resize template to match artwork size
            h, w = card_artwork.shape[:2]
            template_resized = cv2.resize(
                template, (w, h), interpolation=cv2.INTER_AREA
            )

            # Save resized template
            cv2.imwrite(f"template_{slot_idx}_resized.png", template_resized)

            # Template matching
            result = cv2.matchTemplate(
                card_artwork, template_resized, cv2.TM_CCOEFF_NORMED
            )
            score = result[0, 0] if result.size > 0 else 0.0
            print(f"Template match score: {score:.4f} (threshold: 0.70)")
            print(f"Would detect as waiting: {score > 0.70}")

            # Cross-check: try matching this slot with ALL templates
            print(f"\n  Cross-checking slot {slot_idx} against all templates:")
            for other_num, other_tpl in waiting_templates.items():
                other_resized = cv2.resize(
                    other_tpl, (w, h), interpolation=cv2.INTER_AREA
                )
                other_result = cv2.matchTemplate(
                    card_artwork, other_resized, cv2.TM_CCOEFF_NORMED
                )
                other_score = other_result[0, 0] if other_result.size > 0 else 0.0
                match_marker = " <-- MATCH!" if other_score > 0.70 else ""
                print(f"    vs template {other_num}: {other_score:.4f}{match_marker}")
        else:
            print(f"No template found for slot {slot_num}")


if __name__ == "__main__":
    main()
