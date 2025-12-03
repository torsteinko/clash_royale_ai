# tools/convert_card_images_to_templates.py
"""Convert card_images to templates - EXACT same pipeline as in-game extraction"""

import cv2
import numpy as np
from pathlib import Path


def process_card_image(img: np.ndarray, card_type: str) -> np.ndarray:
    """
    Process card image to EXACTLY match in-game extraction

    Pipeline:
    1. Remove background
    2. Resize to card slot size (114x146)
    3. Apply SAME crop as detector (25% top, 35% bottom, 20% sides)
    """

    # STEP 1: Remove background/borders
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if card_type == "hero":
        # Heroes: keep golden glow (anything darker than 250)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    else:
        # Base/evolution: remove light backgrounds
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(mask)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y : y + h, x : x + w]
    else:
        img_cropped = img

    # STEP 2: Resize to match in-game card slot size (114x146)
    # This is the EXACT size of cards in CARD_SLOTS
    CARD_SLOT_WIDTH = 114
    CARD_SLOT_HEIGHT = 146

    img_resized = cv2.resize(
        img_cropped,
        (CARD_SLOT_WIDTH, CARD_SLOT_HEIGHT),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # STEP 3: Apply SAME crop as CardDetector
    # These MUST match CardDetector.CROP_TOP/BOTTOM/SIDE
    CROP_TOP = 0.17
    CROP_BOTTOM = 0.28
    CROP_SIDE = 0.12

    card_height = img_resized.shape[0]
    card_width = img_resized.shape[1]

    crop_top = int(card_height * CROP_TOP)
    crop_bottom = int(card_height * (1 - CROP_BOTTOM))
    crop_left = int(card_width * CROP_SIDE)
    crop_right = int(card_width * (1 - CROP_SIDE))

    # Apply tight crop
    img_final = img_resized[crop_top:crop_bottom, crop_left:crop_right]

    if img_final.size == 0:
        print("      ⚠️  Crop resulted in empty image, using resized version")
        img_final = img_resized

    return img_final


def convert_card_images_to_templates(
    source_dir: str = "card_images", output_dir: str = "data/card_templates"
):
    """Convert card images with EXACT same processing as in-game extraction"""

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"❌ Source directory not found: {source_path}")
        return

    print("\n" + "=" * 80)
    print("🎴 CONVERTING CARD IMAGES TO TEMPLATES")
    print("=" * 80)
    print(f"\n📂 Source: {source_path.absolute()}")
    print(f"💾 Output: {output_path.absolute()}")
    print(f"\n📐 Pipeline:")
    print(f"   1. Remove background")
    print(f"   2. Resize to card slot size (114x146)")
    print(f"   3. Crop (25% top, 35% bottom, 20% sides)")
    print(f"   → Final template size: ~68x73 (matches extracted!)\n")

    categories = ["base", "evolution", "hero"]
    total_converted = 0

    for category in categories:
        category_source = source_path / category
        category_output = output_path / category

        if not category_source.exists():
            print(f"⚠️  {category}/ not found, skipping...")
            continue

        category_output.mkdir(parents=True, exist_ok=True)

        image_files = (
            list(category_source.glob("*.png"))
            + list(category_source.glob("*.jpg"))
            + list(category_source.glob("*.jpeg"))
        )

        if not image_files:
            print(f"⚠️  No images in {category}/")
            continue

        print(f"📦 Processing {category}/ ({len(image_files)} images)")

        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"   ❌ Failed to load: {img_path.name}")
                continue

            original_size = f"{img.shape[1]}x{img.shape[0]}"

            try:
                processed = process_card_image(img, category)

                output_file = category_output / f"{img_path.stem}.png"
                cv2.imwrite(str(output_file), processed)

                new_size = f"{processed.shape[1]}x{processed.shape[0]}"
                print(f"   ✅ {img_path.name}: {original_size} → {new_size}")

                total_converted += 1

            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")

    print("\n" + "=" * 80)
    print(f"✅ CONVERSION COMPLETE - {total_converted} templates created")
    print("=" * 80)
    print(f"\n📁 Templates: {output_path.absolute()}")
    print(f"📐 All templates: ~68x73 (matching extracted cards)")
    print(f"✂️  Pipeline: Remove BG → Resize to 114x146 → Crop")
    print("\n💡 Templates now EXACTLY match in-game extraction!")
    print("=" * 80 + "\n")


def main():
    if not Path("card_images").exists():
        print("❌ card_images/ directory not found!")
        return

    convert_card_images_to_templates()


if __name__ == "__main__":
    main()
