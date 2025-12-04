import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.game_config import CARD_SLOTS
from detection.card_detector_simple import CardDetectorSimple


def prompt(msg, options=None):
    while True:
        val = input(msg)
        if options is None or val in options:
            return val
        print(f"Invalid input. Options: {options}")


def main():
    recordings = Path("recordings")
    subfolders = [f for f in recordings.iterdir() if f.is_dir()]
    if not subfolders:
        print("No recordings found.")
        return
    print("Available recordings:")
    for i, folder in enumerate(subfolders):
        print(f"  {i}: {folder.name}")
    idx = int(
        prompt(
            "Select recordings subfolder index: ",
            [str(i) for i in range(len(subfolders))],
        )
    )
    chosen_folder = subfolders[idx]

    frames = sorted(chosen_folder.glob("frame_*.jpg"))
    if not frames:
        print("No frames found in this folder.")
        return
    print(
        f"Found {len(frames)} frames. Example: {frames[0].name} ... {frames[-1].name}"
    )
    frame_name = prompt(
        "Enter frame filename (e.g. frame_00203.jpg): ", [f.name for f in frames]
    )
    frame_path = chosen_folder / frame_name

    print("Card slots:")
    for i, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
        print(f"  {i}: ({x1},{y1})-({x2},{y2})")
    slot_idx = int(
        prompt(
            f"Select slot index (0-{len(CARD_SLOTS)-1}): ",
            [str(i) for i in range(len(CARD_SLOTS))],
        )
    )

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Error: Could not load frame {frame_path}")
        return
    x1, y1, x2, y2 = CARD_SLOTS[slot_idx]
    card_region = frame[y1:y2, x1:x2]
    detector = CardDetectorSimple()
    card_height, card_width = card_region.shape[:2]
    crop_top = int(card_height * detector.CROP_TOP)
    crop_bottom = int(card_height * (1 - detector.CROP_BOTTOM))
    crop_left = int(card_width * detector.CROP_SIDE)
    crop_right = int(card_width * (1 - detector.CROP_SIDE))
    card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]
    if card_artwork.size == 0:
        print("Error: Cropped card is empty")
        return
    out_name = f"card_template_{frame_name.replace('.jpg','')}_slot{slot_idx}.png"
    out_path = Path(__file__).parent / out_name
    cv2.imwrite(str(out_path), card_artwork)
    print(f"Saved: {out_path} ({card_artwork.shape[1]}x{card_artwork.shape[0]})")


if __name__ == "__main__":
    main()
