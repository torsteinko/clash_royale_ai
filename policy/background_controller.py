import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import subprocess
try:  # win32 window capture is only available on Windows (MEmu workflow)
    import win32gui
    import win32ui
    import win32con

    _HAVE_WIN32 = True
except ImportError:  # Linux/headless -> use the ADB screencap backend below
    _HAVE_WIN32 = False
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.ocr_reader import OCRReader

from policy.offline.visualize_prediction import (
    YOLO_CLASS_NAMES,
    load_models,
    build_state,
    idx2card,
)

from policy.offline.card_list import idx2card as global_idx2card, card2elixir

# --- CONFIGURATION ---
ADB_PATH = "adb"  # If adb is in PATH
CONFIDENCE_THRESHOLD = 0.5
ACTION_DELAY = 1  # Seconds between actions

# --- DECK CONFIGURATION (2.6 Hog Cycle) ---
MY_DECK = [
    "hog_rider",
    "musketeer",
    "cannon",
    "ice_golem",
    "skeleton",
    "ice_spirit",
    "fireball",
    "the_log",
    "skeleton_evo",
    "ice_spirit_evo",
]


# --- LOCAL CLASS DEFINITION TO ENSURE UPDATES ARE USED ---
class CardClassifier:
    def __init__(self, model_path="models/cards_cls.pt", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, img_pil, valid_cards=None):
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_t)
            if valid_cards:
                mask = torch.full_like(logits, float("-inf"))
                found_any = False
                for card_name in valid_cards:
                    if card_name in self.classes:
                        idx = self.classes.index(card_name)
                        mask[0, idx] = logits[0, idx]
                        found_any = True
                if found_any:
                    logits = mask
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.classes[pred_idx]

    def detect_hand(self, img_cv2, valid_deck=None, save_crops=False):
        """
        Detect 4 cards in hand with UNIQUE CONSTRAINT.
        Prevents duplicate cards or base+evo of same card in hand.
        """
        h, w = img_cv2.shape[:2]
        if save_crops:
            debug_dir = Path("debug_hand")
            debug_dir.mkdir(exist_ok=True)

        # --- CROP COORDINATES (Adjusted as per previous turn) ---
        W_PERC = 87 / 720
        H_PERC = 81 / 1280
        CY_PERC = 0.89  # Adjusted down

        CX_START = 232  # Adjusted right
        CX_STEP = 136

        cx_pixels = [CX_START + i * CX_STEP for i in range(4)]

        crops = []
        crop_images = []  # For saving debugs later

        for i, center_x in enumerate(cx_pixels):
            cx = center_x / 720

            x1 = int((cx - W_PERC / 2) * w)
            y1 = int((CY_PERC - H_PERC / 2) * h)
            x2 = int((cx + W_PERC / 2) * w)
            y2 = int((CY_PERC + H_PERC / 2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            card_crop = img_cv2[y1:y2, x1:x2]
            crop_images.append(card_crop)

            if card_crop.size == 0:
                crops.append(None)
                continue

            card_pil = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))
            crops.append(card_pil)

        # --- BATCH PREDICTION WITH LOGIC ---
        # 1. Get raw logits for all 4 slots
        slot_logits = []
        for i, img_pil in enumerate(crops):
            if img_pil is None:
                slot_logits.append(None)
                continue

            img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(img_t)[0]  # Shape (num_classes,)

                # Apply Valid Deck Mask immediately
                if valid_deck:
                    mask = torch.full_like(logits, float("-inf"))
                    for card_name in valid_deck:
                        if card_name in self.classes:
                            idx = self.classes.index(card_name)
                            mask[idx] = logits[idx]
                    logits = mask

            slot_logits.append(logits)

        # 2. Greedy Assignment (Highest confidence first)
        final_hand = ["unknown"] * 4

        # We need a list of (confidence, slot_idx, class_idx) tuples
        candidates = []
        for i, logits in enumerate(slot_logits):
            if logits is None:
                continue

            # Get softmax probabilities to compare confidence across slots
            probs = torch.softmax(logits, dim=0)

            # Add all possible cards for this slot to candidates
            for class_idx, score in enumerate(probs):
                if score > 0.01:  # Optimization: ignore impossible cards
                    candidates.append(
                        {
                            "score": score.item(),
                            "slot": i,
                            "class_idx": class_idx,
                            "name": self.classes[class_idx],
                        }
                    )

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        assigned_slots = set()
        assigned_base_names = (
            set()
        )  # To track "hog_rider", "skeletons" (stripping _evo)

        for cand in candidates:
            slot = cand["slot"]
            name = cand["name"]

            if slot in assigned_slots:
                continue

            # Normalize name to base (remove _evo, -evolution)
            base_name = (
                name.replace("_evo", "")
                .replace("-evolution", "")
                .replace("_evolution", "")
            )

            # CONSTRAINT: Cannot have same base card twice
            if base_name in assigned_base_names:
                continue

            # Assign
            final_hand[slot] = name
            assigned_slots.add(slot)
            assigned_base_names.add(base_name)

            if len(assigned_slots) == 4:
                break

        # Fill remaining slots (if any failed) with raw max (should rarely happen)
        for i in range(4):
            if final_hand[i] == "unknown" and slot_logits[i] is not None:
                # Fallback: Just take max, ignoring constraints
                best_idx = torch.argmax(slot_logits[i]).item()
                final_hand[i] = self.classes[best_idx]

        # --- DEBUG SAVING ---
        if save_crops:
            timestamp = int(time.time() * 1000)
            for i, crop in enumerate(crop_images):
                if crop is not None and crop.size > 0:
                    debug_crop = crop.copy()
                    name = final_hand[i]
                    cv2.putText(
                        debug_crop,
                        name,
                        (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )
                    fname = debug_dir / f"slot{i}_{name}_{timestamp}.jpg"
                    cv2.imwrite(str(fname), debug_crop)

        return final_hand


class MEmuInstance:
    def __init__(self, serial, window_handle, title):
        self.serial = serial
        self.hwnd = window_handle
        self.title = title


def get_adb_devices():
    try:
        result = subprocess.run(
            [ADB_PATH, "devices"], capture_output=True, text=True
        ).stdout
        devices = []
        for line in result.splitlines()[1:]:
            if "\tdevice" in line:
                devices.append(line.split("\t")[0])
        return devices
    except FileNotFoundError:
        print("❌ Error: ADB not found.")
        sys.exit(1)


def find_memu_windows():
    windows = []
    if not _HAVE_WIN32:
        return windows

    def enum_handler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "MEmu" in title:
                windows.append((hwnd, title))

    win32gui.EnumWindows(enum_handler, None)
    return windows


def capture_background_window(hwnd):
    if not _HAVE_WIN32:
        return None
    try:
        rect = win32gui.GetWindowRect(hwnd)
        w_total = rect[2] - rect[0]
        h_total = rect[3] - rect[1]
        if w_total <= 0 or h_total <= 0:
            return None

        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w_total, h_total)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w_total, h_total), dcObj, (0, 0), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")
        img.shape = (h_total, w_total, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)

        # --- CROP MEMU BORDERS ---
        crop_top = 35
        crop_right = 40
        if h_total > crop_top and w_total > crop_right:
            img = img[crop_top:, : w_total - crop_right]
        return img
    except Exception:
        return None


def adb_click(serial, x, y):
    cmd = [ADB_PATH, "-s", serial, "shell", "input", "tap", str(x), str(y)]
    subprocess.run(cmd)


def save_debug_action(
    img_bgr, card_name, grid_x, grid_y, elixir, detections, hand_cards, reason="play"
):
    folder = Path("debug_actions")
    folder.mkdir(exist_ok=True)
    debug_img = img_bgr.copy()
    H, W = debug_img.shape[:2]

    # YOLO
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cls_id = d["class"]
        try:
            label_name = YOLO_CLASS_NAMES[cls_id]
        except:
            label_name = str(cls_id)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug_img,
            label_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Elixir ROI
    e_x1, e_y1 = int(W * 0.20), int(H * 0.93)
    e_x2, e_y2 = int(W * 0.35), int(H * 0.99)
    cv2.rectangle(debug_img, (e_x1, e_y1), (e_x2, e_y2), (255, 0, 255), 3)
    cv2.putText(
        debug_img,
        f"OCR: {elixir}",
        (e_x1, e_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    # Hand
    card_x_ratios = [0.35, 0.52, 0.69, 0.86]
    for i, c_name in enumerate(hand_cards):
        cx = int(W * card_x_ratios[i])
        cy = int(H * 0.96)
        (tw, th), _ = cv2.getTextSize(c_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(
            debug_img,
            c_name,
            (cx - tw // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    # Action
    if reason == "play":
        ARENA_X_MIN, ARENA_X_MAX = W * 0.069, W * 0.920
        ARENA_Y_MIN, ARENA_Y_MAX = H * 0.085, H * 0.766
        GRID_W, GRID_H = 18, 32
        px = int(ARENA_X_MIN + (grid_x / GRID_W) * (ARENA_X_MAX - ARENA_X_MIN))
        py = int(ARENA_Y_MIN + (grid_y / GRID_H) * (ARENA_Y_MAX - ARENA_Y_MIN))
        cv2.circle(debug_img, (px, py), 20, (0, 0, 255), 3)
        cv2.putText(
            debug_img,
            f"PLAY: {card_name}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            debug_img,
            f"DEBUG: {reason}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    timestamp = int(time.time() * 1000)
    cv2.imwrite(str(folder / f"{reason}_{timestamp}_{card_name}.jpg"), debug_img)


def is_inside_box(center_point, box):
    cx, cy = center_point
    bx1, by1, bx2, by2 = box
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def filter_detections(detections, img_w, img_h):
    filtered = []
    # Bush Zones (Goblinstein)
    ref_w, ref_h = 761, 1313
    scale_x, scale_y = img_w / ref_w, img_h / ref_h
    bush_zones = [
        [26, 497, 93, 561],
        [34, 582, 96, 630],
        [620, 494, 676, 554],
        [624, 579, 680, 633],
    ]
    scaled_zones = [
        [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
        for b in bush_zones
    ]
    tower_boxes = [
        d["bbox"]
        for d in detections
        if "tower" in YOLO_CLASS_NAMES[d["class"]]
        and "bar" not in YOLO_CLASS_NAMES[d["class"]]
    ]

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        try:
            name = YOLO_CLASS_NAMES[d["class"]]
        except:
            name = str(d["class"])

        # Kill Zones
        if any(is_inside_box((cx, cy), z) for z in scaled_zones):
            continue
        # Orphan Tower Bars
        if "tower-bar" in name.lower() or "tower_bar" in name.lower():
            has_parent = False
            for tx1, ty1, tx2, ty2 in tower_boxes:
                tcx = (tx1 + tx2) / 2
                if (
                    abs(cx - tcx) < ((tx2 - tx1) * 0.8)
                    and (ty1 - 100 * scale_y) < cy < ty2
                ):
                    has_parent = True
                    break
            if not has_parent:
                continue
        # UI Mask (Hand Area)
        if cy > (img_h * 0.80):
            continue

        filtered.append(d)
    return filtered


def main():
    print("🚀 Starting Background Controller...")
    # 1. Load Models (Ignore old classifier return)
    yolo, policy, config, _, ocr = load_models()

    # 2. Init LOCAL Classifier
    print("📦 Loading Local Card Classifier...")
    card_classifier = CardClassifier("models/cards_cls.pt")

    print("✅ Models & OCR Loaded.")
    print("Classifier has these deck cards:")
    print([c for c in MY_DECK if c in card_classifier.classes])

    print("Example of classifier classes:", card_classifier.classes[:30])

    adb_devices = get_adb_devices()
    windows = find_memu_windows()

    if not adb_devices:
        print("❌ No ADB devices found (start the emulator and enable ADB)!")
        return

    if windows:
        print("\nSelect the WINDOW to capture (visuals):")
        for i, (hwnd, title) in enumerate(windows):
            print(f"  {i}: {title}")
        selected_hwnd, selected_title = windows[int(input("Enter ID: "))]
        use_window_capture = True
    else:
        # No MEmu window (headless/Linux) -> capture straight from the device.
        selected_hwnd, selected_title = None, "adb-screencap"
        use_window_capture = False
        print("\nℹ️  No MEmu windows found -- using ADB screencap capture.")

    print("\nSelect the ADB DEVICE to control (clicks):")
    for i, dev in enumerate(adb_devices):
        print(f"  {i}: {dev}")
    selected_serial = adb_devices[int(input("Enter ID: "))]

    print(f"\n✅ Targeting: '{selected_title}' -> ADB '{selected_serial}'")
    print("running...")

    last_zero_elixir_debug = 0

    while True:
        try:
            if use_window_capture:
                img_bgr = capture_background_window(selected_hwnd)
            else:
                from detection.adb_capture import capture_adb

                img_bgr = capture_adb(selected_serial)
            if img_bgr is None:
                time.sleep(1)
                continue

            current_elixir = ocr.read_elixir(img_bgr)
            if current_elixir is None:
                current_elixir = 5
            print(f"💧 Elixir: {current_elixir}")

            # THIS will now run the local class method with save_crops
            hand_card_names = card_classifier.detect_hand(
                img_bgr, valid_deck=MY_DECK, save_crops=True
            )

            # if current_elixir == 0:
            #     if time.time() - last_zero_elixir_debug > 5:
            #         save_debug_action(
            #             img_bgr,
            #             "none",
            #             0,
            #             0,
            #             0,
            #             [],
            #             hand_card_names,
            #             reason="zero_elixir",
            #         )
            #         last_zero_elixir_debug = time.time()

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            results = yolo(img_pil, conf=0.5, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append(
                        {
                            "class": int(box.cls),
                            "bbox": box.xyxy[0].tolist(),
                            "conf": float(box.conf),
                            "img_width": img_pil.width,
                            "img_height": img_pil.height,
                        }
                    )

            detections = filter_detections(detections, img_pil.width, img_pil.height)
            state = build_state(
                detections, hand_card_names, config, elixir=current_elixir, time_sec=100
            )
            state_t = (
                torch.from_numpy(state)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(next(policy.parameters()).device)
            )

            with torch.no_grad():
                c_logits, p_logits = policy(
                    state_t,
                    torch.zeros(1, 1, 3).to(state_t.device),
                    torch.tensor([[10.0]]).to(state_t.device),
                    torch.tensor([[0]]).to(state_t.device),
                )

            hand_ids = state[2:6].astype(int).tolist()
            masked_logits = torch.full_like(c_logits[0, 0], float("-inf"))
            valid_moves = []
            for i, cid in enumerate(hand_ids):
                if cid == 0:
                    continue
                card_name = global_idx2card.get(cid, "unknown")
                cost = card2elixir.get(card_name, 10)
                if current_elixir >= cost:
                    masked_logits[cid] = c_logits[0, 0, cid]
                    valid_moves.append(cid)

            if not valid_moves:
                continue

            best_card_id = torch.argmax(masked_logits).item()
            best_pos_idx = torch.argmax(p_logits[0, 0]).item()
            grid_x, grid_y = (
                best_pos_idx % config.arena_grid_size[1],
                best_pos_idx // config.arena_grid_size[1],
            )
            best_card_name = global_idx2card.get(best_card_id, "unknown")

            save_debug_action(
                img_bgr,
                best_card_name,
                grid_x,
                grid_y,
                current_elixir,
                detections,
                hand_cards=hand_card_names,
            )
            print(f"🤖 Play {best_card_name} at ({grid_x}, {grid_y})")

            slot_index = -1
            for i, cid in enumerate(hand_ids):
                if cid == best_card_id:
                    slot_index = i
                    break

            if slot_index != -1:
                W, H = img_pil.size
                card_x_ratios = [0.35, 0.52, 0.69, 0.86]
                card_px = int(W * card_x_ratios[slot_index])
                card_py = int(H * 0.91)
                ARENA_X_MIN, ARENA_X_MAX = W * 0.069, W * 0.920
                ARENA_Y_MIN, ARENA_Y_MAX = H * 0.085, H * 0.766
                GRID_W, GRID_H = 18, 32
                arena_px = int(
                    ARENA_X_MIN + (grid_x / GRID_W) * (ARENA_X_MAX - ARENA_X_MIN)
                )
                arena_py = int(
                    ARENA_Y_MIN + (grid_y / GRID_H) * (ARENA_Y_MAX - ARENA_Y_MIN)
                )

                adb_click(selected_serial, card_px, card_py)
                time.sleep(0.1)
                adb_click(selected_serial, arena_px, arena_py)
                time.sleep(ACTION_DELAY)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
