import time
import cv2
import numpy as np
import torch
import subprocess
import win32gui
import win32ui
import win32con
import re
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
from policy.offline.card_list import idx2card as global_idx2card

# --- CONFIGURATION ---
ADB_PATH = "adb"  # If adb is in PATH
CONFIDENCE_THRESHOLD = 0.5
ACTION_DELAY = 1  # Seconds between actions

# --- DECK CONFIGURATION (2.6 Hog Cycle) ---
# Includes 8 base cards + 2 likely evolution slots
MY_DECK = [
    # Base Cards
    "hog-rider",
    "musketeer",
    "cannon",
    "ice-golem",
    "skeletons",
    "ice-spirit",
    "fireball",
    "the-log",
    # Evolution Versions (Add ones you might use)
    "skeletons-evolution",
    "ice-spirit-evolution",
    # "musketeer-evolution",  # Optional, if you use Evo Musk
]


class MEmuInstance:
    def __init__(self, serial, window_handle, title):
        self.serial = serial
        self.hwnd = window_handle
        self.title = title


def get_adb_devices():
    """List all connected ADB devices (emulators)"""
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
        print("❌ Error: ADB not found. Please install ADB or set ADB_PATH.")
        sys.exit(1)


def find_memu_windows():
    """Finds all MEmu windows"""
    windows = []

    def enum_handler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "MEmu" in title:
                windows.append((hwnd, title))

    win32gui.EnumWindows(enum_handler, None)
    return windows


def capture_background_window(hwnd):
    """Captures a specific window in the background using Win32 API"""
    try:
        rect = win32gui.GetWindowRect(hwnd)
        x, y, w, h = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        if w <= 0 or h <= 0:
            return None

        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")
        img.shape = (h, w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img
    except Exception:
        return None


def adb_click(serial, x, y):
    """Sends a background click via ADB"""
    cmd = [ADB_PATH, "-s", serial, "shell", "input", "tap", str(x), str(y)]
    subprocess.run(cmd)


def main():
    print("🚀 Starting Background Controller...")

    # 1. Load Models (Unpacking 5 items now)
    yolo, policy, config, card_classifier, ocr = load_models()
    print("✅ Models & OCR Loaded.")

    # 2. Scan Environment
    adb_devices = get_adb_devices()
    windows = find_memu_windows()

    if not windows:
        print("❌ No MEmu windows found!")
        return
    if not adb_devices:
        print("❌ No ADB devices found! Is MEmu running?")
        return

    # 3. User Selection
    print("\nSelect the WINDOW to capture (visuals):")
    for i, (hwnd, title) in enumerate(windows):
        print(f"  {i}: {title}")
    win_idx = int(input("Enter ID: "))
    selected_hwnd, selected_title = windows[win_idx]

    print("\nSelect the ADB DEVICE to control (clicks):")
    for i, dev in enumerate(adb_devices):
        print(f"  {i}: {dev}")
    adb_idx = int(input("Enter ID: "))
    selected_serial = adb_devices[adb_idx]

    print(f"\n✅ Targeting: '{selected_title}' -> ADB '{selected_serial}'")
    print("running...")

    while True:
        try:
            # --- CAPTURE ---
            img_bgr = capture_background_window(selected_hwnd)
            if img_bgr is None:
                print("⚠️ Window minimized or invalid. Waiting...")
                time.sleep(1)
                continue

            # --- OCR (Elixir) ---
            current_elixir = ocr.read_elixir(img_bgr)
            if current_elixir is None:
                current_elixir = 5
            print(f"💧 Elixir: {current_elixir}")

            # Prepare Image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image

            img_pil = Image.fromarray(img_rgb)

            # --- DETECT ---
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

            # --- HAND (With Deck Filtering) ---
            hand_card_names = card_classifier.detect_hand(
                img_bgr, valid_deck=MY_DECK  # <--- Filter applied here
            )

            # --- TIMER ---
            try:
                big_text_id = YOLO_CLASS_NAMES.index("big-text")
                big_text_boxes = [d for d in detections if d["class"] == big_text_id]
            except (ValueError, NameError):
                big_text_boxes = []

            timer_str = ocr.read_timer(img_bgr, big_text_boxes)

            current_time_sec = None
            if timer_str:
                try:
                    mins, secs = map(int, timer_str.split(":"))
                    current_time_sec = mins * 60 + secs
                    print(f"⏱️ Timer: {timer_str} ({current_time_sec}s)")
                except:
                    pass

            # --- PREDICT ---
            state = build_state(
                detections,
                hand_card_names,
                config,
                elixir=current_elixir,
                time_sec=current_time_sec,
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

            # --- LOGIC ---
            hand_ids = state[2:6].astype(int).tolist()
            masked_logits = torch.full_like(c_logits[0, 0], float("-inf"))
            valid_moves = []
            for i, cid in enumerate(hand_ids):
                if cid != 0:
                    masked_logits[cid] = c_logits[0, 0, cid]
                    valid_moves.append(cid)

            if not valid_moves:
                continue

            best_card_id = torch.argmax(masked_logits).item()
            best_pos_idx = torch.argmax(p_logits[0, 0]).item()

            grid_x = best_pos_idx % config.arena_grid_size[1]
            grid_y = best_pos_idx // config.arena_grid_size[1]

            # --- EXECUTE ---
            print(
                f"🤖 Play {global_idx2card.get(best_card_id, best_card_id)} at ({grid_x}, {grid_y})"
            )

            slot_index = -1
            for i, cid in enumerate(hand_ids):
                if cid == best_card_id:
                    slot_index = i
                    break

            if slot_index != -1:
                W, H = img_pil.size

                # Slots: 0, 1, 2, 3
                card_x_ratios = [0.35, 0.52, 0.69, 0.86]
                card_y_ratio = 0.91

                card_px = int(W * card_x_ratios[slot_index])
                card_py = int(H * card_y_ratio)

                # Arena
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
