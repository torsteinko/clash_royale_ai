# tests/record_gameplay.py
"""Record gameplay screenshots for later debugging"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import win32gui
import win32ui
import win32con
import numpy as np
from datetime import datetime


def list_windows():
    """Find all visible windows"""

    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and len(title) > 0:
                windows.append((hwnd, title))
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows


def capture_window(hwnd):
    """Capture MEmu window"""

    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img = img.reshape((height, width, 4))

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Crop MEmu borders
        crop_top = 32
        crop_right = 40

        if height > crop_top and width > crop_right:
            img = img[crop_top:, : width - crop_right]

        return img

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def select_window():
    """Select MEmu window"""

    print("\n🔍 Searching for windows...")
    windows = list_windows()

    memu_windows = [
        (hwnd, title)
        for hwnd, title in windows
        if "MEmu" in title or "Clash" in title or "Royale" in title
    ]

    if not memu_windows:
        print("⚠️  No MEmu windows found!")
        return None, "None"

    print(f"\n✅ Found {len(memu_windows)} window(s):\n")
    for idx, (hwnd, title) in enumerate(memu_windows, 1):
        print(f"   [{idx}] {title}")

    if len(memu_windows) == 1:
        print(f"\n✅ Auto-selected: {memu_windows[0][1]}")
        return memu_windows[0]

    choice = input(f"\nSelect window (1-{len(memu_windows)}): ")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(memu_windows):
            return memu_windows[idx]
    except:
        pass

    return memu_windows[0]


def main():
    print("\n" + "=" * 80)
    print("📹 GAMEPLAY RECORDER - Save screenshots for debugging")
    print("=" * 80)

    # Select window
    hwnd, window_title = select_window()

    if hwnd is None:
        print("❌ No window selected!")
        return

    # Create recording directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_dir = Path(__file__).parent.parent / "recordings" / timestamp
    recording_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📋 Instructions:")
    print("  1. Start a match in Clash Royale")
    print("  2. Press Enter to begin recording")
    print("  3. Play normally")
    print("  4. Press Ctrl+C to stop recording")
    print(f"\n🖼️  Capturing: {window_title}")
    print(f"💾 Saving to: {recording_dir}")
    print("=" * 80)
    input("\nPress Enter to start recording...")

    frame_count = 0
    start_time = time.time()

    print(f"\n🔴 RECORDING STARTED - Press Ctrl+C to stop\n")

    try:
        while True:
            # Capture frame
            frame = capture_window(hwnd)

            if frame is None:
                print("⚠️  Failed to capture")
                time.sleep(0.1)
                continue

            # Save frame
            frame_count += 1
            filename = recording_dir / f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(str(filename), frame)

            # Print progress every 30 frames (1 second)
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(
                    f"📸 Recorded {frame_count} frames | {fps:.1f} FPS | {filename.name}"
                )

            # 30 FPS
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("✅ RECORDING STOPPED")
        print("=" * 80)
        elapsed = time.time() - start_time
        print(f"\n📊 Statistics:")
        print(f"   • Total frames: {frame_count}")
        print(f"   • Duration: {elapsed:.1f}s")
        print(f"   • Average FPS: {frame_count/elapsed:.1f}")
        print(f"   • Location: {recording_dir.absolute()}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
