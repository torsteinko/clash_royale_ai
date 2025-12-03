# tests/test_live_detection.py
"""Test detection on live Clash Royale gameplay - Multi-window support"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import mss
import time
import win32gui
import win32ui
import win32con
from game_state.state_extractor import GameStateExtractor


def list_memu_windows():
    """Find all MEmu windows"""
    
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if 'MEmu' in title or 'Clash' in title:
                windows.append((hwnd, title))
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows


def capture_window(hwnd):
    """Capture specific window by handle"""
    
    try:
        # Get window dimensions
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        
        # Get window device context
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        # Create bitmap
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # Copy window to bitmap
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
        
        # Convert to numpy array
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img = img.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
        
        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    except Exception as e:
        print(f"❌ Error capturing window: {e}")
        return None


def capture_memu_screen_simple():
    """Simple screen capture (fallback)"""
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def main():
    print("\n" + "="*80)
    print("🎮 LIVE DETECTION TEST - NULLS ROYALE (MEmu)")
    print("="*80)
    
    # Find MEmu windows
    print("\n🔍 Searching for MEmu windows...")
    memu_windows = list_memu_windows()
    
    if not memu_windows:
        print("⚠️  No MEmu windows found!")
        print("   Using full screen capture instead...")
        use_window_capture = False
        selected_hwnd = None
    else:
        print(f"\n✅ Found {len(memu_windows)} window(s):\n")
        for idx, (hwnd, title) in enumerate(memu_windows):
            print(f"   [{idx+1}] {title}")
        
        if len(memu_windows) == 1:
            selected_hwnd = memu_windows[0][0]
            use_window_capture = True
            print(f"\n✅ Auto-selected: {memu_windows[0][1]}")
        else:
            choice = input(f"\nSelect window (1-{len(memu_windows)}): ")
            try:
                selected_hwnd = memu_windows[int(choice)-1][0]
                use_window_capture = True
                print(f"✅ Selected: {memu_windows[int(choice)-1][1]}")
            except:
                print("⚠️  Invalid selection, using full screen")
                use_window_capture = False
                selected_hwnd = None
    
    print("\n" + "="*80)
    print("Instructions:")
    print("  1. Start a match in Nulls Royale")
    print("  2. Press Enter to begin detection...")
    print("  3. Press 'q' to quit")
    print("="*80)
    input()
    
    # Initialize detector
    extractor = GameStateExtractor()
    
    print("\n🔍 Starting detection...\n")
    
    frame_count = 0
    start_time = time.time()
    
    # Disable OpenCV window (causes issues)
    show_window = False  # Set to True if OpenCV GUI works
    
    try:
        while True:
            # Capture screen
            if use_window_capture and selected_hwnd:
                frame = capture_window(selected_hwnd)
                if frame is None:
                    frame = capture_memu_screen_simple()
            else:
                frame = capture_memu_screen_simple()
            
            # Extract state
            state = extractor.extract_state(frame)
            
            # Visualize
            vis_frame = extractor.visualize_state(frame, state)
            
            # Resize for display
            display_frame = cv2.resize(vis_frame, (1280, 720))
            
            # Save periodic screenshots instead of showing window
            if frame_count % 60 == 0 and frame_count > 0:
                cv2.imwrite(f'detection_frame_{frame_count}.jpg', display_frame)
            
            # Show window (if GUI works)
            if show_window:
                try:
                    cv2.imshow('Clash Royale Bot - Detection', display_frame)
                except:
                    show_window = False
                    print("⚠️  OpenCV window disabled (GUI not available)")
            
            # Print state every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                
                print(f"\n{'='*70}")
                print(f"Frame: {frame_count} | FPS: {fps:.1f}")
                print(f"{'='*70}")
                print(f"⚡ Elixir: {state['elixir']}")
                print(f"⏱️  Timer: {state['match_time']}s")
                print(f"🎴 Hand: {state['cards_in_hand']}")
                print(f"🟢 Ally troops: {state['troops']['total_ally']}")
                print(f"🔴 Enemy troops: {state['troops']['total_enemy']}")
            
            # Check for quit
            if show_window:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Without window, use keyboard input (non-blocking)
                import msvcrt
                if msvcrt.kbhit() and msvcrt.getch() == b'q':
                    break
                time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopped by user")
    
    finally:
        if show_window:
            cv2.destroyAllWindows()
        print("\n✅ Detection test complete!")
        print(f"📸 Screenshots saved: detection_frame_*.jpg")


if __name__ == '__main__':
    main()
