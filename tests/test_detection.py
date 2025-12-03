# tests/test_detection.py
"""Detection test with MEmu window selection and portrait mode support"""

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
    """Capture MEmu window and crop to game area only"""
    
    try:
        # Get window rect
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        
        # Get device context
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        # Create bitmap
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # Copy window
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
        
        # Convert to numpy
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img = img.reshape((height, width, 4))
        
        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Crop MEmu borders (adjust these for your MEmu)
        crop_top = 32      # Title bar
        crop_right = 40    # Right menu
        
        if height > crop_top and width > crop_right:
            img = img[crop_top:, :width-crop_right]
        
        return img
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def select_window():
    """Select MEmu window"""
    
    print("\n🔍 Searching for windows...")
    windows = list_windows()
    
    memu_windows = [(hwnd, title) for hwnd, title in windows 
                    if 'MEmu' in title or 'Clash' in title or 'Royale' in title]
    
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
    print("\n" + "="*80)
    print("🎮 CLASH ROYALE BOT - DETECTION TEST (PORTRAIT MODE)")
    print("="*80)
    
    # Select window
    hwnd, window_title = select_window()
    
    if hwnd is None:
        print("❌ No window selected!")
        return
    
    print("\n" + "="*80)
    print("📋 Instructions:")
    print("  1. Make sure Clash Royale is in PORTRAIT mode in MEmu")
    print("  2. Start a match")
    print("  3. Press Enter to begin")
    print("  4. Press Ctrl+C to stop")
    print(f"\n🖼️  Capturing: {window_title}")
    print("💾 Screenshots: screenshots/")
    print("="*80)
    input("\nPress Enter to continue...")
    
    # Initialize
    extractor = GameStateExtractor()
    
    screenshots_dir = Path(__file__).parent.parent / 'screenshots'
    screenshots_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Screenshots: {screenshots_dir.absolute()}\n")
    print("🔍 Detection started! Press Ctrl+C to stop\n")
    
    frame_count = 0
    start_time = time.time()
    last_save = 0
    
    try:
        while True:
            # Capture
            frame = capture_window(hwnd)
            
            if frame is None:
                print("⚠️  Failed to capture window")
                time.sleep(1)
                continue
            
            # Check orientation
            h, w = frame.shape[:2]
            if h < w:
                print(f"⚠️  Warning: Frame is landscape ({w}x{h}), expected portrait!")
            
            # Extract state
            state = extractor.extract_state(frame, debug=True)  # Enable debug mode

            # And use visualize_state to see debug boxes:
            vis_frame = extractor.visualize_state(frame, state)

            
            # Save screenshot every 2 seconds
            frame_count += 1
            current_time = time.time()
            if current_time - last_save >= 2.0:
                filename = screenshots_dir / f'frame_{frame_count:05d}.jpg'
                cv2.imwrite(str(filename), vis_frame)
                print(f"📸 Saved: {filename.name} ({vis_frame.shape[1]}x{vis_frame.shape[0]})")
                last_save = current_time
            
            # Print state every 1 second
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                
                print(f"\n{'='*80}")
                print(f"Frame {frame_count:05d} | FPS: {fps:.1f} | Size: {w}x{h}")
                print(f"{'='*80}")
                print(f"⚡ Elixir:       {state['elixir']}")
                print(f"⏱️  Match Time:   {state['match_time']}s")
                print(f"🎴 Cards:        {', '.join(state['cards_in_hand'])}")
                print(f"🟢 Ally Troops:  {state['troops']['total_ally']}")
                print(f"🔴 Enemy Troops: {state['troops']['total_enemy']}")
                
                # Tower info
                towers = state.get('towers', {})
                if towers:
                    print(f"🏰 Towers: Ally {towers.get('ally_towers_alive', 0)}/3 | Enemy {towers.get('enemy_towers_alive', 0)}/3")
                    print(f"👑 Crowns: Ally {towers.get('ally_crowns', 0)} | Enemy {towers.get('enemy_crowns', 0)}")
                    
                    # Show identified towers
                    for tower_name, tower_state in towers.get('towers', {}).items():
                        if tower_state.get('identified'):
                            hp = tower_state.get('hp', 0)
                            max_hp = tower_state.get('max_hp', 1)
                            hp_pct = (hp / max_hp * 100) if max_hp > 0 else 0
                            
                            # Status emoji based on HP
                            if hp_pct > 66:
                                status = "💚"
                            elif hp_pct > 33:
                                status = "💛"
                            else:
                                status = "❤️"
                            
                            tower_type = tower_state.get('type', 'unknown')
                            print(f"   {status} {tower_name}: {tower_type} - {hp}/{max_hp} ({hp_pct:.0f}%)")
                
                # Show detected troops
                if state['troops']['ally']:
                    ally_types = [t['type'] for t in state['troops']['ally']]
                    print(f"   → Ally troops:  {', '.join(set(ally_types))}")
                
                if state['troops']['enemy']:
                    enemy_types = [t['type'] for t in state['troops']['enemy']]
                    print(f"   → Enemy troops: {', '.join(set(enemy_types))}")
            
            time.sleep(0.033)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("✅ DETECTION TEST STOPPED")
        print("="*80)
        elapsed = time.time() - start_time
        print(f"\n📊 Statistics:")
        print(f"   • Total frames: {frame_count}")
        print(f"   • Duration: {elapsed:.1f}s")
        print(f"   • Average FPS: {frame_count/elapsed:.1f}")
        print(f"   • Window: {window_title}")
        print(f"\n📁 Screenshots: {screenshots_dir.absolute()}")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
