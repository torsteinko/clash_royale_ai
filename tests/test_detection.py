# tests/test_detection.py
"""Simple detection test - saves screenshots"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import mss
import time
from game_state.state_extractor import GameStateExtractor


def capture_screen():
    """Capture screen using MSS"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def main():
    print("\n" + "="*80)
    print("🎮 CLASH ROYALE BOT - DETECTION TEST")
    print("="*80)
    print("\n📋 Instructions:")
    print("  1. Open Nulls Royale in MEmu")
    print("  2. Start a match")
    print("  3. Press Enter to begin")
    print("  4. Press Ctrl+C to stop")
    print("\n💾 Screenshots will be saved to: screenshots/")
    print("="*80)
    input("\nPress Enter to continue...")
    
    # Initialize detector
    extractor = GameStateExtractor()
    
    # Create screenshots folder in ROOT directory
    screenshots_dir = Path(__file__).parent.parent / 'screenshots'
    screenshots_dir.mkdir(exist_ok=True)
    print(f"\n📁 Screenshots folder: {screenshots_dir.absolute()}\n")
    
    print("🔍 Detection started! Press Ctrl+C to stop\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            frame = capture_screen()
            
            # Extract game state
            state = extractor.extract_state(frame)
            
            # Visualize detections
            vis_frame = extractor.visualize_state(frame, state)
            
            # Resize for storage
            vis_frame = cv2.resize(vis_frame, (1280, 720))
            
            # Save screenshot every 2 seconds
            frame_count += 1
            if frame_count % 60 == 0:
                filename = screenshots_dir / f'detection_{frame_count:05d}.jpg'
                cv2.imwrite(str(filename), vis_frame)
                print(f"📸 Saved: {filename.name}")
            
            # Print state every 1 second
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                
                print(f"\n{'='*70}")
                print(f"Frame {frame_count:05d} | FPS: {fps:.1f}")
                print(f"{'='*70}")
                print(f"⚡ Elixir:       {state['elixir']}")
                print(f"⏱️  Match Time:   {state['match_time']}s")
                print(f"🎴 Cards:        {', '.join(state['cards_in_hand'])}")
                print(f"🟢 Ally Troops:  {state['troops']['total_ally']}")
                print(f"🔴 Enemy Troops: {state['troops']['total_enemy']}")
                
                # Show detected troops
                if state['troops']['ally']:
                    ally_types = [t['type'] for t in state['troops']['ally']]
                    print(f"   → Ally:  {', '.join(set(ally_types))}")
                
                if state['troops']['enemy']:
                    enemy_types = [t['type'] for t in state['troops']['enemy']]
                    print(f"   → Enemy: {', '.join(set(enemy_types))}")
            
            # Control frame rate (~30 FPS)
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
        print(f"   • Screenshots saved: {frame_count//60}")
        print(f"\n📁 Check screenshots/ folder: {screenshots_dir.absolute()}")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
