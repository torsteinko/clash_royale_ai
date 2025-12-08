"""
Collect replay data for offline RL training

This script records gameplay and saves state-action-reward sequences
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pickle
import lzma
from datetime import datetime
from tqdm import tqdm
import argparse

from game_state.state_extractor import GameStateExtractor
from policy.builders.state_builder import StateBuilder, ActionBuilder, RewardBuilder
from utils.screen_capture import ScreenCapture
from policy.utils import colorstr


class ReplayCollector:
    """Collect and save replay data"""

    def __init__(
        self,
        yolo_model_path="best.pt",
        deck=None,
        save_dir="replay_data",
        compress=True,
    ):
        """
        Args:
            yolo_model_path: Path to YOLO model
            deck: Optional deck for filtering (list of card names)
            save_dir: Directory to save replays
            compress: Whether to compress replay files (.xz)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.compress = compress

        print(f"\n{'='*80}")
        print(f"🎮 INITIALIZING REPLAY COLLECTOR")
        print(f"{'='*80}")

        # Initialize extractors
        self.state_extractor = GameStateExtractor(
            yolo_model_path=yolo_model_path, deck=deck
        )

        self.state_builder = StateBuilder()
        self.action_builder = ActionBuilder()
        self.reward_builder = RewardBuilder()

        # Screen capture
        self.screen_capture = ScreenCapture()

        # Episode data
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
        }

        self.episodes_collected = 0

        print(f"Save directory: {self.save_dir}")
        print(f"Compression: {'Enabled (.xz)' if compress else 'Disabled (.pkl)'}")
        print(f"{'='*80}\n")

    def reset_episode(self):
        """Reset episode data"""
        self.current_episode = {
            "states": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
        }
        self.state_builder.reset()
        self.action_builder.reset()
        self.reward_builder.reset()

    def collect_frame(self, frame, action_card_slot=-1, action_position=None):
        """
        Collect one frame of data

        Args:
            frame: Screenshot frame (numpy array)
            action_card_slot: Card slot played (-1 for no action)
            action_position: (x, y) position where card was played
        """
        # Extract game state
        game_state = self.state_extractor.extract_state(frame, debug=False)

        # Build training state
        state = self.state_builder.build_state(game_state)

        # Build action
        action = self.action_builder.build_action(action_card_slot, action_position)

        # Calculate reward
        reward = self.reward_builder.calculate_reward(game_state)

        # Store data
        self.current_episode["states"].append(state)
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(reward)
        self.current_episode["terminals"].append(False)  # Not terminal yet

    def end_episode(self, won=False, crowns_won=0, crowns_lost=0):
        """
        Mark episode as complete and save

        Args:
            won: Whether the game was won
            crowns_won: Number of enemy towers destroyed
            crowns_lost: Number of ally towers destroyed
        """
        if not self.current_episode["states"]:
            print("Warning: No data in episode, skipping save")
            return

        # Mark last frame as terminal
        self.current_episode["terminals"][-1] = True

        # Update final reward
        final_reward = self.reward_builder.get_episode_reward(
            won, crowns_won, crowns_lost
        )
        self.current_episode["rewards"][-1] = final_reward

        # Save episode
        self._save_episode()

        # Reset for next episode
        self.reset_episode()
        self.episodes_collected += 1

    def _save_episode(self):
        """Save current episode to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"replay_{timestamp}_ep{self.episodes_collected:04d}"

        # Convert to numpy arrays for efficiency
        data = {
            "states": self.current_episode["states"],
            "actions": self.current_episode["actions"],
            "rewards": np.array(self.current_episode["rewards"], dtype=np.float32),
            "terminals": np.array(self.current_episode["terminals"], dtype=bool),
        }

        if self.compress:
            filepath = self.save_dir / f"{filename}.xz"
            with lzma.open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            filepath = self.save_dir / f"{filename}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

        total_reward = data["rewards"].sum()
        n_frames = len(data["states"])

        print(
            f"Saved episode {self.episodes_collected}: {n_frames} frames, total reward: {total_reward:.2f}"
        )
        print(f"  File: {filepath}")

    def collect_from_recording(self, video_path, manual_annotations=None):
        """
        Collect data from a pre-recorded video

        Args:
            video_path: Path to video file
            manual_annotations: Optional dict with frame annotations
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"\nProcessing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        self.reset_episode()
        frame_count = 0

        pbar = tqdm(total=total_frames, desc="Processing frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Collect frame (no action by default)
            self.collect_frame(frame, action_card_slot=-1, action_position=None)

            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        # End episode (you may want to pass actual results here)
        print("\nEnding episode...")
        self.end_episode(won=False, crowns_won=0, crowns_lost=0)

        print(f"Collected {frame_count} frames from video")

    def collect_live(self, duration_seconds=180, fps=5):
        """
        Collect data from live gameplay

        Args:
            duration_seconds: How long to record (default 3 minutes)
            fps: Frames per second to capture
        """
        print(f"\nStarting live collection for {duration_seconds}s at {fps} FPS")
        print("Press Ctrl+C to stop early\n")

        self.reset_episode()

        import time

        frame_interval = 1.0 / fps
        start_time = time.time()
        next_capture_time = start_time

        try:
            while time.time() - start_time < duration_seconds:
                current_time = time.time()

                if current_time >= next_capture_time:
                    # Capture frame
                    frame = self.screen_capture.capture()

                    if frame is not None:
                        # Collect frame (no action detection for now)
                        self.collect_frame(
                            frame, action_card_slot=-1, action_position=None
                        )
                        print(
                            f"Captured frame {len(self.current_episode['states'])}",
                            end="\r",
                        )

                    next_capture_time += frame_interval

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        # End episode
        print("\n\nEnding episode...")
        self.end_episode(won=False, crowns_won=0, crowns_lost=0)

        print(f"Collected {len(self.current_episode['states'])} frames")


def main():
    parser = argparse.ArgumentParser(description="Collect replay data")
    parser.add_argument(
        "--mode",
        choices=["live", "video"],
        default="video",
        help="Collection mode: live gameplay or from video",
    )
    parser.add_argument("--video", type=str, help="Path to video file (if mode=video)")
    parser.add_argument(
        "--duration", type=int, default=180, help="Duration in seconds (if mode=live)"
    )
    parser.add_argument("--fps", type=int, default=5, help="Capture FPS")
    parser.add_argument("--model", type=str, default="best.pt", help="YOLO model path")
    parser.add_argument(
        "--deck", nargs="+", help="Deck cards (e.g., knight archer fireball)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="replay_data", help="Save directory"
    )
    parser.add_argument(
        "--no-compress", action="store_true", help="Disable compression"
    )
    args = parser.parse_args()

    # Create collector
    collector = ReplayCollector(
        yolo_model_path=args.model,
        deck=args.deck,
        save_dir=args.save_dir,
        compress=not args.no_compress,
    )

    # Collect data
    if args.mode == "video":
        if not args.video:
            print(colorstr("red", "bold", "ERROR: --video required for video mode"))
            return

        if not Path(args.video).exists():
            print(colorstr("red", "bold", f"ERROR: Video file not found: {args.video}"))
            return

        collector.collect_from_recording(args.video)

    else:  # live mode
        collector.collect_live(duration_seconds=args.duration, fps=args.fps)

    print(f"\n{colorstr('green', 'bold', 'Collection complete!')}")
    print(f"Episodes collected: {collector.episodes_collected}")
    print(f"Saved to: {collector.save_dir}")


if __name__ == "__main__":
    main()
