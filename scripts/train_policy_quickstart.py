"""
Example: Quick start for offline RL training
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 80)
    print("🎮 Clash Royale AI - Offline RL Training Quick Start")
    print("=" * 80)
    print()

    # Check if replay_data exists
    replay_dir = Path("replay_data")
    if not replay_dir.exists():
        print("📁 Creating replay_data directory...")
        replay_dir.mkdir()

    # Check for existing replay files
    replay_files = list(replay_dir.glob("*.pkl")) + list(replay_dir.glob("*.xz"))

    if not replay_files:
        print("⚠️  No replay data found!")
        print()
        print("To collect replay data, you have two options:")
        print()
        print("1. From a recorded video:")
        print(
            "   python scripts/collect_replay_data.py --mode video --video recordings/gameplay.mp4"
        )
        print()
        print("2. From live gameplay:")
        print("   python scripts/collect_replay_data.py --mode live --duration 180")
        print()
        print("After collecting data, run this script again to start training.")
        return

    print(f"✅ Found {len(replay_files)} replay files")
    print()

    # Ask user if they want to start training
    response = input("Start training? [y/n]: ").lower().strip()

    if response != "y":
        print("Training cancelled.")
        return

    print()
    print("🚀 Starting training...")
    print("   This may take a while depending on your data and hardware.")
    print("   Monitor progress with: tensorboard --logdir runs/policy_training")
    print()

    # Run training
    cmd = [
        sys.executable,
        "policy/offline/train.py",
        "--replay-dir",
        "replay_data",
        "--batch-size",
        "16",
        "--epochs",
        "50",
        "--lr",
        "1e-4",
    ]

    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 80)
        print("✅ Training complete!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Check training logs: tensorboard --logdir runs/policy_training")
        print("2. Find checkpoints in: runs/policy_training/<timestamp>/checkpoints/")
        print(
            "3. Use the trained model for inference (implement your own inference script)"
        )

    except subprocess.CalledProcessError as e:
        print()
        print("❌ Training failed!")
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user")


if __name__ == "__main__":
    main()
