"""
Test script to verify offline RL training setup
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        import torch

        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
        return False

    try:
        import numpy as np

        print(f"  ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy: {e}")
        return False

    try:
        from tqdm import tqdm

        print(f"  ✅ tqdm")
    except ImportError as e:
        print(f"  ❌ tqdm: {e}")
        return False

    try:
        from torch.utils.tensorboard import SummaryWriter

        print(f"  ✅ TensorBoard")
    except ImportError as e:
        print(f"  ❌ TensorBoard: {e}")
        return False

    return True


def test_project_structure():
    """Test that all required directories exist"""
    print("\nTesting project structure...")

    required_dirs = [
        "policy",
        "policy/offline",
        "policy/offline/models",
        "policy/offline/cnn",
        "policy/builders",
        "policy/utils",
        "scripts",
        "game_state",
        "detection",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} not found")
            all_exist = False

    return all_exist


def test_module_imports():
    """Test that custom modules can be imported"""
    print("\nTesting custom module imports...")

    try:
        from policy.builders.state_builder import (
            StateBuilder,
            ActionBuilder,
            RewardBuilder,
        )

        print("  ✅ State/Action/Reward builders")
    except ImportError as e:
        print(f"  ❌ Builders: {e}")
        return False

    try:
        from policy.offline.dataset import DatasetBuilder

        print("  ✅ Dataset builder")
    except ImportError as e:
        print(f"  ❌ Dataset builder: {e}")
        return False

    try:
        from policy.offline.models.policy_transformer import PolicyTransformer

        print("  ✅ Policy Transformer")
    except ImportError as e:
        print(f"  ❌ Policy Transformer: {e}")
        return False

    try:
        from game_state.state_extractor import GameStateExtractor

        print("  ✅ Game State Extractor")
    except ImportError as e:
        print(f"  ❌ Game State Extractor: {e}")
        return False

    return True


def test_model_creation():
    """Test that model can be instantiated"""
    print("\nTesting model creation...")

    try:
        from policy.offline.models.policy_transformer import PolicyTransformer
        import torch

        model = PolicyTransformer(
            num_cards=108,
            num_troops=200,
            d_model=128,  # Small for testing
            n_head=4,
            n_layers=2,
            arena_grid_size=(32, 18),
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created with {n_params:,} parameters")

        # Test forward pass
        B, T = 2, 4
        rtg = torch.randn(B, T)
        timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)
        actions = torch.zeros(B, T, 3, dtype=torch.long)  # card_id, pos_x, pos_y
        # States are (B, T, state_dim) tensors — matches policy/offline/train.py
        # (`states = batch["states"].to(device)`); state_dim defaults to 126.
        states = torch.randn(B, T, 126)

        card_logits, pos_logits = model(states, actions, rtg, timesteps)
        print(f"  ✅ Forward pass successful")
        print(f"     Output shapes: cards={card_logits.shape}, pos={pos_logits.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_directories():
    """Check data directories"""
    print("\nChecking data directories...")

    replay_dir = Path("replay_data")
    if not replay_dir.exists():
        print(f"  ⚠️  replay_data/ not found, creating...")
        replay_dir.mkdir(exist_ok=True)
        print(f"  ✅ Created replay_data/")
    else:
        replay_files = list(replay_dir.glob("*.pkl")) + list(replay_dir.glob("*.xz"))
        if replay_files:
            print(f"  ✅ replay_data/ exists with {len(replay_files)} files")
        else:
            print(f"  ⚠️  replay_data/ exists but is empty")
            print(f"     Run: python scripts/collect_replay_data.py")

    return True


def main():
    print("=" * 80)
    print("🎮 Offline RL Training - Setup Verification")
    print("=" * 80)
    print()

    results = []

    # Run tests
    results.append(("Import tests", test_imports()))
    results.append(("Project structure", test_project_structure()))
    results.append(("Module imports", test_module_imports()))
    results.append(("Model creation", test_model_creation()))
    results.append(("Data directories", test_data_directories()))

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✅ All tests passed! You're ready to start training.")
        print()
        print("Next steps:")
        print("1. Collect replay data:")
        print(
            "   python scripts/collect_replay_data.py --mode video --video recordings/gameplay.mp4"
        )
        print()
        print("2. Start training:")
        print("   python policy/offline/train.py")
        print()
        print("3. Or use quick start:")
        print("   python scripts/train_policy_quickstart.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("- Install missing dependencies: pip install -r requirements_policy.txt")
        print("- Make sure you're in the project root directory")
        print("- Check Python path if modules can't be imported")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
