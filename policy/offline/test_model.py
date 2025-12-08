"""
Test trained model and visualize predictions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import cv2
from policy.offline.models.policy_transformer import PolicyTransformer
from policy.offline.dataset import DatasetBuilder
import matplotlib.pyplot as plt


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]

    # Recreate model
    model = PolicyTransformer(
        num_cards=config["num_cards"],
        num_troops=config.get("num_troops", 200),
        d_model=config["d_model"],
        n_head=config["n_head"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["sequence_length"],
        dropout=config["dropout"],
        arena_grid_size=tuple(config["arena_grid_size"]),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Global step: {checkpoint['global_step']}")

    return model, config


def visualize_predictions(model, dataloader, device, num_samples=5):
    """Visualize model predictions vs ground truth"""

    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_shown >= num_samples:
                break

            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)

            # Get predictions
            card_logits, position_logits = model(states, actions, rtg, timesteps)

            # Get predicted actions
            card_pred = card_logits.argmax(dim=-1)  # (B, T)
            pos_pred = position_logits.argmax(dim=-1)  # (B, T)

            # Ground truth
            card_true = actions[:, :, 0].long()
            pos_x_true = actions[:, :, 1]
            pos_y_true = actions[:, :, 2]

            # Show first sequence in batch
            B, T = card_pred.shape
            for b in range(min(B, num_samples - samples_shown)):
                print(f"\n{'='*60}")
                print(f"Sample {samples_shown + 1}")
                print(f"{'='*60}")

                for t in range(T):
                    # Extract values
                    state = states[b, t].cpu().numpy()
                    pred_card = card_pred[b, t].item()
                    true_card = card_true[b, t].item()
                    pred_pos = pos_pred[b, t].item()

                    # Convert position to x, y
                    pred_y = pred_pos // 18
                    pred_x = pred_pos % 18
                    true_x = pos_x_true[b, t].item()
                    true_y = pos_y_true[b, t].item()

                    # State info
                    elixir = state[0]
                    time = state[1]
                    cards = state[2:6]

                    print(f"\nTimestep {t}:")
                    print(f"  State: Elixir={elixir:.1f}, Time={time:.1f}s")
                    print(f"  Cards in hand: {cards}")
                    print(f"  Predicted: Card={pred_card}, Pos=({pred_x}, {pred_y})")
                    print(
                        f"  Ground Truth: Card={true_card}, Pos=({true_x:.1f}, {true_y:.1f})"
                    )
                    print(f"  Match: Card={'✅' if pred_card == true_card else '❌'}")

                samples_shown += 1
                if samples_shown >= num_samples:
                    break

            if samples_shown >= num_samples:
                break


def test_single_state(model, device):
    """Test model on a single crafted state"""
    print(f"\n{'='*60}")
    print("Testing on synthetic state")
    print(f"{'='*60}\n")

    # Create a fake state sequence (16 timesteps)
    # state = [elixir, time, card1, card2, card3, card4]
    states = []
    for t in range(16):
        state_t = [
            5.0 + t * 0.1,  # elixir increasing
            30.0 + t,  # time advancing
            10,
            25,
            42,
            56,  # cards in hand
        ]
        states.append(state_t)

    state = torch.FloatTensor([states]).to(device)  # (1, 16, 6)

    # Fake previous actions (zeros = no action)
    actions = torch.zeros(1, 16, 3).to(device)  # (1, 16, 3)

    # RTG (assume winning scenario)
    rtg = torch.ones(1, 16).to(device) * 0.8  # (1, 16)

    # Timesteps
    timesteps = torch.arange(16).unsqueeze(0).to(device)  # (1, 16)

    print(f"Input shapes:")
    print(f"  states: {state.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  rtg: {rtg.shape}")
    print(f"  timesteps: {timesteps.shape}")

    model.eval()
    with torch.no_grad():
        card_logits, position_logits = model(state, actions, rtg, timesteps)

    print(f"\nOutput shapes:")
    print(f"  card_logits: {card_logits.shape}")
    print(f"  position_logits: {position_logits.shape}")

    # Get predictions for last timestep
    pred_card = card_logits[0, -1].argmax().item()
    pred_pos = position_logits[0, -1].argmax().item()

    pred_y = pred_pos // 18
    pred_x = pred_pos % 18

    print(f"\n{'='*60}")
    print("PREDICTION FOR FINAL TIMESTEP")
    print(f"{'='*60}")
    print(f"State: Elixir=6.5, Time=45s, Cards=[10, 25, 42, 56]")
    print(f"RTG: 0.8 (trying to win)")
    print(f"\nModel predicts:")
    print(f"  Card slot: {pred_card}")
    print(f"  Position: ({pred_x}, {pred_y})")
    print(f"\nCard probabilities:")
    probs = torch.softmax(card_logits[0, -1], dim=0).cpu().numpy()
    for i, p in enumerate(probs):
        print(f"  Slot {i}: {p*100:.1f}%")

    print(f"\nPosition heatmap (top 5 positions):")
    pos_probs = torch.softmax(position_logits[0, -1], dim=0).cpu().numpy()
    top5_indices = np.argsort(pos_probs)[-5:][::-1]
    for idx in top5_indices:
        y = idx // 18
        x = idx % 18
        print(f"  ({x:2d}, {y:2d}): {pos_probs[idx]*100:.2f}%")


def main():
    # Find latest checkpoint
    runs_dir = Path("runs/policy_training")
    checkpoints = list(runs_dir.rglob("checkpoint_epoch_*.pt"))

    if not checkpoints:
        print("❌ No checkpoints found in runs/policy_training/")
        return

    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"📂 Using checkpoint: {latest_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, config = load_checkpoint(str(latest_checkpoint), device)

    # Test on synthetic state
    test_single_state(model, device)

    # Load dataset
    print(f"\n📊 Loading test data...")
    dataset_builder = DatasetBuilder("replay_data", config["sequence_length"])
    dataloader = dataset_builder.get_dataset(batch_size=4, num_workers=0, shuffle=False)

    # Visualize predictions
    print(f"\n🔍 Visualizing predictions on real data...")
    visualize_predictions(model, dataloader, device, num_samples=3)


if __name__ == "__main__":
    main()
