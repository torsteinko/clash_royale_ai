"""
Evaluate trained policy on test replays
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from train import PolicyTransformer, TrainConfig
from dataset import DatasetBuilder


def evaluate_model(checkpoint_path, replay_dir, batch_size=16):
    """Evaluate model on replay data"""

    # Load config from checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config_dict = checkpoint.get("config", {})

    # Create config
    config = TrainConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Load dataset
    print(f"\n📂 Loading evaluation data from {replay_dir}")
    dataset_builder = DatasetBuilder(replay_dir, config.sequence_length)
    dataloader = dataset_builder.get_dataset(
        batch_size=batch_size, num_workers=0, shuffle=False  # Don't shuffle for eval
    )

    if dataloader is None:
        print("❌ No data found")
        return

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyTransformer(
        num_cards=config.num_cards,
        num_troops=config.num_troops,
        d_model=config.d_model,
        n_head=config.n_head,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.sequence_length,
        dropout=config.dropout,
        arena_grid_size=config.arena_grid_size,
        state_dim=config.state_dim,
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"   Device: {device}")

    # Evaluation metrics
    total_card_acc = 0
    total_pos_acc = 0
    total_action_frames = 0
    card_predictions = []
    card_targets = []

    print(f"\n🔍 Evaluating on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch is None:
                continue

            # Move to device
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)

            B, T = rtg.shape

            # Shifted actions for teacher forcing
            actions_input = torch.cat(
                [torch.zeros(B, 1, 3, device=device), actions[:, :-1, :]], dim=1
            )

            actions_target = actions

            # Forward pass
            card_logits, position_logits = model(states, actions_input, rtg, timesteps)

            # Get targets
            target_cards = actions_target[:, :, 0].long()

            grid_h, grid_w = config.arena_grid_size
            pos_x_raw = actions_target[:, :, 1]
            pos_y_raw = actions_target[:, :, 2]

            if pos_x_raw.max() <= 1.0 and pos_y_raw.max() <= 1.0:
                pos_x = (pos_x_raw * (grid_w - 1)).long().clamp(0, grid_w - 1)
                pos_y = (pos_y_raw * (grid_h - 1)).long().clamp(0, grid_h - 1)
            else:
                pos_x = pos_x_raw.long().clamp(0, grid_w - 1)
                pos_y = pos_y_raw.long().clamp(0, grid_h - 1)

            target_positions = (pos_y * grid_w + pos_x).long()
            target_positions = target_positions.clamp(0, grid_h * grid_w - 1)

            # Mask for action frames only
            mask = (actions_target[:, :, 0] > 0).float().reshape(-1)
            num_actions = mask.sum()

            if num_actions > 0:
                # Predictions
                card_pred = card_logits.argmax(dim=-1).reshape(-1)
                position_pred = position_logits.argmax(dim=-1).reshape(-1)

                target_cards_flat = target_cards.reshape(-1)
                target_positions_flat = target_positions.reshape(-1)

                # Calculate accuracy on action frames
                card_acc = ((card_pred == target_cards_flat).float() * mask).sum()
                pos_acc = (
                    (position_pred == target_positions_flat).float() * mask
                ).sum()

                total_card_acc += card_acc.item()
                total_pos_acc += pos_acc.item()
                total_action_frames += num_actions.item()

                # Collect predictions for analysis
                action_mask_bool = mask.bool()
                card_predictions.extend(card_pred[action_mask_bool].cpu().numpy())
                card_targets.extend(target_cards_flat[action_mask_bool].cpu().numpy())

    # Print results
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total action frames evaluated: {total_action_frames}")
    print(f"Card accuracy: {total_card_acc / total_action_frames * 100:.2f}%")
    print(f"Position accuracy: {total_pos_acc / total_action_frames * 100:.2f}%")

    # Card prediction distribution
    from collections import Counter

    pred_dist = Counter(card_predictions)
    target_dist = Counter(card_targets)

    print(f"\n📊 Card Prediction Analysis:")
    print(f"   Unique cards predicted: {len(pred_dist)}")
    print(f"   Top 5 predicted cards:")
    for card_id, count in pred_dist.most_common(5):
        pct = count / len(card_predictions) * 100
        print(f"     Card {card_id}: {count} ({pct:.1f}%)")

    print(f"\n   Top 5 target cards:")
    for card_id, count in target_dist.most_common(5):
        pct = count / len(card_targets) * 100
        print(f"     Card {card_id}: {count} ({pct:.1f}%)")

    print(f"{'='*60}\n")

    return {
        "card_accuracy": total_card_acc / total_action_frames,
        "position_accuracy": total_pos_acc / total_action_frames,
        "total_actions": total_action_frames,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--replay-dir", type=str, default="replay_data", help="Replay directory"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.replay_dir, args.batch_size)
