"""
Training script for offline reinforcement learning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime

from policy.offline.models.policy_transformer import PolicyTransformer
from policy.offline.dataset import DatasetBuilder
from policy.utils import colorstr


class TrainConfig:
    """Training configuration"""

    def __init__(self):
        # Model - 2.6 hog cycle deck (8 cards total with evolutions)
        self.num_cards = 8  # Changed from 108
        self.num_troops = 200
        self.d_model = 256
        self.n_head = 8
        self.n_layers = 4
        self.d_ff = 1024
        self.dropout = 0.1
        self.sequence_length = 16
        self.arena_grid_size = (32, 18)  # (height, width)

        # Training
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_grad_norm = 1.0

        # Data
        self.replay_dir = "replay_data"
        self.num_workers = 0

        # Logging
        self.log_dir = (
            Path("runs") / "policy_training" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.save_freq = 2  # Save checkpoint every N epochs
        self.log_freq = 100  # Log to tensorboard every N steps


class Trainer:
    """Trainer for offline RL policy"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*80}")
        print(f"🎮 INITIALIZING POLICY TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Sequence length: {config.sequence_length}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Num cards in deck: {config.num_cards}")
        print(f"Arena grid: {config.arena_grid_size}")
        print(f"{'='*80}\n")

        # Create model
        self.model = PolicyTransformer(
            num_cards=config.num_cards,
            num_troops=config.num_troops,
            d_model=config.d_model,
            n_head=config.n_head,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.sequence_length,
            dropout=config.dropout,
            arena_grid_size=config.arena_grid_size,
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Validate initial model parameters
        print("Checking initial model parameters...")
        has_nan = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ Initial parameter '{name}' contains NaN!")
                has_nan = True
            elif torch.isinf(param).any():
                print(f"❌ Initial parameter '{name}' contains Inf!")
                has_nan = True

        if has_nan:
            raise ValueError("Model initialized with NaN/Inf parameters!")
        print("✓ Model parameters initialized correctly")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
        )

        # Loss functions
        self.card_criterion = nn.CrossEntropyLoss()
        self.position_criterion = nn.CrossEntropyLoss()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Create checkpoint dir
        self.checkpoint_dir = config.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_card_loss = 0
        total_pos_loss = 0
        total_card_acc = 0
        total_pos_acc = 0
        valid_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            try:
                # Skip None batches (filtered out by collate_fn)
                if batch is None:
                    continue

                # Move to device
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                rtg = batch["rtg"].to(self.device)
                timesteps = batch["timesteps"].to(self.device)
                rewards = batch["rewards"].to(self.device)

                B, T = rtg.shape

                # ==================================================================
                # DEBUG: Print action ranges on first batch
                # ==================================================================
                if batch_idx == 0 and self.epoch == 0:
                    print(f"\n{'='*60}")
                    print(f"DATA FORMAT INSPECTION (first batch)")
                    print(f"{'='*60}")
                    print(f"Batch size: {B}, Sequence length: {T}")
                    print(f"States shape: {states.shape}")
                    print(f"Actions shape: {actions.shape}")
                    print(f"Action stats:")
                    print(
                        f"  card_id range: [{actions[:, :, 0].min():.2f}, {actions[:, :, 0].max():.2f}]"
                    )
                    print(
                        f"  pos_x range: [{actions[:, :, 1].min():.2f}, {actions[:, :, 1].max():.2f}]"
                    )
                    print(
                        f"  pos_y range: [{actions[:, :, 2].min():.2f}, {actions[:, :, 2].max():.2f}]"
                    )
                    unique_cards = torch.unique(actions[:, :, 0].long())
                    print(f"Unique card IDs: {unique_cards.tolist()}")
                    print(f"Number of unique cards: {len(unique_cards)}")
                    print(f"{'='*60}\n")

                # ==================================================================
                # CARD TARGET PROCESSING
                # ==================================================================
                # KataCR card_id is ALREADY the slot index (0-4)
                # 0 = no action, 1-4 = card slots
                # We need to convert to 0-3 for model (which predicts 4 card slots)

                card_slots_raw = actions[:, :, 0].long()  # 0-4

                # Debug card_id values before conversion
                # if batch_idx % 50 == 0 or batch_idx < 5:
                #     print(f"\n[Batch {batch_idx}] Card ID Debug:")
                #     print(
                #         f"  card_slots_raw range: [{card_slots_raw.min()}, {card_slots_raw.max()}]"
                #     )
                #     print(
                #         f"  card_slots_raw unique: {torch.unique(card_slots_raw).tolist()}"
                #     )
                #     print(f"  card_slots_raw shape: {card_slots_raw.shape}")

                # Convert card_id to valid range [0-3]:
                # 0 (no action) -> 0 (map to first slot as placeholder)
                # 1-4 (actual slots) -> 0-3
                target_cards = torch.where(
                    card_slots_raw == 0,
                    torch.zeros_like(card_slots_raw),  # no action -> slot 0
                    card_slots_raw - 1,  # slots 1-4 -> 0-3
                ).clamp(0, 3)

                # Validate conversion
                if target_cards.min() < 0 or target_cards.max() >= 4:
                    print(
                        f"❌ [Batch {batch_idx}] Invalid target_cards after conversion!"
                    )
                    print(
                        f"  target_cards range: [{target_cards.min()}, {target_cards.max()}]"
                    )
                    print(
                        f"  target_cards unique: {torch.unique(target_cards).tolist()}"
                    )
                    print(f"  Skipping batch")
                    continue

                # ==================================================================
                # POSITION TARGET PROCESSING
                # ==================================================================
                grid_h, grid_w = self.config.arena_grid_size  # (32, 18)

                pos_x_raw = actions[:, :, 1]
                pos_y_raw = actions[:, :, 2]

                # Auto-detect coordinate format
                if pos_x_raw.max() <= 1.0 and pos_y_raw.max() <= 1.0:
                    # Normalized [0, 1] coordinates
                    pos_x = (pos_x_raw * (grid_w - 1)).long().clamp(0, grid_w - 1)
                    pos_y = (pos_y_raw * (grid_h - 1)).long().clamp(0, grid_h - 1)
                else:
                    # Arena grid coordinates
                    pos_x = pos_x_raw.long().clamp(0, grid_w - 1)
                    pos_y = pos_y_raw.long().clamp(0, grid_h - 1)

                # Convert to flat index: index = y * width + x
                target_positions = (pos_y * grid_w + pos_x).long()
                target_positions = target_positions.clamp(0, grid_h * grid_w - 1)

                # Validate
                max_pos = grid_h * grid_w - 1
                if target_positions.min() < 0 or target_positions.max() > max_pos:
                    print(f"⚠️  Invalid position targets at batch {batch_idx}")
                    print(
                        f"  Range: [{target_positions.min()}, {target_positions.max()}]"
                    )
                    print(f"  Max allowed: {max_pos}")
                    print(f"  Skipping batch")
                    continue

                # ==================================================================
                # FORWARD PASS
                # ==================================================================
                # Validate inputs before forward pass
                if torch.isnan(states).any():
                    print(f"⚠️  [Batch {batch_idx}] NaN in states, skipping")
                    continue
                if torch.isnan(actions).any():
                    print(f"⚠️  [Batch {batch_idx}] NaN in actions, skipping")
                    continue
                if torch.isnan(rtg).any():
                    print(f"⚠️  [Batch {batch_idx}] NaN in rtg, skipping")
                    continue
                if torch.isnan(timesteps).any():
                    print(f"⚠️  [Batch {batch_idx}] NaN in timesteps, skipping")
                    continue

                # Check if model has NaN parameters
                model_has_nan = False
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        print(
                            f"❌ [Batch {batch_idx}] Model parameter '{name}' contains NaN!"
                        )
                        model_has_nan = True
                        break

                if model_has_nan:
                    print(
                        f"❌ Model weights corrupted with NaN. Training cannot continue."
                    )
                    print(
                        f"❌ This likely happened due to gradient explosion in a previous batch."
                    )
                    print(
                        f"❌ Try: 1) Lower learning rate, 2) Increase gradient clipping, 3) Check data normalization"
                    )
                    return {
                        "loss": float("inf"),
                        "card_loss": float("inf"),
                        "pos_loss": float("inf"),
                        "card_acc": 0.0,
                        "pos_acc": 0.0,
                    }

                card_logits, position_logits = self.model(
                    states, actions, rtg, timesteps
                )

                # Check outputs immediately
                if torch.isnan(card_logits).any():
                    print(f"❌ [Batch {batch_idx}] Model produced NaN in card_logits!")
                    print(f"  Input stats:")
                    print(
                        f"    states: [{states.min():.2f}, {states.max():.2f}], mean={states.mean():.2f}"
                    )
                    print(
                        f"    actions: [{actions.min():.2f}, {actions.max():.2f}], mean={actions.mean():.2f}"
                    )
                    print(
                        f"    rtg: [{rtg.min():.2f}, {rtg.max():.2f}], mean={rtg.mean():.2f}"
                    )
                    continue

                if torch.isnan(position_logits).any():
                    print(
                        f"❌ [Batch {batch_idx}] Model produced NaN in position_logits!"
                    )
                    continue

                # ==================================================================
                # LOSS CALCULATION
                # ==================================================================
                # Debug shapes
                # if batch_idx < 3 or (batch_idx >= 104 and batch_idx <= 110):
                #     print(f"\nBatch {batch_idx} shapes:")
                #     print(f"  card_logits: {card_logits.shape}")
                #     print(f"  position_logits: {position_logits.shape}")
                #     print(f"  target_cards: {target_cards.shape}")
                #     print(f"  target_positions: {target_positions.shape}")
                #     print(f"  states: {states.shape}")
                #     print(f"  actions: {actions.shape}")

                # Ensure shapes match before flattening
                if len(card_logits.shape) != 3 or len(target_cards.shape) != 2:
                    print(f"⚠️  Unexpected tensor dimensions at batch {batch_idx}")
                    print(f"  card_logits: {card_logits.shape} (expected 3D)")
                    print(f"  target_cards: {target_cards.shape} (expected 2D)")
                    print(f"  Skipping batch")
                    continue

                B_logits, T_logits, _ = card_logits.shape
                B_targets, T_targets = target_cards.shape

                if B_logits != B_targets or T_logits != T_targets:
                    print(f"⚠️  Shape mismatch at batch {batch_idx}")
                    print(
                        f"  Logits: {card_logits.shape}, Targets: {target_cards.shape}"
                    )
                    print(f"  Skipping batch")
                    continue

                # Reshape using actual dimensions from tensors, not config
                num_card_classes = card_logits.shape[-1]  # Should be 4
                card_logits_flat = card_logits.reshape(-1, num_card_classes)
                target_cards_flat = target_cards.reshape(-1)

                position_logits_flat = position_logits.reshape(-1, grid_h * grid_w)
                target_positions_flat = target_positions.reshape(-1)

                # Final shape validation before loss
                if card_logits_flat.shape[0] != target_cards_flat.shape[0]:
                    print(f"⚠️  Flattened shape mismatch at batch {batch_idx}")
                    print(
                        f"  card_logits: {card_logits.shape} -> flat: {card_logits_flat.shape}"
                    )
                    print(
                        f"  target_cards: {target_cards.shape} -> flat: {target_cards_flat.shape}"
                    )
                    print(f"  B_logits={B_logits}, T_logits={T_logits}")
                    print(f"  B_targets={B_targets}, T_targets={T_targets}")
                    print(f"  Skipping batch")
                    continue

                # Debug before loss calculation
                # if batch_idx % 50 == 0 or batch_idx < 5:
                #     print(f"\n[Batch {batch_idx}] Before loss calculation:")
                #     print(
                #         f"  card_logits_flat: {card_logits_flat.shape}, range: [{card_logits_flat.min():.2f}, {card_logits_flat.max():.2f}]"
                #     )
                #     print(
                #         f"  target_cards_flat: {target_cards_flat.shape}, range: [{target_cards_flat.min()}, {target_cards_flat.max()}]"
                #     )
                #     print(
                #         f"  target_cards_flat unique: {torch.unique(target_cards_flat).tolist()}"
                #     )
                #     print(
                #         f"  card_logits has nan: {torch.isnan(card_logits_flat).any()}"
                #     )
                #     print(
                #         f"  card_logits has inf: {torch.isinf(card_logits_flat).any()}"
                #     )

                card_loss = self.card_criterion(card_logits_flat, target_cards_flat)
                position_loss = self.position_criterion(
                    position_logits_flat, target_positions_flat
                )

                # Check for nan/inf with detailed debugging
                if torch.isnan(card_loss) or torch.isinf(card_loss):
                    print(f"\n❌ [Batch {batch_idx}] NaN/Inf card_loss detected!")
                    print(f"  card_loss value: {card_loss}")
                    print(f"  card_logits_flat stats:")
                    print(f"    shape: {card_logits_flat.shape}")
                    print(
                        f"    range: [{card_logits_flat.min():.4f}, {card_logits_flat.max():.4f}]"
                    )
                    print(
                        f"    mean: {card_logits_flat.mean():.4f}, std: {card_logits_flat.std():.4f}"
                    )
                    print(f"    has nan: {torch.isnan(card_logits_flat).any()}")
                    print(f"    has inf: {torch.isinf(card_logits_flat).any()}")
                    print(f"  target_cards_flat stats:")
                    print(f"    shape: {target_cards_flat.shape}")
                    print(
                        f"    range: [{target_cards_flat.min()}, {target_cards_flat.max()}]"
                    )
                    print(
                        f"    unique values: {torch.unique(target_cards_flat).tolist()}"
                    )
                    print(f"  Skipping batch")
                    continue

                if torch.isnan(position_loss) or torch.isinf(position_loss):
                    print(f"⚠️  NaN/Inf position_loss at batch {batch_idx}, skipping")
                    continue

                loss = card_loss + position_loss

                # ==================================================================
                # BACKWARD PASS
                # ==================================================================
                self.optimizer.zero_grad()
                loss.backward()

                # Check for NaN gradients before clipping
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"⚠️  [Batch {batch_idx}] NaN gradient in '{name}'")
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(
                        f"⚠️  [Batch {batch_idx}] Skipping update due to NaN gradients"
                    )
                    self.optimizer.zero_grad()  # Clear bad gradients
                    continue

                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # Log extreme gradient norms (only if very large)
                if grad_norm > 50.0:
                    print(
                        f"⚠️  [Batch {batch_idx}] Large gradient norm: {grad_norm:.2f}"
                    )

                self.optimizer.step()
                self.scheduler.step()

                # ==================================================================
                # METRICS
                # ==================================================================
                with torch.no_grad():
                    card_pred = card_logits.argmax(dim=-1)
                    card_acc = (card_pred == target_cards).float().mean()

                    position_pred = position_logits.argmax(dim=-1)
                    pos_acc = (position_pred == target_positions).float().mean()

                # Update metrics
                total_loss += loss.item()
                total_card_loss += card_loss.item()
                total_pos_loss += position_loss.item()
                total_card_acc += card_acc.item()
                total_pos_acc += pos_acc.item()
                valid_batches += 1

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "card_loss": f"{card_loss.item():.4f}",
                        "pos_loss": f"{position_loss.item():.4f}",
                        "card_acc": f"{card_acc.item():.3f}",
                        "pos_acc": f"{pos_acc.item():.3f}",
                    }
                )

                # Log to tensorboard
                if self.global_step % self.config.log_freq == 0:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar(
                        "train/card_loss", card_loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/position_loss", position_loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/card_accuracy", card_acc.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/position_accuracy", pos_acc.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/learning_rate",
                        self.scheduler.get_last_lr()[0],
                        self.global_step,
                    )

                self.global_step += 1

            except Exception as e:
                print(f"⚠️  Error in batch {batch_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Return average metrics
        if valid_batches == 0:
            print("❌ No valid batches processed!")
            return {
                "loss": float("inf"),
                "card_loss": float("inf"),
                "pos_loss": float("inf"),
                "card_acc": 0.0,
                "pos_acc": 0.0,
            }

        return {
            "loss": total_loss / valid_batches,
            "card_loss": total_card_loss / valid_batches,
            "pos_loss": total_pos_loss / valid_batches,
            "card_acc": total_card_acc / valid_batches,
            "pos_acc": total_pos_acc / valid_batches,
        }

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def train(self, dataloader):
        """Main training loop"""
        print(f"\n{colorstr('green', 'bold', 'Starting training...')}\n")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train one epoch
            metrics = self.train_epoch(dataloader)

            # Log epoch metrics
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Card Loss: {metrics['card_loss']:.4f}")
            print(f"  Position Loss: {metrics['pos_loss']:.4f}")
            print(f"  Card Accuracy: {metrics['card_acc']:.3f}")
            print(f"  Position Accuracy: {metrics['pos_acc']:.3f}")

            self.writer.add_scalar("epoch/loss", metrics["loss"], epoch)
            self.writer.add_scalar("epoch/card_accuracy", metrics["card_acc"], epoch)
            self.writer.add_scalar("epoch/position_accuracy", metrics["pos_acc"], epoch)

            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch + 1)

        # Save final checkpoint
        self.save_checkpoint(self.config.num_epochs)

        print(f"\n{colorstr('green', 'bold', 'Training complete!')}\n")
        self.writer.close()


def main():
    print("\n" + "=" * 80)
    print("🎮 CLASH ROYALE POLICY TRAINING")
    print("=" * 80 + "\n")

    parser = argparse.ArgumentParser(description="Train offline RL policy")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="replay_data",
        help="Directory with replay files",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--sequence-length", type=int, default=16, help="Sequence length"
    )
    args = parser.parse_args()

    # Validate replay directory
    replay_path = Path(args.replay_dir)
    if not replay_path.exists():
        print(f"❌ ERROR: Replay directory does not exist: {replay_path.absolute()}")
        return

    # Check for replay files
    replay_files = list(replay_path.rglob("*.xz")) + list(replay_path.rglob("*.pkl"))
    print(f"Found {len(replay_files)} replay files in {replay_path}")

    if len(replay_files) == 0:
        print(f"❌ ERROR: No .xz or .pkl files found")
        print(f"\nDirectory contents:")
        for item in list(replay_path.iterdir())[:10]:
            print(f"  {item.name}")
        return

    # Create config
    config = TrainConfig()
    config.replay_dir = args.replay_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.sequence_length = args.sequence_length

    # Load dataset
    print(f"\n{colorstr('blue', 'bold', 'Loading dataset...')}\n")
    dataset_builder = DatasetBuilder(config.replay_dir, config.sequence_length)
    dataloader = dataset_builder.get_dataset(
        batch_size=config.batch_size, num_workers=config.num_workers
    )

    if dataloader is None:
        print(colorstr("red", "bold", "ERROR: No replay data found!"))
        print(f"Please add .pkl or .xz replay files to: {config.replay_dir}")
        return

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
