"""
Training script for offline reinforcement learning
WITH CRITICAL FIXES FOR NO-ACTION DOMINANCE
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
import numpy as np

from policy.offline.models.policy_transformer import PolicyTransformer
from policy.offline.dataset import DatasetBuilder
from policy.utils import colorstr


class TrainConfig:
    """Training configuration"""

    def __init__(self):
        # Model - WILL BE SET DYNAMICALLY from dataset
        self.num_cards = 114  # Default (updated after loading dataset)
        self.num_troops = 200
        self.d_model = 256
        self.n_head = 8
        self.n_layers = 4
        self.d_ff = 1024
        self.dropout = 0.1
        self.sequence_length = 16
        self.arena_grid_size = (32, 18)

        # Update state dimension
        self.state_dim = 126

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
        self.save_freq = 4  # How many epochs between saves
        self.log_freq = 100


class Trainer:
    """Trainer for offline RL policy"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")  # Force CPU for compatibility
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
            state_dim=config.state_dim,
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

        # Loss functions (removed - we'll use functional API with masking)
        # self.card_criterion = nn.CrossEntropyLoss()
        # self.position_criterion = nn.CrossEntropyLoss()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Create checkpoint dir
        self.checkpoint_dir = config.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, dataloader):
        """
        Train for one epoch

        CRITICAL FIXES IMPLEMENTED:
        1. Action sequence shifting (teacher forcing)
        2. Loss masking (only compute loss on action frames)
        3. Proper mask handling
        """
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

                    print("CARD DISTRIBUTION ANALYSIS")
                    print("=" * 60)

                    # Get card distribution across entire batch
                    card_ids_flat = actions[:, :, 0].flatten().cpu().numpy()
                    unique_cards, counts = np.unique(card_ids_flat, return_counts=True)
                    print(f"Card ID distribution in this batch:")
                    for card_id, count in zip(unique_cards, counts):
                        percentage = (count / len(card_ids_flat)) * 100
                        print(
                            f"  Card {int(card_id)}: {count:4d} times ({percentage:5.1f}%)"
                        )

                    # Check if cards are mostly zeros (no-action)
                    zero_actions = (card_ids_flat == 0).sum()
                    print(
                        f"\nZero actions (no-action): {zero_actions}/{len(card_ids_flat)} ({100*zero_actions/len(card_ids_flat):.1f}%)"
                    )
                    print("=" * 60 + "\n")

                # ==================================================================
                # CRITICAL FIX #1: ACTION SHIFTING FOR TEACHER FORCING
                # Model predicts NEXT action, so shift inputs right
                # ==================================================================
                actions_input = torch.cat(
                    [
                        torch.zeros(B, 1, 3, device=self.device),  # Prepend zero action
                        actions[:, :-1, :],  # Shift everything right
                    ],
                    dim=1,
                )

                # Actions are the targets (what we want to predict)
                actions_target = actions

                # ==================================================================
                # CARD TARGET PROCESSING - CARD NAME MODE
                # ==================================================================
                # actions_target[:, :, 0] now contains CARD NAME IDs (not slot indices!)
                # These are the actual card IDs from the game (e.g., 5=knight, 12=arrows)

                target_cards = actions_target[:, :, 0].long()  # Already card IDs

                # Validate - make sure within vocabulary
                if (
                    target_cards.min() < 0
                    or target_cards.max() >= self.config.num_cards
                ):
                    print(f"❌ [Batch {batch_idx}] Invalid target_cards!")
                    print(
                        f"  target_cards range: [{target_cards.min()}, {target_cards.max()}]"
                    )
                    print(f"  Expected range: [0, {self.config.num_cards-1}]")
                    print(
                        f"  target_cards unique: {torch.unique(target_cards).tolist()}"
                    )
                    print(f"  Skipping batch")
                    continue

                # No conversion needed! Actions already contain card name IDs
                # The rest of the code (position processing, masking, etc.) stays the same

                # ==================================================================
                # POSITION TARGET PROCESSING
                # ==================================================================
                grid_h, grid_w = self.config.arena_grid_size  # (32, 18)
                pos_x_raw = actions_target[:, :, 1]
                pos_y_raw = actions_target[:, :, 2]

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
                    print(f"⚠️ Invalid position targets at batch {batch_idx}")
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
                    print(f"⚠️ [Batch {batch_idx}] NaN in states, skipping")
                    continue
                if torch.isnan(actions_input).any():
                    print(f"⚠️ [Batch {batch_idx}] NaN in actions_input, skipping")
                    continue
                if torch.isnan(rtg).any():
                    print(f"⚠️ [Batch {batch_idx}] NaN in rtg, skipping")
                    continue
                if torch.isnan(timesteps).any():
                    print(f"⚠️ [Batch {batch_idx}] NaN in timesteps, skipping")
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

                # CRITICAL: Use shifted actions as input
                card_logits, position_logits = self.model(
                    states, actions_input, rtg, timesteps
                )

                # Check outputs immediately
                if torch.isnan(card_logits).any():
                    print(f"❌ [Batch {batch_idx}] Model produced NaN in card_logits!")
                    print(f"  Input stats:")
                    print(
                        f"    states: [{states.min():.2f}, {states.max():.2f}], mean={states.mean():.2f}"
                    )
                    print(
                        f"    actions: [{actions_input.min():.2f}, {actions_input.max():.2f}], mean={actions_input.mean():.2f}"
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
                # MASKED LOSS (ONLY ON ACTION FRAMES)
                # ==================================================================

                # ==================================================================
                # LOSS CALCULATION WITH CRITICAL FIXES
                # ==================================================================

                # Ensure shapes match before flattening
                if len(card_logits.shape) != 3 or len(target_cards.shape) != 2:
                    print(f"⚠️ Unexpected tensor dimensions at batch {batch_idx}")
                    print(f"  card_logits: {card_logits.shape} (expected 3D)")
                    print(f"  target_cards: {target_cards.shape} (expected 2D)")
                    print(f"  Skipping batch")
                    continue

                B_logits, T_logits, _ = card_logits.shape
                B_targets, T_targets = target_cards.shape

                if B_logits != B_targets or T_logits != T_targets:
                    print(f"⚠️ Shape mismatch at batch {batch_idx}")
                    print(
                        f"  Logits: {card_logits.shape}, Targets: {target_cards.shape}"
                    )
                    print(f"  Skipping batch")
                    continue

                # Reshape using actual dimensions from tensors
                num_card_classes = card_logits.shape[-1]  # Should be 4
                card_logits_flat = card_logits.reshape(-1, num_card_classes)
                target_cards_flat = target_cards.reshape(-1)

                position_logits_flat = position_logits.reshape(-1, grid_h * grid_w)
                target_positions_flat = target_positions.reshape(-1)

                # Final shape validation before loss
                if card_logits_flat.shape[0] != target_cards_flat.shape[0]:
                    print(f"⚠️ Flattened shape mismatch at batch {batch_idx}")
                    print(
                        f"  card_logits: {card_logits.shape} -> flat: {card_logits_flat.shape}"
                    )
                    print(
                        f"  target_cards: {target_cards.shape} -> flat: {target_cards_flat.shape}"
                    )
                    print(f"  Skipping batch")
                    continue

                # ==================================================================
                # CRITICAL FIX #2: MASKED LOSS (ONLY ON ACTION FRAMES)
                # This is the KEY fix - only compute loss where actions happen!
                # ==================================================================

                # Create mask: 1 for action frames, 0 for no-action frames
                # actions_target[:, :, 0] is card_id: 0=no action, >0=action
                mask = (actions_target[:, :, 0] > 0).float().reshape(-1)  # (B*T,)

                # Count valid actions
                num_actions = mask.sum()

                if num_actions > 0:
                    # Card loss - ONLY on action frames
                    card_loss_unreduced = nn.functional.cross_entropy(
                        card_logits_flat, target_cards_flat, reduction="none"
                    )
                    card_loss = (card_loss_unreduced * mask).sum() / (
                        num_actions + 1e-6
                    )

                    # Position loss - ONLY on action frames
                    position_loss_unreduced = nn.functional.cross_entropy(
                        position_logits_flat, target_positions_flat, reduction="none"
                    )
                    position_loss = (position_loss_unreduced * mask).sum() / (
                        num_actions + 1e-6
                    )
                else:
                    # No actions in this batch - zero loss
                    card_loss = torch.tensor(0.0, device=self.device)
                    position_loss = torch.tensor(0.0, device=self.device)

                # Check for nan/inf with detailed debugging
                if torch.isnan(card_loss) or torch.isinf(card_loss):
                    print(f"\n❌ [Batch {batch_idx}] NaN/Inf card_loss detected!")
                    print(f"  card_loss value: {card_loss}")
                    print(f"  num_actions: {num_actions}")
                    print(f"  card_logits_flat stats:")
                    print(f"    shape: {card_logits_flat.shape}")
                    print(
                        f"    range: [{card_logits_flat.min():.4f}, {card_logits_flat.max():.4f}]"
                    )
                    print(
                        f"    mean: {card_logits_flat.mean():.4f}, std: {card_logits_flat.std():.4f}"
                    )
                    print(f"  Skipping batch")
                    continue

                if torch.isnan(position_loss) or torch.isinf(position_loss):
                    print(f"⚠️ NaN/Inf position_loss at batch {batch_idx}, skipping")
                    continue

                # Combined loss
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
                        print(f"⚠️ [Batch {batch_idx}] NaN gradient in '{name}'")
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"⚠️ [Batch {batch_idx}] Skipping update due to NaN gradients")
                    self.optimizer.zero_grad()  # Clear bad gradients
                    continue

                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # Log extreme gradient norms (only if very large)
                if grad_norm > 100.0:
                    print(
                        f"⚠️  [Batch {batch_idx}] Large gradient norm: {grad_norm:.2f}"
                    )

                self.optimizer.step()
                self.scheduler.step()

                # ==================================================================
                # METRICS (computed on action frames only for accuracy)
                # ==================================================================
                with torch.no_grad():
                    if num_actions > 0:
                        # Card accuracy on action frames only
                        card_pred = card_logits.argmax(dim=-1).reshape(-1)
                        card_acc = (
                            (card_pred == target_cards_flat).float() * mask
                        ).sum() / (num_actions + 1e-6)

                        # Position accuracy on action frames only
                        position_pred = position_logits.argmax(dim=-1).reshape(-1)
                        pos_acc = (
                            (position_pred == target_positions_flat).float() * mask
                        ).sum() / (num_actions + 1e-6)
                        # Check prediction distribution to catch "always predict X" collapse
                        if batch_idx % 500 == 0:  # Every 500 batches
                            action_mask_bool = mask.bool()
                            card_pred_on_actions = card_pred[action_mask_bool]
                            target_cards_on_actions = target_cards_flat[
                                action_mask_bool
                            ]

                            print(
                                f"\n📊 [Batch {batch_idx}] Prediction Analysis on ACTION frames:"
                            )
                            print(
                                f"   Target distribution: {torch.bincount(target_cards_on_actions, minlength=self.config.num_cards).tolist()}"
                            )
                            print(
                                f"   Prediction distribution: {torch.bincount(card_pred_on_actions, minlength=self.config.num_cards).tolist()}"
                            )
                            print(f"   Card accuracy: {card_acc.item():.3f}")
                            print(f"   Num actions in batch: {num_actions.item()}")
                    else:
                        card_acc = torch.tensor(0.0, device=self.device)
                        pos_acc = torch.tensor(0.0, device=self.device)

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
                print(f"⚠️ Error in batch {batch_idx}: {e}")
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

        avg_loss = total_loss / valid_batches
        avg_card_loss = total_card_loss / valid_batches
        avg_pos_loss = total_pos_loss / valid_batches
        avg_card_acc = total_card_acc / valid_batches
        avg_pos_acc = total_pos_acc / valid_batches

        # Check for NaN/Inf in metrics (training diverged)
        if not np.isfinite(avg_loss):
            print("❌ Training diverged (loss is NaN/Inf)")
            print("   This usually means:")
            print("   - Learning rate is too high")
            print("   - Gradient explosion occurred")
            print("   - Data has extreme outliers")
            return {
                "loss": float("inf"),
                "card_loss": float("inf"),
                "pos_loss": float("inf"),
                "card_acc": 0.0,
                "pos_acc": 0.0,
            }

        return {
            "loss": avg_loss,
            "card_loss": avg_card_loss,
            "pos_loss": avg_pos_loss,
            "card_acc": avg_card_acc,
            "pos_acc": avg_pos_acc,
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
        print(f"💾 Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"\n🔄 Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Restore model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded model state")

        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✓ Loaded optimizer state")

        # Restore scheduler
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"✓ Loaded scheduler state")

        # Restore training state
        self.global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)

        print(f"✓ Resuming from epoch {start_epoch}, step {self.global_step}")

        return start_epoch

    def train(self, dataloader, start_epoch=0):
        """Main training loop"""
        print(f"\n{colorstr('green', 'bold', 'Starting training...')}\n")

        self.best_loss = float("inf")
        patience = 10  # Stop if no improvement for 10 epochs
        patience_counter = 0

        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch

            # Train one epoch
            metrics = self.train_epoch(dataloader)

            # Stop training if epoch returned inf loss (diverged)
            if metrics["loss"] == float("inf"):
                print(f"\n❌ Training stopped at epoch {epoch + 1} due to divergence")
                break

            # Log epoch metrics
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch + 1}/{self.config.num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"   Loss: {metrics['loss']:.4f}")
            print(f"   Card Loss: {metrics['card_loss']:.4f}")
            print(f"   Position Loss: {metrics['pos_loss']:.4f}")
            print(f"   Card Accuracy: {metrics['card_acc']:.3f}")
            print(f"   Position Accuracy: {metrics['pos_acc']:.3f}")
            print(f"{'='*60}\n")

            self.writer.add_scalar("epoch/loss", metrics["loss"], epoch)
            self.writer.add_scalar("epoch/card_accuracy", metrics["card_acc"], epoch)
            self.writer.add_scalar("epoch/position_accuracy", metrics["pos_acc"], epoch)

            # Save best model
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                patience_counter = 0  # Reset patience

                # Save best model checkpoint
                best_checkpoint = {
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config.__dict__,
                    "loss": self.best_loss,
                    "metrics": metrics,
                }
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(best_checkpoint, best_path)
                print(f"   💾 New best model saved! Loss: {self.best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   📊 No improvement for {patience_counter} epoch(s)")

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered!")
                print(f"   No improvement for {patience} consecutive epochs")
                print(f"   Best loss: {self.best_loss:.4f}")
                break

            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch + 1)

        # Save final checkpoint
        self.save_checkpoint(self.config.num_epochs)

        print(f"\n{colorstr('green', 'bold', '✅ Training complete!')}\n")
        print(f"Best loss achieved: {self.best_loss:.4f}")
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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

    # Update num_cards based on actual dataset vocabulary
    if hasattr(dataset_builder, "card_vocab_size") and dataset_builder.card_vocab_size:
        config.num_cards = dataset_builder.card_vocab_size
        print(
            f"\n✅ Updated num_cards to {config.num_cards} based on dataset vocabulary"
        )
    else:
        print(f"\n⚠️  Using default num_cards={config.num_cards}")

    dataloader = dataset_builder.get_dataset(
        batch_size=config.batch_size, num_workers=config.num_workers
    )

    if dataloader is None:
        print(colorstr("red", "bold", "ERROR: No replay data found!"))
        print(f"Please add .pkl or .xz replay files to: {config.replay_dir}")
        return

    # Create trainer and train
    trainer = Trainer(config)
    start_epoch = 0
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            start_epoch = trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"❌ ERROR: Checkpoint file not found: {checkpoint_path}")
            print("Starting training from scratch instead.")

    trainer.train(dataloader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
