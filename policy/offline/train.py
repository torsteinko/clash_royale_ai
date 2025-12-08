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
        # Model
        self.num_cards = 108
        self.num_troops = 200
        self.d_model = 256
        self.n_head = 8
        self.n_layers = 4
        self.d_ff = 1024
        self.dropout = 0.1
        self.sequence_length = 16
        self.arena_grid_size = (32, 18)

        # Training
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_grad_norm = 1.0

        # Data
        self.replay_dir = "replay_data"
        self.num_workers = 4

        # Logging
        self.log_dir = (
            Path("runs") / "policy_training" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.save_freq = 5  # Save checkpoint every N epochs
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

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rtg = batch["rtg"].to(self.device)
            timesteps = batch["timesteps"].to(self.device)
            rewards = batch["rewards"].to(self.device)

            # For now, create dummy actions and targets
            # In practice, you'd extract these from the batch
            B, T = rtg.shape
            actions = torch.zeros(B, T, 2, dtype=torch.long, device=self.device)
            target_cards = torch.randint(0, 4, (B, T), device=self.device)
            target_positions = torch.randint(0, 32 * 18, (B, T), device=self.device)

            # Forward pass
            card_logits, position_logits = self.model(
                batch["states"], actions, rtg, timesteps
            )

            # Calculate losses
            card_loss = self.card_criterion(
                card_logits.reshape(-1, 4), target_cards.reshape(-1)
            )

            position_loss = self.position_criterion(
                position_logits.reshape(-1, 32 * 18), target_positions.reshape(-1)
            )

            loss = card_loss + position_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            # Calculate accuracy
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

        # Return average metrics
        n_batches = len(dataloader)
        return {
            "loss": total_loss / n_batches,
            "card_loss": total_card_loss / n_batches,
            "pos_loss": total_pos_loss / n_batches,
            "card_acc": total_card_acc / n_batches,
            "pos_acc": total_pos_acc / n_batches,
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
