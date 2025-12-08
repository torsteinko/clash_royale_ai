"""
Dataset builder for offline reinforcement learning

Loads replay data and creates training batches
"""

import torch
import numpy as np
import lzma
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import pickle


def convert_katacd_state_to_tensor(state_dict: Dict) -> np.ndarray:
    """
    Convert KataCR state dict to fixed-size tensor

    KataCR state format:
    {
        'time': float,
        'unit_infos': list of unit dicts,
        'cards': list of card ids,
        'elixir': float
    }

    For now, just extract basic features that are always present
    """
    # Start with elixir (1 feature) - handle None/NaN
    elixir = state_dict.get("elixir", 0.0)
    if elixir is None or (isinstance(elixir, float) and np.isnan(elixir)):
        elixir = 0.0
    features = [float(elixir)]

    # Add time if available - handle None/NaN
    time = state_dict.get("time", 0.0)
    if time is None or (isinstance(time, float) and np.isnan(time)):
        time = 0.0
    features.append(float(time))

    # Add cards in hand (pad to 4 cards) - handle None/NaN
    cards = state_dict.get("cards", [])
    if cards is None:
        cards = []

    for i in range(4):
        if i < len(cards):
            card_val = cards[i]
            if card_val is None or (isinstance(card_val, float) and np.isnan(card_val)):
                card_val = 0
            features.append(float(card_val))
        else:
            features.append(0.0)

    # Total: 6 features [elixir, time, card1, card2, card3, card4]
    result = np.array(features, dtype=np.float32)

    # Final safety check
    if np.isnan(result).any():
        print(f"WARNING: NaN detected in state conversion!")
        print(f"  state_dict: {state_dict}")
        print(f"  features before conversion: {features}")
        # Replace NaN with 0
        result = np.nan_to_num(result, nan=0.0)

    return result


def convert_katacd_action_to_tensor(action_dict: Dict, state_dict: Dict) -> np.ndarray:
    """
    Convert KataCR action dict to [card_slot, pos_x, pos_y]

    KataCR action format:
    {
        'card_id': int (0-4, card slot index! 0=no action, 1-4=card slots),
        'xy': [x, y] or None
    }

    NOTE: card_id is ALREADY the slot index (0-4), not the card name index!
    """
    card_slot = action_dict.get("card_id", 0)  # Already 0-4

    # Handle None/NaN in card_slot
    if card_slot is None or (isinstance(card_slot, float) and np.isnan(card_slot)):
        card_slot = 0
    card_slot = float(card_slot)

    xy = action_dict.get("xy", None)

    # Check if xy is None or contains None values
    if xy is None:
        # No action / wait action
        return np.array([card_slot, 0.0, 0.0], dtype=np.float32)

    # Handle list/array case
    if isinstance(xy, (list, np.ndarray)):
        if len(xy) >= 2:
            x, y = xy[0], xy[1]
            # Handle None/NaN
            if x is None or (isinstance(x, float) and np.isnan(x)):
                x = 0.0
            if y is None or (isinstance(y, float) and np.isnan(y)):
                y = 0.0
            result = np.array([card_slot, float(x), float(y)], dtype=np.float32)

            # Final safety check
            if np.isnan(result).any():
                print(f"WARNING: NaN in action conversion!")
                print(f"  action_dict: {action_dict}")
                result = np.nan_to_num(result, nan=0.0)

            return result

    # Fallback
    return np.array([card_slot, 0.0, 0.0], dtype=np.float32)


class ReplayDataset(Dataset):
    """Dataset for offline RL training from replay buffers"""

    def __init__(self, replay_data: Dict, sequence_length: int = 16):
        """
        Args:
            replay_data: Dict with 'states', 'actions', 'rewards', 'terminals'
            sequence_length: Number of timesteps in each sequence
        """
        self.replay_data = replay_data
        self.sequence_length = sequence_length

        # Find valid sequence start indices
        self.valid_indices = self._find_valid_indices()

        print(f"Loaded {len(self.valid_indices)} valid sequences")

    def _find_valid_indices(self) -> List[int]:
        """Find valid sequences with balanced action/no-action ratio"""
        terminals = self.replay_data["terminals"]
        actions = self.replay_data["actions"]
        n_frames = len(terminals)

        action_sequences = []  # Sequences with actions
        no_action_sequences = []  # Sequences without actions

        episode_start = 0
        for i, is_terminal in enumerate(terminals):
            if is_terminal:
                for start_idx in range(episode_start, i - self.sequence_length + 2):
                    if start_idx >= 0:
                        # Count actions in this sequence
                        sequence_actions = actions[
                            start_idx : start_idx + self.sequence_length
                        ]
                        num_actions = sum(1 for a in sequence_actions if a[0] != 0)

                        if num_actions >= 2:
                            # Has meaningful actions
                            action_sequences.append(start_idx)
                        elif num_actions == 0:
                            # Pure waiting sequence
                            no_action_sequences.append(start_idx)
                        # Ignore sequences with only 1 action (mixed, less useful)

                episode_start = i + 1

        # Balance: Keep all action sequences, but only sample some no-action sequences
        # Aim for 50/50 or 70/30 ratio
        num_action_seq = len(action_sequences)
        num_no_action_to_keep = num_action_seq  # 50/50 balance

        if len(no_action_sequences) > num_no_action_to_keep:
            # Randomly sample no-action sequences
            import random

            no_action_sequences = random.sample(
                no_action_sequences, num_no_action_to_keep
            )

        valid = action_sequences + no_action_sequences
        random.shuffle(valid)  # Mix them up

        print(f"   Action sequences: {len(action_sequences)}")
        print(f"   No-action sequences: {len(no_action_sequences)}")
        print(f"   Total balanced dataset: {len(valid)} sequences")
        print(
            f"   Ratio: {len(action_sequences)/len(valid)*100:.1f}% action, {len(no_action_sequences)/len(valid)*100:.1f}% waiting"
        )

        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a sequence of (state, action, reward, return-to-go)"""
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        # Extract sequence
        states_raw = self.replay_data["states"][start_idx:end_idx]
        actions_raw = self.replay_data["actions"][start_idx:end_idx]
        rewards = self.replay_data["rewards"][start_idx:end_idx]

        # Convert to numpy arrays with consistent shapes
        # states_raw and actions_raw are already numpy arrays from conversion
        try:
            states = np.stack(states_raw)  # (T, state_dim)
            actions = np.stack(actions_raw)  # (T, 3) for [card_slot, x, y]
        except Exception as e:
            print(f"Error stacking at idx {idx}, start {start_idx}, end {end_idx}")
            print(f"  states_raw len: {len(states_raw)}")
            print(f"  actions_raw len: {len(actions_raw)}")
            print(f"  Error: {e}")
            # Return dummy data to avoid crash
            states = np.zeros((self.sequence_length, 6), dtype=np.float32)
            actions = np.zeros((self.sequence_length, 3), dtype=np.float32)

        # Ensure rewards is numpy array
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards, dtype=np.float32)

        # Calculate return-to-go
        rtg = np.zeros(len(rewards), dtype=np.float32)
        rtg[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            rtg[i] = rewards[i] + rtg[i + 1]

        # Prepare batch
        batch = {
            "states": states.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "rtg": rtg,
            "timesteps": np.arange(start_idx, end_idx, dtype=np.int32),
        }

        return batch


class DatasetBuilder:
    """Build training dataset from replay files"""

    def __init__(self, replay_dir: str, sequence_length: int = 16):
        """
        Args:
            replay_dir: Directory containing .pkl or .xz replay files
            sequence_length: Length of training sequences
        """
        self.replay_dir = Path(replay_dir)
        self.sequence_length = sequence_length
        self.replay_data = None
        self._load_replays()

    def _load_replays(self):
        """Load all replay files and concatenate"""
        print(f"Loading replays from {self.replay_dir}")

        # Find all replay files recursively
        replay_files = list(self.replay_dir.rglob("*.pkl"))
        replay_files += list(self.replay_dir.rglob("*.xz"))

        if not replay_files:
            print(f"❌ No replay files found in {self.replay_dir}")
            print(f"   Looking for: *.pkl or *.xz files")
            print(f"   Directory contents:")
            try:
                for item in list(self.replay_dir.iterdir())[:10]:
                    print(f"     - {item.name}")
            except:
                print(f"     (Could not list directory)")

            self.replay_data = {
                "states": [],
                "actions": [],
                "rewards": [],
                "terminals": [],
            }
            return

        print(f"Found {len(replay_files)} replay files")

        all_states = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        files_processed = 0

        for replay_file in tqdm(replay_files, desc="Loading replays"):
            try:
                if replay_file.suffix == ".xz":
                    # Load compressed replay - try numpy first (KataCR format), then pickle
                    with lzma.open(replay_file, "rb") as f:
                        try:
                            # Try loading as numpy array first (KataCR format)
                            data = np.load(f, allow_pickle=True).item()
                        except:
                            # Fall back to pickle
                            f.seek(0)
                            data = pickle.load(f)
                else:
                    # Load pickle directly
                    with open(replay_file, "rb") as f:
                        data = pickle.load(f)

                # Handle KataCR format: {'state', 'action', 'reward'} -> our format
                if "state" in data and "states" not in data:
                    # Convert KataCR format to our format
                    raw_states = data.get("state", [])
                    raw_actions = data.get("action", [])
                    raw_rewards = data.get("reward", [])

                    # Convert KataCR dicts to tensors
                    states = [convert_katacd_state_to_tensor(s) for s in raw_states]
                    # Pass state to action converter to map card_id to card_slot
                    actions = [
                        convert_katacd_action_to_tensor(a, s)
                        for a, s in zip(raw_actions, raw_states)
                    ]

                    converted_data = {
                        "states": states,
                        "actions": actions,
                        "rewards": raw_rewards,
                        "terminals": [False] * (len(states) - 1)
                        + [True],  # Last frame is terminal
                    }
                    data = converted_data

                    # In _load_replays(), after creating converted_data, ADD:
                    if files_processed == 0:  # Only print for first file
                        print(f"\n🔍 DEBUG: First replay file rewards:")
                        print(f"   File: {replay_file.name}")
                        print(f"   Total frames: {len(raw_rewards)}")
                        print(f"   Reward stats:")
                        reward_arr = np.array(raw_rewards)
                        print(f"     Min: {reward_arr.min()}")
                        print(f"     Max: {reward_arr.max()}")
                        print(f"     Mean: {reward_arr.mean()}")
                        print(f"     Non-zero: {(reward_arr != 0).sum()}")
                        print(f"   First 10 rewards: {reward_arr[:10]}")
                        print(f"   Last 10 rewards: {reward_arr[-10:]}")

                        # Check unique values
                        unique_rewards = np.unique(reward_arr)
                        print(f"   Unique reward values: {unique_rewards}")

                # Validate data structure
                if not all(
                    k in data for k in ["states", "actions", "rewards", "terminals"]
                ):
                    print(
                        f"Skipping {replay_file.name}: missing required keys (has: {list(data.keys())})"
                    )
                    continue

                # Skip very short episodes
                if len(data["states"]) < self.sequence_length:
                    # Don't print for every short episode, too noisy
                    continue

                # Verify states/actions are the right type
                if len(data["states"]) > 0:
                    if not isinstance(data["states"][0], np.ndarray):
                        print(f"Skipping {replay_file.name}: states not numpy arrays")
                        continue

                all_states.extend(data["states"])
                all_actions.extend(data["actions"])
                all_rewards.extend(data["rewards"])
                all_terminals.extend(data["terminals"])
                files_processed += 1

            except Exception as e:
                print(f"Error loading {replay_file.name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        self.replay_data = {
            "states": all_states,
            "actions": all_actions,
            "rewards": (
                np.array(all_rewards, dtype=np.float32)
                if all_rewards
                else np.array([], dtype=np.float32)
            ),
            "terminals": (
                np.array(all_terminals, dtype=bool)
                if all_terminals
                else np.array([], dtype=bool)
            ),
        }

        if len(all_states) > 0:
            print(f"\n✅ Successfully loaded data:")
            print(
                f"   {len(all_states)} frames from {files_processed}/{len(replay_files)} files"
            )
            print(f"   Total episodes: {self.replay_data['terminals'].sum()}")
            print(f"   Mean reward: {self.replay_data['rewards'].mean():.2f}")
            print(f"   State shape: {all_states[0].shape if all_states else 'N/A'}")
            print(f"   Action shape: {all_actions[0].shape if all_actions else 'N/A'}")
        else:
            print(
                f"\n❌ WARNING: No valid replay data loaded from {len(replay_files)} files"
            )
            print(f"   Files processed: {files_processed}")
            print(
                "   Check that replay files contain 'state'/'states', 'action'/'actions', 'reward'/'rewards' keys"
            )

    def get_dataset(
        self, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True
    ) -> DataLoader:
        """
        Create DataLoader for training

        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        if not self.replay_data["states"]:
            print("WARNING: Empty replay data, cannot create dataset")
            return None

        dataset = ReplayDataset(self.replay_data, self.sequence_length)

        if len(dataset) == 0:
            print("WARNING: No valid sequences in dataset")
            return None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            drop_last=True,  # Drop incomplete batches to ensure consistent batch sizes
        )


def collate_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for batching sequences"""
    try:
        # Filter out any None items
        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
            return None

        # Check that all items have the same sequence length
        seq_lengths = [item["states"].shape[0] for item in batch]
        if len(set(seq_lengths)) > 1:
            # Filter to only keep items with the most common sequence length
            from collections import Counter

            most_common_length = Counter(seq_lengths).most_common(1)[0][0]
            batch = [
                item for item in batch if item["states"].shape[0] == most_common_length
            ]
            print(
                f"Warning: Filtered batch to {len(batch)} items with seq_len={most_common_length}"
            )

        if len(batch) == 0:
            return None

        # Stack all sequences - they should all be numpy arrays of same shape
        batched = {
            "states": torch.FloatTensor(
                np.stack([item["states"] for item in batch])
            ),  # (B, T, state_dim)
            "actions": torch.FloatTensor(
                np.stack([item["actions"] for item in batch])
            ),  # (B, T, 3)
            "rewards": torch.FloatTensor(
                np.stack([item["rewards"] for item in batch])
            ),  # (B, T)
            "rtg": torch.FloatTensor(
                np.stack([item["rtg"] for item in batch])
            ),  # (B, T)
            "timesteps": torch.LongTensor(
                np.stack([item["timesteps"] for item in batch])
            ),  # (B, T)
        }
        return batched
    except Exception as e:
        print(f"Error in collate_batch: {e}")
        print(f"Batch length: {len(batch)}")
        if batch:
            print(f"First item keys: {batch[0].keys()}")
            for k in batch[0].keys():
                v = batch[0][k]
                print(
                    f"  {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}"
                )
        raise


if __name__ == "__main__":
    # Test dataset builder
    print("Testing dataset builder...")
    builder = DatasetBuilder("replay_data", sequence_length=16)

    if builder.replay_data and builder.replay_data["states"]:
        print("\nCreating dataloader...")
        dataloader = builder.get_dataset(batch_size=8, num_workers=0)

        if dataloader:
            print("Testing one batch...")
            for batch in dataloader:
                print("\n✅ Batch loaded successfully!")
                print("Batch keys:", batch.keys())
                print("Batch shapes:")
                for k, v in batch.items():
                    print(f"  {k}: {v.shape}")
                break
        else:
            print("❌ Failed to create dataloader")
    else:
        print("❌ No data to test")
