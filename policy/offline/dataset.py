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
        """Find all valid sequence start indices"""
        terminals = self.replay_data["terminals"]
        n_frames = len(terminals)
        valid = []

        episode_start = 0
        for i, is_terminal in enumerate(terminals):
            if is_terminal:
                # Add all valid sequences from this episode
                for start_idx in range(episode_start, i - self.sequence_length + 2):
                    if start_idx >= 0:
                        valid.append(start_idx)
                episode_start = i + 1

        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a sequence of (state, action, reward, return-to-go)"""
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        # Extract sequence
        states = self.replay_data["states"][start_idx:end_idx]
        actions = self.replay_data["actions"][start_idx:end_idx]
        rewards = self.replay_data["rewards"][start_idx:end_idx]

        # Calculate return-to-go
        rtg = np.zeros(len(rewards), dtype=np.float32)
        rtg[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            rtg[i] = rewards[i] + rtg[i + 1]

        # Prepare batch
        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
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

        # Find all replay files
        replay_files = list(self.replay_dir.glob("*.pkl"))
        replay_files += list(self.replay_dir.glob("*.xz"))

        if not replay_files:
            print(f"No replay files found in {self.replay_dir}")
            print("Creating empty replay structure")
            self.replay_data = {
                "states": [],
                "actions": [],
                "rewards": [],
                "terminals": [],
            }
            return

        all_states = []
        all_actions = []
        all_rewards = []
        all_terminals = []

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
                    # KataCR uses 'state', 'action', 'reward' (singular)
                    # We use 'states', 'actions', 'rewards', 'terminals' (plural)
                    converted_data = {
                        "states": data.get("state", []),
                        "actions": data.get("action", []),
                        "rewards": data.get("reward", []),
                        "terminals": [False] * (len(data.get("state", [])) - 1)
                        + [True],  # Last frame is terminal
                    }
                    data = converted_data

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
                    print(
                        f"Skipping {replay_file.name}: too short ({len(data['states'])} frames)"
                    )
                    continue

                all_states.extend(data["states"])
                all_actions.extend(data["actions"])
                all_rewards.extend(data["rewards"])
                all_terminals.extend(data["terminals"])

            except Exception as e:
                print(f"Error loading {replay_file.name}: {e}")
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
            print(f"Loaded {len(all_states)} frames from {len(replay_files)} replays")
            print(f"Total episodes: {self.replay_data['terminals'].sum()}")
            print(f"Mean reward: {self.replay_data['rewards'].mean():.2f}")
        else:
            print(
                f"WARNING: No valid replay data loaded from {len(replay_files)} files"
            )
            print(
                "Check that replay files contain 'state'/'states', 'action'/'actions', 'reward'/'rewards' keys"
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
            print("WARNING: Empty replay data, creating dummy dataset")
            return None

        dataset = ReplayDataset(self.replay_data, self.sequence_length)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def collate_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for batching sequences"""
    # Stack all sequences
    batched = {
        "states": [item["states"] for item in batch],
        "actions": [item["actions"] for item in batch],
        "rewards": torch.FloatTensor([item["rewards"] for item in batch]),
        "rtg": torch.FloatTensor([item["rtg"] for item in batch]),
        "timesteps": torch.LongTensor([item["timesteps"] for item in batch]),
    }

    return batched


if __name__ == "__main__":
    # Test dataset builder
    builder = DatasetBuilder("replay_data", sequence_length=16)

    if builder.replay_data and builder.replay_data["states"]:
        dataloader = builder.get_dataset(batch_size=8)

        # Test one batch
        for batch in dataloader:
            print("Batch keys:", batch.keys())
            print("States:", len(batch["states"]))
            print("Rewards shape:", batch["rewards"].shape)
            print("RTG shape:", batch["rtg"].shape)
            break
    else:
        print("No data to test")
