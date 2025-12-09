"""
Dataset builder for offline reinforcement learning
Loads replay data and creates training batches
"""

import torch
import numpy as np
import lzma
import random
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional
import pickle


def convert_katacd_action_to_tensor(action_dict: Dict, state_dict: Dict) -> np.ndarray:
    """
    Convert KataCR action dict to [card_name_id, pos_x, pos_y]

    CARD NAME PREDICTION MODE (for multi-deck training):
    - Maps slot index (1-4) to actual card ID from state['cards']
    - card_id in action_dict is slot index: 0=no action, 1-4=slots
    - state['cards'] contains actual card IDs (e.g., [5, 12, 23, 34] for knight, arrows, etc.)
    """
    card_slot = action_dict.get("card_id", 0)  # Slot index: 0-4

    # Handle None/NaN in card_slot
    if card_slot is None or (isinstance(card_slot, float) and np.isnan(card_slot)):
        card_slot = 0
    card_slot = int(card_slot)

    # Map slot to actual card name ID
    card_name_id = 0  # Default: no action

    if card_slot > 0:  # If actually playing a card (not waiting)
        # Get cards in hand from state
        cards = state_dict.get("cards", [])
        if cards is None:
            cards = []

        # card_slot is 1-indexed (1=first slot, 2=second slot, etc.)
        # Convert to 0-indexed for array access
        slot_idx = card_slot - 1

        if 0 <= slot_idx < len(cards):
            card_name_id = cards[slot_idx]  # Get actual card ID from hand

            # Handle None/NaN in card ID
            if card_name_id is None or (
                isinstance(card_name_id, float) and np.isnan(card_name_id)
            ):
                card_name_id = 0
            else:
                card_name_id = int(card_name_id)

    # Get position
    xy = action_dict.get("xy", None)

    if xy is None:
        return np.array([card_name_id, 0.0, 0.0], dtype=np.float32)

    if isinstance(xy, (list, np.ndarray)) and len(xy) >= 2:
        x, y = xy[0], xy[1]

        # Handle None/NaN
        if x is None or (isinstance(x, float) and np.isnan(x)):
            x = 0.0
        if y is None or (isinstance(y, float) and np.isnan(y)):
            y = 0.0

        result = np.array([card_name_id, float(x), float(y)], dtype=np.float32)

        # Final safety check
        if np.isnan(result).any():
            print(f"WARNING: NaN in action conversion!")
            print(f"  action_dict: {action_dict}")
            print(f"  state cards: {state_dict.get('cards', [])}")
            result = np.nan_to_num(result, nan=0.0)

        return result

    return np.array([card_name_id, 0.0, 0.0], dtype=np.float32)


def convert_katacd_state_to_tensor(state_dict: Dict) -> np.ndarray:
    """
    Convert KataCR state dict to fixed-size tensor

    KataCR state format:
        'time': float,
        'unit_infos': list of unit dicts,
        'cards': list of card ids,
        'elixir': float

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

        # Find valid sequence start indices AND calculate sample weights
        self.valid_indices, self.sample_weights = self._find_valid_indices()

        print(f"Loaded {len(self.valid_indices)} valid sequences")

    def _find_valid_indices(self) -> tuple:
        """
        Find sequences using KataCR's weighted sampling approach

        CRITICAL FIX #1: Weight sequences by whether END frame has action
        CRITICAL FIX #2: Aggressive reweighting with context propagation
        """
        terminals = self.replay_data["terminals"]
        actions = self.replay_data["actions"]

        valid_sequences = []
        sample_weights = []

        episode_start = 0

        for i, is_terminal in enumerate(terminals):
            if is_terminal:
                # Find all valid sequence start indices in this episode
                for start_idx in range(episode_start, i - self.sequence_length + 2):
                    if start_idx >= 0:
                        end_idx = start_idx + self.sequence_length - 1
                        valid_sequences.append(start_idx)

                        # Weight = 1 if action at end frame, 0 if waiting
                        has_action = actions[end_idx][0] > 0
                        sample_weights.append(1.0 if has_action else 0.0)

                episode_start = i + 1

        sample_weights = np.array(sample_weights, dtype=np.float32)

        # Calculate action ratio
        action_ratio = (
            sample_weights.sum() / len(sample_weights) if len(sample_weights) > 0 else 0
        )

        print(f"\n📊 Dataset Statistics:")
        print(f"   Total sequences: {len(valid_sequences)}")
        print(f"   Action sequences: {int(sample_weights.sum())}")
        print(f"   Action ratio: {action_ratio*100:.1f}%")

        if action_ratio == 0:
            print("   ⚠️  WARNING: No action sequences found!")
            return valid_sequences, sample_weights

        # CRITICAL FIX: Reweight using KataCR's formula (aggressive)
        reweighted = sample_weights * (1 / action_ratio) + (1 - sample_weights) * (
            1 / (1 - action_ratio)
        )

        # CRITICAL FIX: Propagate weight to nearby frames (context matters!)
        for i in np.where(sample_weights > 0)[0]:
            for j in range(i, min(i + self.sequence_length, len(reweighted))):
                if j >= len(valid_sequences):
                    break

                seq_end_idx = valid_sequences[j] + self.sequence_length - 1
                if seq_end_idx < len(terminals) and terminals[seq_end_idx]:
                    break

                alpha = 1.0 / (j - i + 1)
                reweighted[j] = max(reweighted[j], alpha * (1 / action_ratio))

        print(
            f"   Sampling ratio: {1/action_ratio:.1f}x actions : {1/(1-action_ratio):.1f}x waiting"
        )
        print(f"   Max sample weight: {reweighted.max():.2f}")
        print(f"   Min sample weight: {reweighted.min():.2f}")

        return valid_sequences, reweighted

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
        """
        Load all replay files and concatenate

        CRITICAL FIX #3: Clip episodes to last action frame
        """
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

                    # CRITICAL FIX: Find last action frame and clip episode
                    last_action_idx = len(raw_actions) - 1
                    for idx in range(len(raw_actions) - 1, -1, -1):
                        if raw_actions[idx].get("card_id", 0) != 0:
                            last_action_idx = idx
                            break

                    # Clip to last action + 1
                    raw_states = raw_states[: last_action_idx + 1]
                    raw_actions = raw_actions[: last_action_idx + 1]
                    raw_rewards = raw_rewards[: last_action_idx + 1]

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

            print(f"\n🎴 CARD VOCABULARY ANALYSIS:")
            print("=" * 60)

            # Extract unique card IDs from states (cards in hand)
            cards_in_hand = set()
            for state in all_states:
                # state vector: [elixir, time, card1, card2, card3, card4]
                # indices 2-5 are the 4 cards in hand
                for card_idx in range(2, 6):
                    card_id = int(state[card_idx])
                    if card_id > 0:  # Skip empty slots (0)
                        cards_in_hand.add(card_id)

            # Extract unique card IDs from actions (cards played)
            cards_played = set()
            for action in all_actions:
                card_id = int(action[0])  # First element is now card_name_id
                if card_id > 0:  # Skip "no action" (0)
                    cards_played.add(card_id)

            # Combine both sets
            all_playable_cards = cards_in_hand | cards_played

            print(f"   Cards found in hand: {sorted(cards_in_hand)}")
            print(f"   Cards found in actions: {sorted(cards_played)}")
            print(f"   Total unique playable cards: {len(all_playable_cards)}")
            print(f"   All card IDs: {sorted(all_playable_cards)}")

            if all_playable_cards:
                self.card_vocab_size = max(all_playable_cards) + 1
                print(f"\n   ✅ Recommended num_cards: {self.card_vocab_size}")
                print(f"      (max card ID {max(all_playable_cards)} + 1 for index 0)")
            else:
                self.card_vocab_size = 114  # Default for all CR cards
                print(f"\n   ⚠️  No cards found, using default: {self.card_vocab_size}")

            print("=" * 60)

            # Global card usage analysis
            print(f"\n🃏 GLOBAL CARD USAGE ANALYSIS:")
            print("=" * 60)

            # Extract all card IDs from actions (first column)
            all_card_ids = [action[0] for action in all_actions]
            unique_cards = sorted(set(all_card_ids))
            print(f"   Unique cards used: {unique_cards}")
            print(f"   Number of unique cards: {len(unique_cards)}")

            # Count frequency of each card
            from collections import Counter

            card_counts = Counter(all_card_ids)
            print(f"\n   Card frequency distribution:")
            total_actions = len(all_card_ids)
            for card_id in sorted(card_counts.keys()):
                count = card_counts[card_id]
                percentage = (count / total_actions) * 100
                print(
                    f"     Card {int(card_id)}: {count:6d} times ({percentage:5.1f}%)"
                )

            # Check for card 0 (no-action) dominance
            card_0_count = card_counts.get(0, 0)
            card_0_pct = (card_0_count / total_actions) * 100
            print(f"\n   Card 0 (no-action) percentage: {card_0_pct:.1f}%")
            if card_0_pct > 40:
                print(f"   ⚠️  WARNING: Card 0 dominates dataset!")
                print(f"   This will make card prediction trivially easy.")
            print("=" * 60)

    def get_dataset(
        self, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader with weighted sampling"""
        if not self.replay_data["states"]:
            print("WARNING: Empty replay data")
            return None

        dataset = ReplayDataset(self.replay_data, self.sequence_length)

        if len(dataset) == 0:
            print("WARNING: No valid sequences")
            return None

        # Use weighted sampler
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset.sample_weights),
            replacement=True,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            drop_last=True,
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
            # ===== ADD THIS BEFORE BATCH TESTING =====
            print("\n" + "=" * 60)
            print("FULL DATASET STATISTICS (before batching)")
            print("=" * 60)

            dataset = dataloader.dataset
            print(f"Total sequences in dataset: {len(dataset)}")
            print(f"Total frames: {len(dataset) * 16}")

            # Sample 1000 random sequences to get action distribution
            import random

            sample_size = min(1000, len(dataset))
            sample_indices = random.sample(range(len(dataset)), sample_size)

            total_slot_0 = 0
            total_real_actions = 0
            total_frames_sampled = 0

            print(f"\nSampling {sample_size} random sequences...")
            for idx in sample_indices:
                start_idx = dataset.valid_indices[idx]
                end_idx = start_idx + dataset.sequence_length
                seq_actions = dataset.replay_data["actions"][start_idx:end_idx]

                for action in seq_actions:
                    total_frames_sampled += 1
                    if action[0] == 0:
                        total_slot_0 += 1
                    else:
                        total_real_actions += 1

            slot_0_pct = (total_slot_0 / total_frames_sampled) * 100
            real_action_pct = (total_real_actions / total_frames_sampled) * 100

            print(f"\n📊 FULL DATASET ACTION DISTRIBUTION:")
            print(
                f"   Slot 0 (wait): {total_slot_0}/{total_frames_sampled} ({slot_0_pct:.1f}%)"
            )
            print(
                f"   Real actions: {total_real_actions}/{total_frames_sampled} ({real_action_pct:.1f}%)"
            )

            if real_action_pct < 35:
                print(f"   ⚠️  WARNING: Real actions < 35%")
                print(f"   Recommendation: Increase min_actions filter to 8 or 10")
            elif real_action_pct > 45:
                print(f"   ✅ GOOD: Real actions > 45%")
            else:
                print(f"   ✅ OK: Real actions in acceptable range")

            print("=" * 60)
            # ===== END NEW SECTION =====

            print("\nTesting one batch...")
            for batch in dataloader:
                print("\n✅ Batch loaded successfully!")
                print("Batch keys:", batch.keys())
                print("Batch shapes:")
                for k, v in batch.items():
                    print(f"  {k}: {v.shape}")

                # ===== DETAILED STATE/ACTION INSPECTION =====
                print("\n🔍 INSPECTING ACTUAL STATE VALUES:")
                print("=" * 60)
                states = batch["states"][0]  # First sequence
                actions = batch["actions"][0]  # First sequence

                print(f"First 5 timesteps of first sequence:")
                for t in range(min(5, len(states))):
                    state = states[t]
                    action = actions[t]
                    elixir, time, c1, c2, c3, c4 = state.numpy()
                    card_slot, x, y = action.numpy()

                    print(f"\n  Timestep {t}:")
                    print(f"    Elixir: {elixir:.2f}, Time: {time:.2f}")
                    print(
                        f"    Cards in hand: [{c1:.0f}, {c2:.0f}, {c3:.0f}, {c4:.0f}]"
                    )
                    print(
                        f"    Action: Play slot {card_slot:.0f} at ({x:.1f}, {y:.1f})"
                    )

                print("\n" + "=" * 60)
                print("❓ DIAGNOSTIC QUESTIONS:")
                print("=" * 60)

                # Check if cards are always zero
                all_states = batch["states"].numpy()  # (B, T, 6)
                card_values = all_states[:, :, 2:]  # (B, T, 4) - just the card columns

                num_zero_cards = (card_values == 0).sum()
                total_card_slots = card_values.size
                zero_percentage = (num_zero_cards / total_card_slots) * 100

                print(f"\n1. Card Information in States:")
                print(f"   Total card slots: {total_card_slots}")
                print(f"   Zero values: {num_zero_cards} ({zero_percentage:.1f}%)")

                if zero_percentage > 90:
                    print(f"   ❌ PROBLEM: Cards are mostly zeros!")
                    print(f"   → Model doesn't know which cards are available")
                elif zero_percentage < 10:
                    print(f"   ✅ GOOD: Cards have meaningful values")
                else:
                    print(f"   ⚠️  MIXED: Some cards have values, some don't")

                # Check action distribution
                all_actions = batch["actions"].numpy()  # (B, T, 3)
                card_slots_used = all_actions[:, :, 0]  # (B, T)

                print(f"\n2. Action Distribution in This Batch:")
                unique_slots, counts = np.unique(card_slots_used, return_counts=True)
                for slot, count in zip(unique_slots, counts):
                    pct = (count / card_slots_used.size) * 100
                    print(f"   Slot {int(slot)}: {count:3d} times ({pct:4.1f}%)")

                slot_0_pct = (card_slots_used == 0).sum() / card_slots_used.size * 100
                if slot_0_pct > 50:
                    print(
                        f"   ⚠️  WARNING: Slot 0 (no-action) dominates this batch ({slot_0_pct:.1f}%)"
                    )

                # Check position values for actual actions
                print(f"\n3. Position Values for Real Actions (slot > 0):")
                # Flatten everything first to match dimensions
                all_actions_flat = all_actions.reshape(-1, 3)  # (B*T, 3)
                card_slots_flat = card_slots_used.flatten()  # (B*T,)
                real_actions_mask = card_slots_flat > 0

                if real_actions_mask.any():
                    real_positions = all_actions_flat[real_actions_mask][
                        :, 1:
                    ]  # (N, 2) - x,y only
                    print(f"   Number of real actions: {len(real_positions)}")
                    print(
                        f"   X range: [{real_positions[:, 0].min():.2f}, {real_positions[:, 0].max():.2f}]"
                    )
                    print(
                        f"   Y range: [{real_positions[:, 1].min():.2f}, {real_positions[:, 1].max():.2f}]"
                    )

                    # Check if positions are diverse or always the same
                    unique_positions = len(np.unique(real_positions, axis=0))
                    print(f"   Unique positions: {unique_positions}")

                    if unique_positions < 5:
                        print(
                            f"   ⚠️  WARNING: Very few unique positions! Actions might be repetitive."
                        )
                else:
                    print(f"   ❌ No real actions in this batch!")

                print("\n" + "=" * 60)
                print("SUMMARY:")
                print("=" * 60)
                print("If you see:")
                print("  • Cards mostly 0 → Model can't learn card strategy")
                print("  • Slot 0 dominates → Model just learns to wait")
                print("  • Few unique positions → Model memorizes, doesn't generalize")
                print("=" * 60)

                break
        else:
            print("❌ Failed to create dataloader")
    else:
        print("❌ No data to test")
