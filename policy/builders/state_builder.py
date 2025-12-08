"""
Build state representation from game state extractor output

Adapts KataCR's state_builder to work with our YOLO-based detection system
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class StateBuilder:
    """Convert game state from extractor to training format"""

    def __init__(self):
        """Initialize state builder"""
        self.reset()

    def reset(self):
        """Reset episode state"""
        self.frame_count = 0
        self.time = 0

    def build_state(self, game_state: Dict) -> Dict:
        """
        Convert GameStateExtractor output to training state format

        Args:
            game_state: Output from GameStateExtractor.extract_state()
                Contains: troops, cards_in_hand, elixir, match_time, towers

        Returns:
            state: Dict with training-compatible format
                - 'troops': List of troop info dicts
                - 'cards': List of 4 card names in hand
                - 'elixir': Current elixir (0-10)
                - 'time': Match time in seconds
                - 'towers': Tower status dict
        """
        state = {}

        # Extract troops - convert to simple list format
        troops_info = []

        # Process ally troops
        for troop in game_state.get("troops", {}).get("ally", []):
            troops_info.append(
                {
                    "xy": troop["position"],  # (x, y) center position
                    "type": troop[
                        "type"
                    ],  # Troop class name (already stripped of "ally_")
                    "team": 0,  # 0 = ally
                    "bbox": troop["bbox"],  # (x1, y1, x2, y2)
                    "confidence": troop.get("confidence", 1.0),
                    "level": troop.get("level"),
                    "track_id": troop.get("track_id"),
                }
            )

        # Process enemy troops
        for troop in game_state.get("troops", {}).get("enemy", []):
            troops_info.append(
                {
                    "xy": troop["position"],
                    "type": troop["type"],  # Already stripped of "enemy_"
                    "team": 1,  # 1 = enemy
                    "bbox": troop["bbox"],
                    "confidence": troop.get("confidence", 1.0),
                    "level": troop.get("level"),
                    "track_id": troop.get("track_id"),
                }
            )

        state["troops"] = troops_info

        # Extract cards (4 cards in hand)
        cards = game_state.get("cards_in_hand", ["unknown"] * 4)
        state["cards"] = [c if c != "unknown" else "empty" for c in cards[:4]]

        # Extract elixir
        elixir = game_state.get("elixir")
        state["elixir"] = int(elixir) if elixir is not None else None

        # Extract time (convert from match_time if available)
        match_time = game_state.get("match_time")
        if match_time is not None:
            # Match time counts down from 180 (3 min)
            # Training expects elapsed time
            state["time"] = int(180 - match_time) if match_time <= 180 else 0
        else:
            state["time"] = self.time

        # Extract tower status
        state["towers"] = game_state.get("towers", {})

        # Update internal state
        self.frame_count += 1

        return state

    def build_arena_grid(self, troops: List[Dict], grid_size=(32, 18)) -> np.ndarray:
        """
        Build spatial grid representation of arena

        Args:
            troops: List of troop dicts with xy, type, team
            grid_size: (rows, cols) for arena grid

        Returns:
            grid: (rows, cols, features) array
                features = [troop_present, team, troop_class_id]
        """
        rows, cols = grid_size
        grid = np.zeros((rows, cols, 3), dtype=np.int32)

        for troop in troops:
            x, y = troop["xy"]

            # Convert pixel coords to grid coords
            # Assuming screen is 720x1280 (portrait)
            # Arena is roughly middle section
            grid_x = int((x / 720) * cols)
            grid_y = int((y / 1280) * rows)

            grid_x = np.clip(grid_x, 0, cols - 1)
            grid_y = np.clip(grid_y, 0, rows - 1)

            # Mark presence
            grid[grid_y, grid_x, 0] = 1
            # Mark team (-1 for ally, 1 for enemy)
            grid[grid_y, grid_x, 1] = 1 if troop["team"] == 1 else -1
            # Mark type (would need troop_name -> id mapping)
            # For now, use hash of type name
            grid[grid_y, grid_x, 2] = hash(troop["type"]) % 256

        return grid


class ActionBuilder:
    """Build action representation for training"""

    def __init__(self):
        """Initialize action builder"""
        self.reset()

    def reset(self):
        """Reset episode state"""
        self.last_action_frame = 0

    def build_action(self, card_slot: int, position: tuple) -> Dict:
        """
        Build action dict from card selection and placement

        Args:
            card_slot: Index of card (0-3), or -1 for no action
            position: (x, y) pixel coordinates of placement

        Returns:
            action: Dict with 'card_id' and 'xy' (or None for no action)
        """
        if card_slot < 0 or position is None:
            return {"card_id": 0, "xy": None}  # 0 = no action

        # Convert pixel position to grid coordinates
        x, y = position
        grid_x = int((x / 720) * 18)
        grid_y = int((y / 1280) * 32)

        return {
            "card_id": card_slot + 1,  # 1-4 for cards (0 is reserved for no-action)
            "xy": (np.clip(grid_x, 0, 17), np.clip(grid_y, 0, 31)),
        }


class RewardBuilder:
    """Calculate rewards for reinforcement learning"""

    def __init__(self):
        """Initialize reward builder"""
        self.reset()

    def reset(self):
        """Reset episode state"""
        self.last_towers = {
            "ally_left": True,
            "ally_right": True,
            "ally_king": True,
            "enemy_left": True,
            "enemy_right": True,
            "enemy_king": True,
        }
        self.total_reward = 0

    def calculate_reward(self, game_state: Dict) -> float:
        """
        Calculate reward from current game state

        Args:
            game_state: Current game state from extractor

        Returns:
            reward: Scalar reward value
        """
        reward = 0.0

        towers = game_state.get("towers", {})

        # Reward for destroying enemy towers
        if not towers.get("enemy_left", True) and self.last_towers.get(
            "enemy_left", True
        ):
            reward += 1.0
        if not towers.get("enemy_right", True) and self.last_towers.get(
            "enemy_right", True
        ):
            reward += 1.0
        if not towers.get("enemy_king", True) and self.last_towers.get(
            "enemy_king", True
        ):
            reward += 3.0  # King tower worth more

        # Penalty for losing towers
        if not towers.get("ally_left", True) and self.last_towers.get(
            "ally_left", True
        ):
            reward -= 1.0
        if not towers.get("ally_right", True) and self.last_towers.get(
            "ally_right", True
        ):
            reward -= 1.0
        if not towers.get("ally_king", True) and self.last_towers.get(
            "ally_king", True
        ):
            reward -= 3.0

        # Update last towers
        self.last_towers = towers.copy()
        self.total_reward += reward

        return reward

    def get_episode_reward(self, won: bool, crowns_won: int, crowns_lost: int) -> float:
        """
        Calculate final episode reward

        Args:
            won: Whether the episode was won
            crowns_won: Number of crowns won (towers destroyed)
            crowns_lost: Number of crowns lost

        Returns:
            total_reward: Final episode reward
        """
        reward = self.total_reward

        # Bonus for winning
        if won:
            reward += 5.0
        else:
            reward -= 5.0

        # Additional crown bonuses
        reward += crowns_won * 0.5
        reward -= crowns_lost * 0.5

        return reward


if __name__ == "__main__":
    # Test state builder
    sb = StateBuilder()

    # Mock game state
    game_state = {
        "troops": {
            "ally": [
                {
                    "position": (360, 800),
                    "type": "knight",
                    "bbox": (340, 780, 380, 820),
                    "confidence": 0.9,
                    "level": 11,
                }
            ],
            "enemy": [
                {
                    "position": (360, 400),
                    "type": "archer",
                    "bbox": (340, 380, 380, 420),
                    "confidence": 0.85,
                    "level": 10,
                }
            ],
        },
        "cards_in_hand": ["knight", "archer", "fireball", "goblin"],
        "elixir": 7,
        "match_time": 150,
        "towers": {
            "ally_left": True,
            "ally_right": True,
            "ally_king": True,
            "enemy_left": True,
            "enemy_right": False,  # Enemy tower destroyed
            "enemy_king": True,
        },
    }

    state = sb.build_state(game_state)
    print("State:", state)

    # Test action builder
    ab = ActionBuilder()
    action = ab.build_action(card_slot=2, position=(360, 640))
    print("Action:", action)

    # Test reward builder
    rb = RewardBuilder()
    reward = rb.calculate_reward(game_state)
    print(f"Reward: {reward}")
