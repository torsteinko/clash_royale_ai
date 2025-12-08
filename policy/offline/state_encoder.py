"""
State encoder using YOLO detection for rich game state representation
"""

import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2


class GameStateEncoder:
    """Encode game frames into state vectors using YOLO detection"""

    def __init__(
        self,
        yolo_model_path: str = "runs/synthetic/train_20251207_183426_single/weights/best.pt",
    ):
        """
        Args:
            yolo_model_path: Path to trained YOLO model
        """
        self.yolo = YOLO(yolo_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State grid size (simplified spatial representation)
        self.grid_h = 8  # Divide arena into 8x8 grid
        self.grid_w = 8

        # Class categories from your YOLO model
        # TODO: Update with your actual class names
        self.troop_classes = {
            "ally_hog_rider": 0,
            "ally_musketeer": 1,
            "ally_ice_golem": 2,
            "ally_cannon": 3,
            "ally_ice_spirit": 4,
            "ally_skeletons": 5,
            "enemy_hog_rider": 6,
            "enemy_musketeer": 7,
            # ... add all your classes
        }

        print(f"✅ GameStateEncoder initialized with YOLO model: {yolo_model_path}")
        print(f"   Device: {self.device}")
        print(f"   Grid size: {self.grid_h}x{self.grid_w}")

    def encode_frame(
        self, frame: np.ndarray, cards_in_hand: list, elixir: float, time: float
    ) -> np.ndarray:
        """
        Encode a game frame into a state vector

        Args:
            frame: Game screenshot (BGR image)
            cards_in_hand: List of 4 card IDs currently in hand
            elixir: Current elixir amount (0-10)
            time: Game time in seconds

        Returns:
            state_vector: Fixed-size numpy array representing game state
        """
        # Run YOLO detection
        results = self.yolo(frame, verbose=False)[0]

        # Initialize state components
        state = {}

        # 1. Scalar features (6 values)
        state["scalars"] = np.array(
            [
                elixir / 10.0,  # Normalize to [0, 1]
                time / 300.0,  # Normalize assuming max 5 min game
                cards_in_hand[0] / 107.0,  # Normalize card IDs
                cards_in_hand[1] / 107.0,
                cards_in_hand[2] / 107.0,
                cards_in_hand[3] / 107.0,
            ],
            dtype=np.float32,
        )

        # 2. Spatial features: troop positions on grid (8x8x2 = 128 values)
        # Separate channels for ally and enemy troops
        ally_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        enemy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.cpu()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Get class name
                cls_name = results.names[cls_id]

                # Calculate grid position (center of bounding box)
                cx = (x1 + x2) / 2 / frame.shape[1]  # Normalize to [0, 1]
                cy = (y1 + y2) / 2 / frame.shape[0]

                grid_x = int(cx * self.grid_w)
                grid_y = int(cy * self.grid_h)

                # Clamp to grid
                grid_x = max(0, min(grid_x, self.grid_w - 1))
                grid_y = max(0, min(grid_y, self.grid_h - 1))

                # Add to appropriate grid (accumulate confidence)
                if "ally" in cls_name:
                    ally_grid[grid_y, grid_x] += conf
                elif "enemy" in cls_name:
                    enemy_grid[grid_y, grid_x] += conf

        # Flatten grids
        state["ally_troops"] = ally_grid.flatten()  # 64 values
        state["enemy_troops"] = enemy_grid.flatten()  # 64 values

        # 3. Aggregate troop counts (10 values: 5 ally + 5 enemy troop types)
        troop_counts = np.zeros(10, dtype=np.float32)
        if results.boxes is not None:
            for box in results.boxes.cpu():
                cls_name = results.names[int(box.cls[0])]
                # Simple count aggregation (you can make this more sophisticated)
                if "hog" in cls_name.lower():
                    idx = 0 if "ally" in cls_name else 5
                    troop_counts[idx] += 1
                elif "musketeer" in cls_name.lower():
                    idx = 1 if "ally" in cls_name else 6
                    troop_counts[idx] += 1
                # ... add more troop types

        state["troop_counts"] = troop_counts / 10.0  # Normalize

        # Concatenate all features
        # Total: 6 + 64 + 64 + 10 = 144 features
        state_vector = np.concatenate(
            [
                state["scalars"],
                state["ally_troops"],
                state["enemy_troops"],
                state["troop_counts"],
            ]
        )

        return state_vector

    def encode_batch(self, frames: list, metadata: list) -> np.ndarray:
        """
        Encode a batch of frames

        Args:
            frames: List of frame images
            metadata: List of dicts with 'cards', 'elixir', 'time'

        Returns:
            batch_states: (B, state_dim) numpy array
        """
        states = []
        for frame, meta in zip(frames, metadata):
            state = self.encode_frame(
                frame, meta["cards"], meta["elixir"], meta["time"]
            )
            states.append(state)

        return np.stack(states)


# Utility function to update dataset.py
def convert_katacd_state_to_rich_tensor(
    state_dict: Dict, frame: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert KataCR state to rich tensor using YOLO (if frame available)
    Falls back to simple encoding if no frame

    Args:
        state_dict: KataCR state dictionary
        frame: Optional game frame for YOLO detection

    Returns:
        state_vector: Encoded state (144 features if frame available, else 6)
    """
    # Extract basic info
    elixir = state_dict.get("elixir", 0.0)
    time = state_dict.get("time", 0.0)
    cards = state_dict.get("cards", [0, 0, 0, 0])

    if frame is not None:
        # Use rich encoding with YOLO
        encoder = GameStateEncoder()
        return encoder.encode_frame(frame, cards, elixir, time)
    else:
        # Fallback to simple encoding (original 6 features)
        features = [elixir, time]
        for i in range(4):
            features.append(cards[i] if i < len(cards) else 0)
        return np.array(features, dtype=np.float32)
