# detection/tower_detector.py
"""OCR-based tower detection with GPU acceleration via EasyOCR"""

import numpy as np
from typing import Dict, Optional, Tuple
from config.tower_stats import identify_tower, get_max_hp, TOWER_DISPLAY_NAMES
from detection.ocr_reader import OCRReader


class TowerDetector:
    """Detect towers using OCR at fixed positions"""

    # Fixed tower positions (adjust for your resolution/capture)
    TOWER_POSITIONS = {
        "ally_left": {
            "hp_region": (60, 780, 190, 810),
            "level_region": (100, 750, 150, 775),
            "is_king": False,
        },
        "ally_right": {
            "hp_region": (530, 780, 660, 810),
            "level_region": (570, 750, 620, 775),
            "is_king": False,
        },
        "ally_king": {
            "hp_region": (280, 900, 440, 930),
            "level_region": (340, 850, 400, 875),
            "is_king": True,
        },
        "enemy_left": {
            "hp_region": (60, 200, 190, 230),
            "level_region": (100, 150, 150, 175),
            "is_king": False,
        },
        "enemy_right": {
            "hp_region": (530, 200, 660, 230),
            "level_region": (570, 150, 620, 175),
            "is_king": False,
        },
        "enemy_king": {
            "hp_region": (280, 130, 440, 160),
            "level_region": (340, 80, 400, 105),
            "is_king": True,
        },
    }

    def __init__(self, ocr_reader: OCRReader = None):
        # Use shared OCR reader for GPU efficiency
        self.ocr_reader = ocr_reader if ocr_reader else OCRReader()

        # Initialize tower states
        self.tower_states = {}
        for tower_name in self.TOWER_POSITIONS.keys():
            self.tower_states[tower_name] = {
                "alive": True,
                "type": None,
                "hp": None,
                "max_hp": None,
                "level": None,
                "identified": False,
            }

        self.match_started = False

    def detect_towers(self, frame: np.ndarray) -> Dict:
        """
        Detect all towers in frame

        Skip first few frames to avoid menu state
        """

        for tower_name, tower_info in self.TOWER_POSITIONS.items():
            state = self.tower_states[tower_name]

            hp_region = tower_info["hp_region"]
            level_region = tower_info["level_region"]
            is_king = tower_info["is_king"]

            # Read HP and Level from OCR
            hp = self._read_hp(frame, hp_region)
            level = self._read_level(frame, level_region)

            # Case 1: No indicators visible
            if hp is None and level is None:
                # Only mark as destroyed if it was previously identified
                if state["identified"] and state["alive"]:
                    print(f"💀 Tower destroyed: {tower_name} ({state['type']})")
                    state["alive"] = False
                    state["hp"] = 0
                # Otherwise, just skip (might be menu/loading)
                continue

            # Case 2: Tower visible with stats
            if hp is not None:
                state["alive"] = True
                state["hp"] = hp

                if level is not None:
                    state["level"] = level

                # Identify tower type ONCE (at full HP)
                if not state["identified"] and level is not None:
                    tower_type = identify_tower(hp, level, is_king)

                    if tower_type != "unknown":
                        state["type"] = tower_type
                        state["identified"] = True
                        state["max_hp"] = get_max_hp(tower_type, level)

                        display_name = TOWER_DISPLAY_NAMES.get(tower_type, tower_type)
                        print(
                            f"✅ Identified {tower_name}: {display_name} (Lvl {level}, {hp} HP)"
                        )

                        if not self.match_started:
                            self.match_started = True

        return self.get_tower_summary()

    def _read_hp(
        self, frame: np.ndarray, region: Tuple[int, int, int, int]
    ) -> Optional[int]:
        """Read HP value from region"""

        x1, y1, x2, y2 = region
        h, w = frame.shape[:2]

        # Validate region
        if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return None

        hp_img = frame[y1:y2, x1:x2]

        if hp_img.size == 0:
            return None

        # Use OCRReader for GPU-accelerated reading
        return self.ocr_reader.read_tower_hp(frame, region)

    def _read_level(
        self, frame: np.ndarray, region: Tuple[int, int, int, int]
    ) -> Optional[int]:
        """Read level value from region using OCRReader"""

        x1, y1, x2, y2 = region
        h, w = frame.shape[:2]

        # Validate region
        if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return None

        # Use OCRReader for GPU-accelerated reading
        return self.ocr_reader.read_tower_level(frame, region)

    def get_tower_summary(self) -> Dict:
        """Get summary of tower states"""

        ally_alive = sum(
            1
            for name, state in self.tower_states.items()
            if "ally" in name and state["alive"]
        )
        enemy_alive = sum(
            1
            for name, state in self.tower_states.items()
            if "enemy" in name and state["alive"]
        )

        return {
            "ally_towers_alive": ally_alive,
            "enemy_towers_alive": enemy_alive,
            "ally_crowns": 3 - ally_alive,
            "enemy_crowns": 3 - enemy_alive,
            "towers": self.tower_states,
            "match_started": self.match_started,
        }

    def reset(self):
        """Reset for new match"""
        for tower_name in self.TOWER_POSITIONS.keys():
            self.tower_states[tower_name] = {
                "alive": True,
                "type": None,
                "hp": None,
                "max_hp": None,
                "level": None,
                "identified": False,
            }
        self.match_started = False
