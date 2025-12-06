# detection/tower_detector.py
"""OCR-based tower detection with GPU acceleration via EasyOCR"""

import numpy as np
from typing import Dict, Optional, Tuple
from config.tower_stats import identify_tower, get_max_hp, TOWER_DISPLAY_NAMES
from detection.ocr_reader import OCRReader


class TowerDetector:
    """Detect towers using OCR at fixed positions"""

    # Hero types that affect tower positions
    HERO_TYPES = ["standard", "royal_chef", "dagger_duchess", "cannoneer"]

    # King tower coordinates by hero type
    # Format: (x1, y1, x2, y2) = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    # Added 3px padding on left and right sides
    KING_TOWER_COORDS = {
        "royal_chef": {
            "enemy_level": (346, 30, 371, 52),  # 349-3, 30, 368+3, 52
            "ally_level": (346, 963, 371, 983),  # 349-3, 963, 368+3, 983
            "enemy_hp": (337, 15, 400, 36),  # 340-3, 15, 397+3, 36
            "ally_hp": (338, 971, 402, 993),  # 341-3, 971, 399+3, 993
        },
        "standard": {
            "enemy_level": (346, 36, 370, 58),  # 349-3, 36, 367+3, 58
            "ally_level": (346, 962, 370, 984),  # 349-3, 962, 367+3, 984
            "enemy_hp": (336, 21, 400, 46),  # 339-3, 21, 397+3, 46
            "ally_hp": (337, 970, 406, 993),  # 340-3, 970, 403+3, 993
        },
        # dagger_duchess and cannoneer use standard king tower coords
        "dagger_duchess": {
            "enemy_level": (346, 36, 370, 58),
            "ally_level": (346, 962, 370, 984),
            "enemy_hp": (336, 21, 400, 46),
            "ally_hp": (337, 970, 406, 993),
        },
        "cannoneer": {
            "enemy_level": (346, 36, 370, 58),
            "ally_level": (346, 962, 370, 984),
            "enemy_hp": (336, 21, 400, 46),
            "ally_hp": (337, 970, 406, 993),
        },
    }

    # Princess tower coordinates by hero type
    # Added 3px padding on left and right sides
    PRINCESS_TOWER_COORDS = {
        "standard": {
            "enemy_left_level": (120, 185, 140, 201),  # 123-3, 185, 137+3, 201
            "enemy_right_level": (498, 185, 519, 201),  # 501-3, 185, 516+3, 201
            "ally_left_level": (120, 791, 141, 807),  # 123-3, 791, 138+3, 807
            "ally_right_level": (498, 791, 519, 807),  # 501-3, 791, 516+3, 807
            "enemy_left_hp": (143, 168, 200, 191),  # 146-3, 171, 197+3, 190
            "enemy_right_hp": (521, 168, 580, 191),  # 524-3, 171, 575+3, 190
            "ally_left_hp": (144, 796, 200, 815),  # 147-3, 796, 197+3, 815
            "ally_right_hp": (521, 796, 580, 815),  # 524-3, 795, 576+3, 815
        },
        "royal_chef": {
            "enemy_left_level": (121, 175, 141, 192),  # 124-3, 175, 138+3, 192
            "enemy_right_level": (498, 175, 520, 192),  # 501-3, 175, 517+3, 192
            "ally_left_level": (121, 802, 141, 818),  # 124-3, 802, 138+3, 818
            "ally_right_level": (498, 802, 519, 818),  # 501-3, 802, 516+3, 818
            "enemy_left_hp": (142, 161, 199, 182),  # 145-3, 161, 196+3, 182
            "enemy_right_hp": (520, 161, 579, 181),  # 523-3, 161, 574+3, 181
            "ally_left_hp": (143, 807, 200, 828),  # 146-3, 807, 197+3, 828
            "ally_right_hp": (521, 807, 578, 827),  # 524-3, 807, 575+3, 827
        },
        "dagger_duchess": {
            "enemy_left_level": (121, 185, 140, 201),  # 124-3, 185, 137+3, 201
            "enemy_right_level": (498, 185, 519, 201),  # 501-3, 185, 516+3, 201
            "ally_left_level": (120, 791, 141, 807),  # 123-3, 791, 138+3, 807
            "ally_right_level": (498, 791, 518, 808),  # 501-3, 791, 515+3, 808
            "enemy_left_hp": (142, 171, 201, 191),  # 145-3, 171, 198+3, 191
            "enemy_right_hp": (521, 171, 579, 191),  # 524-3, 171, 576+3, 191
            "ally_left_hp": (143, 795, 201, 814),  # 146-3, 795, 198+3, 814
            "ally_right_hp": (521, 795, 579, 814),  # 524-3, 795, 576+3, 814
        },
        "cannoneer": {
            "enemy_left_level": (121, 187, 141, 203),  # 124-3, 187, 138+3, 203
            "enemy_right_level": (498, 187, 519, 203),  # 501-3, 187, 516+3, 203
            "ally_left_level": (121, 791, 140, 807),  # 124-3, 791, 137+3, 807
            "ally_right_level": (498, 791, 518, 807),  # 501-3, 791, 515+3, 807
            "enemy_left_hp": (143, 172, 196, 192),  # 146-3, 172, 193+3, 192
            "enemy_right_hp": (521, 172, 574, 193),  # 524-3, 172, 571+3, 193
            "ally_left_hp": (143, 795, 196, 814),  # 146-3, 795, 193+3, 814
            "ally_right_hp": (521, 795, 574, 814),  # 524-3, 795, 571+3, 814
        },
    }

    def __init__(self, ocr_reader: OCRReader = None, debug_dir: str = None):
        # Use shared OCR reader for GPU efficiency
        self.ocr_reader = ocr_reader if ocr_reader else OCRReader()

        # Debug directory for saving OCR debug images
        self.debug_dir = debug_dir
        self._frame_count = 0

        # Current hero type (affects coordinates) - default to standard
        self.ally_hero_type = "standard"
        self.enemy_hero_type = "standard"

        # Locked levels - once detected, don't check again
        self._levels_locked = False
        self._ally_king_level: Optional[int] = None
        self._enemy_king_level: Optional[int] = None
        self._ally_princess_level: Optional[int] = None
        self._enemy_princess_level: Optional[int] = None

        # Initialize tower states
        self.tower_states = {
            "ally_left": self._empty_tower_state(is_king=False),
            "ally_right": self._empty_tower_state(is_king=False),
            "ally_king": self._empty_tower_state(is_king=True),
            "enemy_left": self._empty_tower_state(is_king=False),
            "enemy_right": self._empty_tower_state(is_king=False),
            "enemy_king": self._empty_tower_state(is_king=True),
        }

        self.match_started = False

    # Number of consecutive frames without HP required to confirm state change
    DEBOUNCE_FRAMES = 5

    def _empty_tower_state(self, is_king: bool) -> Dict:
        """Create empty tower state dict"""
        return {
            "alive": True,
            "type": None,
            "hp": None,
            "max_hp": None,
            "level": None,
            "identified": False,
            "is_king": is_king,
            "activated": False,  # King tower only - becomes active when damaged or princess destroyed
            "no_hp_count": 0,  # Counter for consecutive frames without HP (for debouncing)
            "has_hp_count": 0,  # Counter for consecutive frames with HP (for king activation)
            "destruction_announced": False,  # Prevent duplicate destruction messages
            "activation_announced": False,  # Prevent duplicate activation messages (king tower)
        }

    def _get_tower_regions(self, tower_name: str) -> Tuple[Tuple, Tuple]:
        """Get HP and level regions for a tower based on current hero types"""
        is_ally = "ally" in tower_name
        is_king = "king" in tower_name

        # Determine which hero type to use for coordinates
        hero_type = self.ally_hero_type if is_ally else self.enemy_hero_type

        if is_king:
            coords = self.KING_TOWER_COORDS[hero_type]
            if is_ally:
                return coords["ally_hp"], coords["ally_level"]
            else:
                return coords["enemy_hp"], coords["enemy_level"]
        else:
            coords = self.PRINCESS_TOWER_COORDS[hero_type]
            if "left" in tower_name:
                if is_ally:
                    return coords["ally_left_hp"], coords["ally_left_level"]
                else:
                    return coords["enemy_left_hp"], coords["enemy_left_level"]
            else:  # right
                if is_ally:
                    return coords["ally_right_hp"], coords["ally_right_level"]
                else:
                    return coords["enemy_right_hp"], coords["enemy_right_level"]

    def _get_all_level_regions(self, tower_name: str) -> list:
        """Get all possible level regions for a tower across all hero types"""
        is_ally = "ally" in tower_name
        is_king = "king" in tower_name

        regions = []

        for hero_type in self.HERO_TYPES:
            if is_king:
                coords = self.KING_TOWER_COORDS[hero_type]
                if is_ally:
                    regions.append((hero_type, coords["ally_level"]))
                else:
                    regions.append((hero_type, coords["enemy_level"]))
            else:
                coords = self.PRINCESS_TOWER_COORDS[hero_type]
                if "left" in tower_name:
                    if is_ally:
                        regions.append((hero_type, coords["ally_left_level"]))
                    else:
                        regions.append((hero_type, coords["enemy_left_level"]))
                else:  # right
                    if is_ally:
                        regions.append((hero_type, coords["ally_right_level"]))
                    else:
                        regions.append((hero_type, coords["enemy_right_level"]))

        return regions

    def _get_all_hp_regions(self, tower_name: str) -> list:
        """Get all possible HP regions for a tower across all hero types"""
        is_ally = "ally" in tower_name
        is_king = "king" in tower_name

        regions = []

        for hero_type in self.HERO_TYPES:
            if is_king:
                coords = self.KING_TOWER_COORDS[hero_type]
                if is_ally:
                    regions.append((hero_type, coords["ally_hp"]))
                else:
                    regions.append((hero_type, coords["enemy_hp"]))
            else:
                coords = self.PRINCESS_TOWER_COORDS[hero_type]
                if "left" in tower_name:
                    if is_ally:
                        regions.append((hero_type, coords["ally_left_hp"]))
                    else:
                        regions.append((hero_type, coords["enemy_left_hp"]))
                else:  # right
                    if is_ally:
                        regions.append((hero_type, coords["ally_right_hp"]))
                    else:
                        regions.append((hero_type, coords["enemy_right_hp"]))

        return regions

    def _try_read_level_all_coords(
        self, frame: np.ndarray, tower_name: str
    ) -> Optional[int]:
        """Try reading level from all possible coordinate sets until one works"""
        all_regions = self._get_all_level_regions(tower_name)

        for hero_type, region in all_regions:
            debug_name = f"{tower_name}_{hero_type}" if self.debug_dir else None
            level = self._read_level(frame, region, debug_name)
            if level is not None:
                # Found a valid level - update the hero type for this side
                is_ally = "ally" in tower_name
                if is_ally:
                    self.ally_hero_type = hero_type
                else:
                    self.enemy_hero_type = hero_type
                return level

        return None

    def _try_read_hp_all_coords(
        self, frame: np.ndarray, tower_name: str
    ) -> Optional[int]:
        """Try reading HP from all possible coordinate sets until one works"""
        all_regions = self._get_all_hp_regions(tower_name)

        for hero_type, region in all_regions:
            hp = self._read_hp(frame, region)
            if hp is not None:
                return hp

        return None

    def set_hero_types(self, ally_hero: str = "standard", enemy_hero: str = "standard"):
        """Set hero types to use correct tower coordinates"""
        if ally_hero in self.HERO_TYPES:
            self.ally_hero_type = ally_hero
        if enemy_hero in self.HERO_TYPES:
            self.enemy_hero_type = enemy_hero

    def detect_towers(self, frame: np.ndarray) -> Dict:
        """
        Detect all towers in frame

        Levels are only checked until locked (they don't change during match)
        King tower HP only shows after a princess tower is destroyed
        """
        # Increment frame count for debug naming
        self._frame_count += 1

        # First, try to lock in levels if not already done
        if not self._levels_locked:
            self._try_lock_levels(frame)

        for tower_name in self.tower_states.keys():
            state = self.tower_states[tower_name]
            hp_region, level_region = self._get_tower_regions(tower_name)
            is_king = state["is_king"]

            # Read HP
            hp = self._read_hp(frame, hp_region)

            # Only read level if not already locked
            level = None
            if not self._levels_locked:
                level = self._read_level(frame, level_region, tower_name)
            else:
                # Use cached level
                if is_king:
                    level = (
                        self._ally_king_level
                        if "ally" in tower_name
                        else self._enemy_king_level
                    )
                else:
                    level = (
                        self._ally_princess_level
                        if "ally" in tower_name
                        else self._enemy_princess_level
                    )

            # Case 1: No HP visible
            if hp is None:
                # Reset has_hp counter since we don't see HP
                state["has_hp_count"] = 0

                # Increment no_hp counter
                state["no_hp_count"] += 1

                # King tower HP only shows when:
                # a) King tower took damage (activated)
                # b) A princess tower is destroyed
                # So don't mark king as destroyed just because HP is None

                # Only mark princess tower as destroyed after DEBOUNCE_FRAMES consecutive frames without HP
                if not is_king and state["identified"] and state["alive"]:
                    if state["no_hp_count"] >= self.DEBOUNCE_FRAMES:
                        if not state.get("destruction_announced", False):
                            print(
                                f"💀 Tower destroyed: {tower_name} ({state['type']}) - confirmed after {self.DEBOUNCE_FRAMES} frames"
                            )
                            state["destruction_announced"] = True
                        state["alive"] = False
                        state["hp"] = 0
                continue

            # Case 2: Tower visible with HP
            # Reset no_hp counter since we see HP
            state["no_hp_count"] = 0

            # Increment has_hp counter
            state["has_hp_count"] += 1

            state["alive"] = True
            state["hp"] = hp

            # If we see HP, the tower is clearly NOT destroyed
            # Reset destruction announced flag in case of previous false positive
            if state.get("destruction_announced", False):
                print(
                    f"   ⚠️ Tower {tower_name} HP visible again - was NOT destroyed (false positive)"
                )
                state["destruction_announced"] = False

            # Track king tower activation - only after seeing HP for DEBOUNCE_FRAMES consecutive frames
            if is_king and not state["activated"]:
                if state["has_hp_count"] >= self.DEBOUNCE_FRAMES:
                    state["activated"] = True
                    if not state.get("activation_announced", False):
                        print(
                            f"👑 King tower ACTIVATED: {tower_name} (HP visible for {self.DEBOUNCE_FRAMES} frames)"
                        )
                        state["activation_announced"] = True

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

    def _try_lock_levels(self, frame: np.ndarray):
        """Try to detect and lock all tower levels by checking all possible coordinate sets"""
        # Check ally king level - try all hero type coords
        if self._ally_king_level is None:
            level = self._try_read_level_all_coords(frame, "ally_king")
            if level is not None:
                self._ally_king_level = level
                print(
                    f"🔒 Locked ally king level: {level} (hero: {self.ally_hero_type})"
                )

        # Check enemy king level - try all hero type coords
        if self._enemy_king_level is None:
            level = self._try_read_level_all_coords(frame, "enemy_king")
            if level is not None:
                self._enemy_king_level = level
                print(
                    f"🔒 Locked enemy king level: {level} (hero: {self.enemy_hero_type})"
                )

        # Check ally princess level (use left tower) - try all hero type coords
        if self._ally_princess_level is None:
            level = self._try_read_level_all_coords(frame, "ally_left")
            if level is not None:
                self._ally_princess_level = level
                print(
                    f"🔒 Locked ally princess level: {level} (hero: {self.ally_hero_type})"
                )

        # Check enemy princess level (use left tower) - try all hero type coords
        if self._enemy_princess_level is None:
            level = self._try_read_level_all_coords(frame, "enemy_left")
            if level is not None:
                self._enemy_princess_level = level
                print(
                    f"🔒 Locked enemy princess level: {level} (hero: {self.enemy_hero_type})"
                )

        # Also capture right tower debug images on first frame
        if self._frame_count == 1 and self.debug_dir:
            self._try_read_level_all_coords(frame, "ally_right")
            self._try_read_level_all_coords(frame, "enemy_right")
            hp_region, level_region = self._get_tower_regions("enemy_right")
            self._read_level(frame, level_region, "enemy_right")

        # Lock if we have all 4 levels
        if all(
            [
                self._ally_king_level,
                self._enemy_king_level,
                self._ally_princess_level,
                self._enemy_princess_level,
            ]
        ):
            self._levels_locked = True
            print("🔒 All tower levels locked - stopping level OCR")

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
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        tower_name: str = None,
    ) -> Optional[int]:
        """Read level value from region using OCRReader"""

        x1, y1, x2, y2 = region
        h, w = frame.shape[:2]

        # Validate region
        if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return None

        # Build debug path if debug_dir is set
        debug_path = None
        if self.debug_dir and tower_name:
            import os

            os.makedirs(self.debug_dir, exist_ok=True)
            debug_path = os.path.join(
                self.debug_dir, f"frame{self._frame_count:04d}_{tower_name}_level"
            )

        # Use OCRReader for GPU-accelerated reading
        return self.ocr_reader.read_tower_level(frame, region, debug_path=debug_path)

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

        # Check king tower activation status
        ally_king_activated = self.tower_states["ally_king"]["activated"]
        enemy_king_activated = self.tower_states["enemy_king"]["activated"]

        return {
            "ally_towers_alive": ally_alive,
            "enemy_towers_alive": enemy_alive,
            "ally_crowns": 3 - ally_alive,
            "enemy_crowns": 3 - enemy_alive,
            "towers": self.tower_states,
            "match_started": self.match_started,
            "levels_locked": self._levels_locked,
            "ally_hero": self.ally_hero_type,
            "enemy_hero": self.enemy_hero_type,
            "ally_king_activated": ally_king_activated,
            "enemy_king_activated": enemy_king_activated,
        }

    def reset(self):
        """Reset for new match"""
        self.tower_states = {
            "ally_left": self._empty_tower_state(is_king=False),
            "ally_right": self._empty_tower_state(is_king=False),
            "ally_king": self._empty_tower_state(is_king=True),
            "enemy_left": self._empty_tower_state(is_king=False),
            "enemy_right": self._empty_tower_state(is_king=False),
            "enemy_king": self._empty_tower_state(is_king=True),
        }
        self.match_started = False
        self._levels_locked = False
        self._ally_king_level = None
        self._enemy_king_level = None
        self._ally_princess_level = None
        self._enemy_princess_level = None

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw tower OCR debug visualization on frame"""
        import cv2

        vis = frame.copy()

        # Draw all tower HP regions
        for tower_name in self.tower_states.keys():
            hp_region, level_region = self._get_tower_regions(tower_name)
            state = self.tower_states[tower_name]
            is_king = state["is_king"]

            # Determine color based on state
            if state["alive"]:
                if is_king and state["activated"]:
                    color = (0, 255, 255)  # Yellow for activated king
                elif state["hp"] is not None:
                    color = (0, 255, 0)  # Green for visible HP
                else:
                    color = (128, 128, 128)  # Gray for no HP visible
            else:
                color = (0, 0, 255)  # Red for destroyed

            # Draw HP region rectangle
            x1, y1, x2, y2 = hp_region
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw HP value if available
            hp_text = str(state["hp"]) if state["hp"] is not None else "--"
            text_x = x1
            text_y = y1 - 5 if y1 > 30 else y2 + 15

            # Background for text
            text_size = cv2.getTextSize(hp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                vis,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis, hp_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            # Draw level region (smaller, dashed effect via dots)
            lx1, ly1, lx2, ly2 = level_region
            cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (255, 255, 0), 1)

            # Draw level value
            level = state["level"]
            if level is not None:
                level_text = f"L{level}"
                cv2.putText(
                    vis,
                    level_text,
                    (lx1, ly1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                )

            # Draw tower name label
            label = tower_name.replace("_", " ").title()
            if is_king and state["activated"]:
                label += " ⚡"
            elif not state["alive"]:
                label += " 💀"

            label_y = y2 + 12 if y2 < vis.shape[0] - 20 else y1 - 18
            cv2.putText(
                vis,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

        # Draw summary info at top
        summary = self.get_tower_summary()
        info_lines = [
            f"Ally: {summary['ally_towers_alive']}/3 | Enemy: {summary['enemy_towers_alive']}/3",
            f"Crowns: {summary['enemy_crowns']} - {summary['ally_crowns']}",
        ]
        if summary["levels_locked"]:
            info_lines.append("Levels: LOCKED")
        if summary["ally_king_activated"]:
            info_lines.append("Ally King: ACTIVE")
        if summary["enemy_king_activated"]:
            info_lines.append("Enemy King: ACTIVE")

        y_offset = 15
        for line in info_lines:
            cv2.putText(
                vis,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            y_offset += 16

        return vis
