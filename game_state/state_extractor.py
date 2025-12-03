# game_state/state_extractor.py
"""Complete game state extractor with troop tracking"""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from typing import Dict, List, Optional, Tuple

from detection.card_detector_simple import CardDetector
from detection.tower_detector import TowerDetector
from config.game_config import CARD_SLOTS, ELIXIR_REGION, TIMER_REGION


class TroopTracker:
    """Track troops across frames for consistency"""

    def __init__(self):
        self.tracked_troops = {}  # {troop_id: {data}}
        self.next_id = 0
        self.max_age = 1.0  # Forget troops after 1 second
        self.max_distance = 150  # Max pixels a troop can move between frames

    def update(self, new_detections: List[Dict]) -> List[Dict]:
        """
        Match new detections with tracked troops

        Args:
            new_detections: List of troops detected this frame

        Returns:
            Updated detections with consistent team assignments
        """

        current_time = time.time()

        # Clean up old tracks
        self._cleanup_old_tracks(current_time)

        # Match new detections with existing tracks
        matched_detections = []
        unmatched_new = list(range(len(new_detections)))

        for troop_id, tracked in list(self.tracked_troops.items()):
            best_match_idx = None
            best_distance = self.max_distance

            # Find closest matching detection
            for idx in unmatched_new:
                detection = new_detections[idx]

                # Must be same type
                if detection["type"] != tracked["type"]:
                    continue

                # Calculate distance
                dist = self._distance(detection["position"], tracked["position"])

                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = idx

            # Update track with matched detection
            if best_match_idx is not None:
                detection = new_detections[best_match_idx]

                # Keep the tracked team (consistency)
                detection["team"] = tracked["team"]
                detection["track_id"] = troop_id

                # Update tracked data
                tracked["position"] = detection["position"]
                tracked["bbox"] = detection["bbox"]
                tracked["last_seen"] = current_time

                matched_detections.append(detection)
                unmatched_new.remove(best_match_idx)

        # Handle unmatched new detections (new troops)
        for idx in unmatched_new:
            detection = new_detections[idx]

            # Assign new track ID
            troop_id = self.next_id
            self.next_id += 1

            detection["track_id"] = troop_id

            # Store in tracker
            self.tracked_troops[troop_id] = {
                "type": detection["type"],
                "team": detection["team"],
                "position": detection["position"],
                "bbox": detection["bbox"],
                "last_seen": current_time,
            }

            matched_detections.append(detection)

        return matched_detections

    def get_team_for_position(
        self, position: Tuple[float, float], troop_type: str
    ) -> Optional[str]:
        """
        Get team assignment from nearby tracked troop

        Useful when color detection is uncertain
        """

        min_dist = self.max_distance
        best_team = None

        for tracked in self.tracked_troops.values():
            if tracked["type"] == troop_type:
                dist = self._distance(position, tracked["position"])
                if dist < min_dist:
                    min_dist = dist
                    best_team = tracked["team"]

        return best_team

    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _cleanup_old_tracks(self, current_time: float):
        """Remove tracks that haven't been seen recently"""
        to_remove = []

        for troop_id, tracked in self.tracked_troops.items():
            if current_time - tracked["last_seen"] > self.max_age:
                to_remove.append(troop_id)

        for troop_id in to_remove:
            del self.tracked_troops[troop_id]

    def reset(self):
        """Clear all tracks"""
        self.tracked_troops.clear()
        self.next_id = 0


class GameStateExtractor:
    """Extract complete game state from screenshot"""

    def __init__(
        self, yolo_model_path="runs/detect/clash_royale_FINAL_1280px/weights/best.pt"
    ):
        print("\n" + "=" * 80)
        print("🎮 INITIALIZING GAME STATE EXTRACTOR")
        print("=" * 80)

        # Load YOLO model
        print("📦 Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        print(f"   ✅ YOLO loaded")

        # Load card detector
        print("🎴 Loading card templates...")
        self.card_detector = CardDetector()
        print(f"   ✅ Card detector ready")

        # Load tower detector
        print("🏰 Loading tower detector...")
        self.tower_detector = TowerDetector()
        print(f"   ✅ Tower detector ready")

        # Load troop tracker
        print("🎯 Loading troop tracker...")
        self.troop_tracker = TroopTracker()
        print(f"   ✅ Troop tracker ready")

        # Configure OCR
        try:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            self.ocr_enabled = True
        except:
            print("   ⚠️  Tesseract not found, OCR disabled")
            self.ocr_enabled = False

        print("=" * 80)
        print("✅ Ready to detect!")
        print("=" * 80 + "\n")

    def extract_state(self, frame: np.ndarray, debug: bool = False) -> Dict:
        """Extract complete game state"""

        if frame is None or frame.size == 0:
            return self._empty_state()

        state = {}

        # 1. Detect troops (with debug)
        raw_troops = self._detect_troops(frame, debug=debug)

        # Store debug frame if available
        if debug and "debug_frame" in raw_troops:
            state["debug_frame"] = raw_troops["debug_frame"]

        # 2. Apply tracking
        ally_troops = self.troop_tracker.update(raw_troops["ally"])
        enemy_troops = self.troop_tracker.update(raw_troops["enemy"])

        state["troops"] = {
            "ally": ally_troops,
            "enemy": enemy_troops,
            "total_ally": len(ally_troops),
            "total_enemy": len(enemy_troops),
        }

        # 3. Detect cards in hand (with debug info if enabled)
        if debug and hasattr(self.card_detector, "detect_cards_with_debug"):
            cards_debug = self.card_detector.detect_cards_with_debug(
                frame, state.get("debug_frame")
            )
            state["cards_in_hand"] = [c["name"] for c in cards_debug]
            state["cards_debug"] = cards_debug
        else:
            state["cards_in_hand"] = self.card_detector.detect_cards_in_hand(frame)

        # 4. Detect towers (OCR-based)
        tower_info = self.tower_detector.detect_towers(frame)
        state["towers"] = tower_info

        # 5. OCR (optional, skip if fails)
        state["elixir"] = None
        state["match_time"] = None

        if self.ocr_enabled:
            try:
                state["elixir"] = self._read_elixir(frame)
            except:
                pass

            try:
                state["match_time"] = self._read_timer(frame)
            except:
                pass

        # 6. Derived metrics
        state["is_double_elixir"] = (
            state["match_time"] is not None and state["match_time"] <= 60
        )
        state["is_overtime"] = (
            state["match_time"] is not None and state["match_time"] <= 0
        )

        return state

    def _empty_state(self):
        """Return empty state"""
        return {
            "troops": {"ally": [], "enemy": [], "total_ally": 0, "total_enemy": 0},
            "cards_in_hand": ["unknown", "unknown", "unknown", "unknown"],
            "elixir": None,
            "match_time": None,
            "towers": {},
            "is_double_elixir": False,
            "is_overtime": False,
        }

    def _detect_troops(self, frame: np.ndarray, debug: bool = False) -> Dict:
        """Detect troops using YOLO + color-based team detection"""

        try:
            results = self.yolo.predict(frame, conf=0.5, imgsz=1280, verbose=False)
        except:
            return {"ally": [], "enemy": [], "total_ally": 0, "total_enemy": 0}

        ally_troops = []
        enemy_troops = []

        # Create debug frame if needed
        debug_frame = frame.copy() if debug else None

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]

                # Determine team (pass debug_frame)
                side = self._detect_team_by_color(
                    frame, int(x1), int(y1), int(x2), int(y2), class_name, debug_frame
                )

                troop_data = {
                    "type": class_name,
                    "position": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "team": side,
                }

                if side == "ally":
                    ally_troops.append(troop_data)
                else:
                    enemy_troops.append(troop_data)

        result = {
            "ally": ally_troops,
            "enemy": enemy_troops,
            "total_ally": len(ally_troops),
            "total_enemy": len(enemy_troops),
        }

        if debug:
            result["debug_frame"] = debug_frame

        return result

    def _is_tower(self, class_name: str) -> bool:
        """Check if unit is a tower"""
        tower_keywords = ["tower", "king", "princess", "arena"]
        return any(kw in class_name.lower() for kw in tower_keywords)

    def _detect_team_by_color(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        class_name: str = "",
        debug_frame: np.ndarray = None,
    ) -> str:
        """
        Enhanced detection - search INSIDE top of unit bbox

        Level indicator is AT the top of the unit, not above it!
        """

        # Towers use position only
        if self._is_tower(class_name):
            center_y = (y1 + y2) / 2
            return "ally" if center_y > frame.shape[0] / 2 else "enemy"

        # Calculate position zones
        center_y = (y1 + y2) / 2
        h = frame.shape[0]

        in_ally_zone = center_y > h * 0.60
        in_enemy_zone = center_y < h * 0.40
        in_middle_zone = not in_ally_zone and not in_enemy_zone

        # Search area: TOP PORTION of unit bbox (level is inside, not above!)
        unit_width = x2 - x1
        unit_height = y2 - y1

        # Search in top 25% of unit height (where level/HP indicators are)
        search_top_portion = int(unit_height * 0.25)

        # SEARCH INSIDE THE BBOX TOP
        search_y1 = y1  # Start at top of unit
        search_y2 = min(y1 + search_top_portion, y2)  # Top 25% of unit

        # Keep search within unit width + small margin
        search_x1 = max(0, x1 - int(unit_width * 0.1))
        search_x2 = min(frame.shape[1], x2 + int(unit_width * 0.1))

        # DRAW DEBUG BOXES
        if debug_frame is not None:
            # Color detection area (purple)
            cv2.rectangle(
                debug_frame,
                (search_x1, search_y1),
                (search_x2, search_y2),
                (255, 0, 255),
                1,
            )

            # Level detection area (cyan) - even smaller, centered
            level_height = min(20, int(unit_height * 0.15))
            level_y1 = y1
            level_y2 = y1 + level_height
            level_x1 = x1 + int(unit_width * 0.25)
            level_x2 = x2 - int(unit_width * 0.25)
            cv2.rectangle(
                debug_frame,
                (level_x1, level_y1),
                (level_x2, level_y2),
                (0, 255, 255),
                1,
            )

        if search_y2 <= search_y1 or search_x2 <= search_x1:
            return "ally" if in_ally_zone or not in_enemy_zone else "enemy"

        search_region = frame[search_y1:search_y2, search_x1:search_x2]

        if search_region.size == 0:
            return "ally" if in_ally_zone or not in_enemy_zone else "enemy"

        try:
            # === FIRST: Try OCR to detect level (enemy always has level) ===
            level_detected = self._detect_level_ocr(frame, x1, y1, x2, y2)

            if level_detected:
                # Level detected = ENEMY
                if debug_frame is not None:
                    cv2.putText(
                        debug_frame,
                        f"LVL:{level_detected}",
                        (search_x1, search_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )
                return "enemy"

            # === COLOR DETECTION ===
            hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)

            # RED detection
            red_mask1 = cv2.inRange(
                hsv, np.array([0, 60, 70]), np.array([15, 255, 255])
            )
            red_mask2 = cv2.inRange(
                hsv, np.array([165, 60, 70]), np.array([180, 255, 255])
            )
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels_hsv = cv2.countNonZero(red_mask)

            # BLUE detection
            blue_mask = cv2.inRange(
                hsv, np.array([90, 60, 70]), np.array([130, 255, 255])
            )
            blue_pixels_hsv = cv2.countNonZero(blue_mask)

            # BGR channel analysis
            b, g, r = cv2.split(search_region)

            red_dominant = ((r.astype(int) - b.astype(int)) > 35) & (r > 80)
            red_pixels_bgr = np.count_nonzero(red_dominant)

            blue_dominant = ((b.astype(int) - r.astype(int)) > 35) & (b > 80)
            blue_pixels_bgr = np.count_nonzero(blue_dominant)

            # Pink detection
            pink_mask = (r > 120) & (b > 60) & ((r.astype(int) - b.astype(int)) > 20)
            pink_pixels = np.count_nonzero(pink_mask)

            # Total scores
            total_red = red_pixels_hsv + red_pixels_bgr + pink_pixels
            total_blue = blue_pixels_hsv + blue_pixels_bgr

            # DRAW DEBUG INFO
            if debug_frame is not None:
                cv2.putText(
                    debug_frame,
                    f"R:{total_red} B:{total_blue}",
                    (search_x1, search_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 255),
                    1,
                )

            # === DECISION LOGIC ===

            # STRONG COLOR SIGNAL
            if total_red > 50:
                return "enemy"

            if total_blue > 50:
                return "ally"

            # MODERATE COLOR SIGNAL
            if total_red > 20:
                if in_enemy_zone:
                    return "enemy"
                elif in_middle_zone and total_red > total_blue * 2:
                    return "enemy"
                elif in_ally_zone and total_red > 40:
                    return "enemy"

            if total_blue > 20:
                if in_ally_zone:
                    return "ally"
                elif in_middle_zone and total_blue > total_red * 2:
                    return "ally"
                elif in_enemy_zone and total_blue > 40:
                    return "ally"

            # WEAK/NO SIGNAL - Check tracking
            position = ((x1 + x2) / 2, (y1 + y2) / 2)
            tracked_team = self.troop_tracker.get_team_for_position(
                position, class_name
            )

            if tracked_team:
                return tracked_team

            # Final fallback: Position
            if in_ally_zone:
                return "ally"
            elif in_enemy_zone:
                return "enemy"
            else:
                return "ally"

        except Exception as e:
            if in_ally_zone:
                return "ally"
            elif in_enemy_zone:
                return "enemy"
            else:
                return "ally"

    def _detect_level_ocr(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Optional[int]:
        """
        Detect level number (1-23) at TOP of unit bbox

        Level badge is INSIDE the unit bbox at the top, not above it!
        """

        if not self.ocr_enabled:
            return None

        try:
            unit_width = x2 - x1
            unit_height = y2 - y1

            # Level badge is at the TOP of the unit, inside bbox
            level_height = min(20, int(unit_height * 0.15))  # Top 15% of unit height
            level_y1 = y1  # Start at top of bbox
            level_y2 = y1 + level_height

            # Center 50% of unit width (level badge is centered)
            level_x1 = x1 + int(unit_width * 0.25)
            level_x2 = x2 - int(unit_width * 0.25)

            h, w = frame.shape[:2]

            if (
                level_y1 >= level_y2
                or level_x1 >= level_x2
                or level_x2 > w
                or level_y2 > h
            ):
                return None

            level_region = frame[level_y1:level_y2, level_x1:level_x2]

            if (
                level_region.size == 0
                or level_region.shape[0] < 5
                or level_region.shape[1] < 5
            ):
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(level_region, cv2.COLOR_BGR2GRAY)

            # Level badges are typically yellow/white on colored background
            # Try multiple thresholds
            thresholds = [150, 120, 100]

            for thresh_val in thresholds:
                _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

                # Resize for better OCR
                scale = 4
                binary_scaled = cv2.resize(
                    binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

                # OCR config for 1-2 digits
                config = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"

                text = pytesseract.image_to_string(binary_scaled, config=config).strip()

                # Parse level (1-23)
                if text.isdigit():
                    level = int(text)
                    if 1 <= level <= 23:
                        return level

            return None

        except:
            return None

    def _read_elixir(self, frame: np.ndarray) -> Optional[float]:
        """Read elixir count (safe)"""

        x1, y1, x2, y2 = ELIXIR_REGION
        h, w = frame.shape[:2]

        if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return None

        elixir_img = frame[y1:y2, x1:x2]

        if elixir_img.size == 0:
            return None

        try:
            gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            binary = cv2.resize(binary, None, fx=3, fy=3)

            config = "--psm 7 --oem 3 -c tesseract_char_whitelist=0123456789./"
            text = pytesseract.image_to_string(binary, config=config).strip()

            if "/" in text:
                current = float(text.split("/")[0])
            else:
                current = float(text)

            return min(10.0, max(0.0, current))
        except:
            return None

    def _read_timer(self, frame: np.ndarray) -> Optional[int]:
        """Read match timer (safe)"""

        x1, y1, x2, y2 = TIMER_REGION
        h, w = frame.shape[:2]

        if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return None

        timer_img = frame[y1:y2, x1:x2]

        if timer_img.size == 0:
            return None

        try:
            gray = cv2.cvtColor(timer_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            binary = cv2.resize(binary, None, fx=3, fy=3)

            config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:"
            text = pytesseract.image_to_string(binary, config=config).strip()

            if ":" in text:
                parts = text.split(":")
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            return int(text)
        except:
            return None

    def visualize_state(self, frame: np.ndarray, state: Dict) -> np.ndarray:
        """Draw detections on frame with debug info"""

        if frame is None or frame.size == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        # Use debug frame if available (already has card detection visualization)
        if "debug_frame" in state:
            vis_frame = state["debug_frame"].copy()
        else:
            vis_frame = frame.copy()

        # Draw ally troops (green)
        for troop in state["troops"]["ally"]:
            x1, y1, x2, y2 = troop["bbox"]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            track_id = troop.get("track_id", "?")
            label = f"{troop['type']} #{track_id}"
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw enemy troops (red)
        for troop in state["troops"]["enemy"]:
            x1, y1, x2, y2 = troop["bbox"]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            track_id = troop.get("track_id", "?")
            label = f"{troop['type']} #{track_id}"
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # NOTE: Card slot visualization is now handled by CardDetector._draw_debug_visualization
        # Only draw if debug_frame is NOT available (fallback)
        if "debug_frame" not in state:
            h, w = vis_frame.shape[:2]
            for idx, card in enumerate(state["cards_in_hand"]):
                if idx < len(CARD_SLOTS):
                    x1, y1, x2, y2 = CARD_SLOTS[idx]
                    if x2 <= w and y2 <= h:
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(
                            vis_frame,
                            card,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

        # Draw elixir
        if state.get("elixir") is not None:
            cv2.putText(
                vis_frame,
                f"Elixir: {state['elixir']:.1f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

        # Draw timer
        if state.get("match_time") is not None:
            mins = state["match_time"] // 60
            secs = state["match_time"] % 60
            cv2.putText(
                vis_frame,
                f"Time: {mins}:{secs:02d}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )

        return vis_frame
