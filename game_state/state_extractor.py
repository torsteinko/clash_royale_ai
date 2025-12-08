# game_state/state_extractor.py
"""Simplified game state extractor for ally/enemy prefixed model"""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from typing import Dict, List, Optional, Tuple

from detection.card_detector_simple import CardDetector
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
            Updated detections with consistent IDs
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

                # Keep the tracked ID
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
    """Extract complete game state from screenshot using ally/enemy prefixed model"""

    def __init__(
        self,
        yolo_model_path="runs/synthetic/train_20251207_183426_single/weights/best.pt",
        deck: Optional[List[str]] = None,
    ):
        """
        Args:
            yolo_model_path: Path to YOLO model weights
            deck: Optional list of 8 card names in your deck (e.g., ["knight", "archer", ...])
                  Should be base card names WITHOUT "ally_" or "enemy_" prefix
                  Will filter card detections and ally troops to only these cards + evolutions
        """
        print("\n" + "=" * 80)
        print("🎮 INITIALIZING GAME STATE EXTRACTOR (SYNTHETIC MODEL)")
        if deck:
            print(f"🎴 Deck filtering enabled: {len(deck)} cards")
            print(f"   Cards: {', '.join(deck)}")
        else:
            print("🎴 No deck filtering (all cards detected)")
        print("=" * 80)

        # Normalize deck names: lowercase, strip any "ally_"/"enemy_" prefix
        if deck:
            normalized_deck = []
            for card in deck:
                card_lower = card.lower()
                # Remove team prefixes if user accidentally included them
                card_lower = card_lower.replace("ally_", "").replace("enemy_", "")
                normalized_deck.append(card_lower)
            self.deck = normalized_deck
            print(f"🎴 Deck filtering enabled: {len(self.deck)} cards")
            print(f"   Cards: {', '.join(self.deck)}")
        else:
            self.deck = None  # Load YOLO model with fallback mechanism
        print("📦 Loading YOLO model...")

        # Try to load the requested model
        model_path = Path(yolo_model_path)
        fallback_models = [
            "runs/synthetic/train_20251206_111056_single/weights/last.pt",
            "runs/synthetic/train_20251206_111056_single/weights/epoch10.pt",
            "best.pt",  # Repo root fallback
            "yolo11n.pt",  # Base model fallback
        ]

        model_loaded = False
        for attempt_path in [yolo_model_path] + fallback_models:
            try:
                p = Path(attempt_path)
                # Check if file exists and has reasonable size (>1MB for a trained model)
                if p.exists() and p.stat().st_size > 1_000_000:
                    self.yolo = YOLO(str(p))
                    print(f"   ✅ YOLO loaded from: {p}")
                    model_loaded = True
                    break
                elif p.exists():
                    print(f"   ⚠️  Skipping {p.name} (file too small, likely corrupted)")
            except Exception as e:
                print(f"   ⚠️  Failed to load {attempt_path}: {e}")
                continue

        if not model_loaded:
            print("   ⚠️  All model paths failed, loading base YOLO11n...")
            self.yolo = YOLO("yolo11n.pt")
            print("   ⚠️  WARNING: Using base model, not trained for Clash Royale!")

        # Load card detector
        print("🎴 Loading card templates...")
        self.card_detector = CardDetector(deck=self.deck)
        print(f"   ✅ Card detector ready")

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

        # 1. Detect troops (with debug) - now uses ally/enemy prefixes
        raw_troops = self._detect_troops(frame, debug=debug)

        # 2. Filter ally troops by deck if specified
        ally_troops_filtered = raw_troops["ally"]
        if self.deck:
            ally_troops_filtered = self._filter_troops_by_deck(ally_troops_filtered)
            # Also update debug frame to only show filtered troops
            if debug and "debug_frame" in raw_troops:
                raw_troops["debug_frame"] = self._redraw_debug_frame(
                    frame, ally_troops_filtered, raw_troops["enemy"]
                )

        # Store debug frame if available
        if debug and "debug_frame" in raw_troops:
            state["debug_frame"] = raw_troops["debug_frame"]

        # 3. Apply tracking
        ally_troops = self.troop_tracker.update(ally_troops_filtered)
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

        # 4. Detect towers from YOLO detections
        tower_info = self._detect_towers_from_yolo(raw_troops)
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
        """Detect troops using YOLO with ally/enemy prefixed classes"""

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

                # Ignore troops under y=1000 (game UI)
                if y2 < 1000:
                    continue

                # Parse team from class name prefix
                if class_name.startswith("ally_"):
                    team = "ally"
                    troop_type = class_name[5:]  # Remove "ally_" prefix
                elif class_name.startswith("enemy_"):
                    team = "enemy"
                    troop_type = class_name[6:]  # Remove "enemy_" prefix
                else:
                    # Non-troop classes (UI elements, etc.) - skip
                    continue

                # Try to detect level using OCR (useful for enemy troops)
                level = None
                if not self._is_tower(troop_type):
                    level = self._detect_level_ocr(
                        frame, int(x1), int(y1), int(x2), int(y2)
                    )

                troop_data = {
                    "type": troop_type,
                    "position": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "team": team,
                    "level": level,
                }

                # Draw debug box if enabled
                if debug_frame is not None:
                    color = (0, 255, 0) if team == "ally" else (0, 0, 255)
                    cv2.rectangle(
                        debug_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    label = f"{team[:1].upper()}: {troop_type}"
                    if level:
                        label += f" L{level}"
                    label += f" {conf:.2f}"
                    cv2.putText(
                        debug_frame,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                if team == "ally":
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
        tower_keywords = ["tower", "king"]
        return any(kw in class_name.lower() for kw in tower_keywords)

    def _filter_troops_by_deck(self, troops: List[Dict]) -> List[Dict]:
        """
        Filter ally troops to only include cards from the deck (including evolutions)

        Note: troop["type"] already has "ally_" prefix removed by _detect_troops()
        So we can directly compare against deck card names
        """
        if not self.deck:
            return troops

        filtered = []
        for troop in troops:
            troop_type = troop.get("type", "").lower()

            # Skip towers - always show them
            if self._is_tower(troop_type):
                filtered.append(troop)
                continue

            # Handle evolution suffix (e.g., "knight_evolution" matches "knight")
            base_type = troop_type.replace("_evolution", "").replace("_evo", "")

            # Handle building prefix (e.g., "building_cannon" matches "cannon")
            base_type_no_building = base_type.replace("building_", "")

            # Check if troop matches any card in deck
            for deck_card in self.deck:
                deck_card_lower = deck_card.lower()

                # Exact match or base match
                if (
                    troop_type == deck_card_lower
                    or base_type == deck_card_lower
                    or base_type_no_building == deck_card_lower
                ):
                    filtered.append(troop)
                    break

                # Partial match (e.g., "goblin_gang" contains "goblin")
                if deck_card_lower in troop_type or deck_card_lower in base_type:
                    filtered.append(troop)
                    break

        return filtered

    def _redraw_debug_frame(
        self, frame: np.ndarray, ally_troops: List[Dict], enemy_troops: List[Dict]
    ) -> np.ndarray:
        """Redraw debug frame with only the filtered troops"""
        debug_frame = frame.copy()

        # Draw ally troops (green)
        for troop in ally_troops:
            bbox = troop.get("bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            level = troop.get("level")
            conf = troop.get("confidence", 0)
            troop_type = troop.get("type", "")

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"A: {troop_type}"
            if level:
                label += f" L{level}"
            label += f" {conf:.2f}"
            cv2.putText(
                debug_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Draw enemy troops (red)
        for troop in enemy_troops:
            bbox = troop.get("bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            level = troop.get("level")
            conf = troop.get("confidence", 0)
            troop_type = troop.get("type", "")

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"E: {troop_type}"
            if level:
                label += f" L{level}"
            label += f" {conf:.2f}"
            cv2.putText(
                debug_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return debug_frame

    def _detect_towers_from_yolo(self, raw_troops: Dict) -> Dict:
        """
        Detect tower status from YOLO detections

        Instead of OCR, we check for presence of tower detections:
        - ally_princess_tower (left/right based on x position)
        - enemy_princess_tower (left/right based on x position)
        - ally_king / enemy_king

        Returns dict with tower status (alive/down) for each position
        """

        towers = {
            "ally": {
                "left_princess": {"status": "down", "hp": 0, "max_hp": 0, "level": 0},
                "right_princess": {"status": "down", "hp": 0, "max_hp": 0, "level": 0},
                "king": {"status": "down", "hp": 0, "max_hp": 0, "level": 0},
            },
            "enemy": {
                "left_princess": {"status": "down", "hp": 0, "max_hp": 0, "level": 0},
                "right_princess": {"status": "down", "hp": 0, "max_hp": 0, "level": 0},
                "king": {"status": "down", "hp": 0, "max_hp": 0, "level": 0},
            },
        }

        # Process all detections (ally + enemy combined)
        all_detections = raw_troops.get("ally", []) + raw_troops.get("enemy", [])

        for detection in all_detections:
            troop_type = detection.get("type", "")
            team = detection.get("team", "")
            bbox = detection.get("bbox", (0, 0, 0, 0))
            level = detection.get("level")

            # Calculate center x position
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2

            # Check if it's a tower detection
            if "princess_tower" in troop_type:
                # Determine left/right based on x position
                # Screen center is roughly 360 (720/2)
                if center_x < 360:
                    position = "left_princess"
                else:
                    position = "right_princess"

                towers[team][position]["status"] = "alive"
                if level:
                    towers[team][position]["level"] = level

            elif "king" in troop_type and "tower" not in troop_type:
                # King tower detected
                towers[team]["king"]["status"] = "alive"
                if level:
                    towers[team]["king"]["level"] = level

        return towers

    def _detect_level_ocr(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Optional[int]:
        """
        Detect level number (1-23) at TOP of unit bbox

        Level badge is INSIDE the unit bbox at the top, not above it!
        This is now used to extract level information, not for team detection.
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
                    if 3 <= level <= 23:
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
            if troop.get("level"):
                label += f" L{troop['level']}"
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
            if troop.get("level"):
                label += f" L{troop['level']}"
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
