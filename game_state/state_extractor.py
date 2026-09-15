# game_state/state_extractor.py
"""Game state extractor for the ally/enemy prefixed YOLO model.

Fixes applied Sept 2026 (see git history / PR description):
  * The arena/UI y-filter was inverted (`if y2 < 1000: continue` kept ONLY the
    bottom UI strip and dropped all gameplay) -- now everything whose top edge
    is below ARENA_BOTTOM_Y is treated as UI, plus meta/decorative classes are
    filtered by name.
  * Team ownership now comes from the rendered blue/red tint
    (game_state.team_classifier), with the model's ally_/enemy_ prefix as a
    hint and position only as a last resort.  Works for taxonomy v2 models
    that have no side prefix.
  * Elixir is read from the purple bar (pixel math, no OCR) and the timer via
    OCRReader; the old per-file tesseract paths/regions were broken on Linux
    and wrong on Windows.
  * Cards-in-hand names are normalized (no "base/" prefix, waiting -> unknown)
    and the deck-cycle snapshot is exposed via state["deck"].
"""

import cv2
import numpy as np
import time
import logging
import platform
import shutil
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from typing import Dict, List, Optional, Tuple

from detection.card_detector_simple import CardDetector
from detection.ocr_reader import OCRReader
from config.game_config import CARD_SLOTS, ARENA_BOTTOM_Y, is_junk_class
from game_state.team_classifier import classify_team, TeamVoter

logger = logging.getLogger(__name__)


class TroopTracker:
    """Track troops across frames for consistency.

    Matching requires the same unit type AND compatible teams, so an ally
    knight and an enemy knight near each other cannot swap identities.
    Supports wall-clock aging (live play) and frame-count aging (offline
    batch processing of recordings).
    """

    def __init__(
        self,
        max_age: float = 1.0,
        max_distance: float = 150,
        frame_based: bool = False,
        max_age_frames: int = 3,
    ):
        self.tracked_troops = {}  # {troop_id: {data}}
        self.next_id = 0
        self.max_age = max_age  # Forget troops after N seconds (live mode)
        self.max_age_frames = max_age_frames  # ... or N frames (batch mode)
        self.max_distance = max_distance  # Max pixels a troop can move between frames
        self.frame_based = frame_based
        self.frame_count = 0
        self.voter = TeamVoter()

    def update(self, new_detections: List[Dict]) -> List[Dict]:
        """
        Match new detections with tracked troops

        Args:
            new_detections: List of troops detected this frame

        Returns:
            Updated detections with consistent IDs (and temporally-voted teams)
        """

        current_time = time.time()
        self.frame_count += 1

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

                # ... and compatible teams (never match blue vs red)
                det_team = detection.get("team")
                trk_team = tracked.get("team")
                if (
                    det_team in ("ally", "enemy")
                    and trk_team in ("ally", "enemy")
                    and det_team != trk_team
                ):
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

                # Temporal team vote (stable across spell flashes / occlusion)
                voted, _mean = self.voter.vote(troop_id, detection.get("team_score"))
                if voted is not None:
                    detection["team"] = voted

                # Update tracked data
                tracked["position"] = detection["position"]
                tracked["bbox"] = detection["bbox"]
                tracked["team"] = detection["team"]
                tracked["last_seen"] = current_time
                tracked["last_seen_frame"] = self.frame_count

                matched_detections.append(detection)
                unmatched_new.remove(best_match_idx)

        # Handle unmatched new detections (new troops)
        for idx in unmatched_new:
            detection = new_detections[idx]

            # Assign new track ID
            troop_id = self.next_id
            self.next_id += 1

            detection["track_id"] = troop_id

            voted, _mean = self.voter.vote(troop_id, detection.get("team_score"))
            if voted is not None:
                detection["team"] = voted

            # Store in tracker
            self.tracked_troops[troop_id] = {
                "type": detection["type"],
                "team": detection["team"],
                "position": detection["position"],
                "bbox": detection["bbox"],
                "last_seen": current_time,
                "last_seen_frame": self.frame_count,
            }

            matched_detections.append(detection)

        # Sort by track id for stable output ordering
        matched_detections.sort(key=lambda d: d.get("track_id", 0))

        return matched_detections

    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _cleanup_old_tracks(self, current_time: float):
        """Remove tracks that haven't been seen recently"""
        to_remove = []

        for troop_id, tracked in self.tracked_troops.items():
            if self.frame_based:
                age_frames = self.frame_count - tracked.get("last_seen_frame", 0)
                if age_frames > self.max_age_frames:
                    to_remove.append(troop_id)
            else:
                if current_time - tracked["last_seen"] > self.max_age:
                    to_remove.append(troop_id)

        for troop_id in to_remove:
            del self.tracked_troops[troop_id]
            self.voter.forget(troop_id)

    def reset(self):
        """Clear all tracks"""
        self.tracked_troops.clear()
        self.next_id = 0
        self.frame_count = 0
        self.voter.reset()


class GameStateExtractor:
    """Extract complete game state from screenshot using ally/enemy prefixed model"""

    def __init__(
        self,
        yolo_model_path="runs/synthetic/train_20251207_183426_single/weights/best.pt",
        deck: Optional[List[str]] = None,
        read_levels: bool = True,
        frame_based_tracking: bool = False,
    ):
        """
        Args:
            yolo_model_path: Path to YOLO model weights
            deck: Optional list of 8 card names in your deck (e.g., ["knight", "archer", ...])
                  Should be base card names WITHOUT "ally_" or "enemy_" prefix
                  Will filter card detections and ally troops to only these cards + evolutions
            read_levels: Read unit level badges via OCR (slow; disable for batch runs)
            frame_based_tracking: Age troop tracks by frames instead of wall clock
                                  (use for offline processing of recordings)
        """
        self.read_levels = read_levels
        self.frame_based_tracking = frame_based_tracking
        self.team_mismatches = 0
        self.skipped_unknown_team = 0

        print("\n" + "=" * 80)
        print("🎮 INITIALIZING GAME STATE EXTRACTOR (SYNTHETIC MODEL)")
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
        self.troop_tracker = TroopTracker(frame_based=frame_based_tracking)
        print(f"   ✅ Troop tracker ready")

        # Configure OCR.  Only override the tesseract path on Windows and only
        # when the file actually exists; on Linux/macOS pytesseract finds the
        # binary on PATH.  (The previous code hardcoded a Windows path for all
        # platforms, silently disabling OCR on Linux.)
        if platform.system() == "Windows":
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(win_path).exists():
                pytesseract.pytesseract.tesseract_cmd = win_path
        self.ocr_enabled = shutil.which("tesseract") is not None or Path(
            pytesseract.pytesseract.tesseract_cmd
        ).exists()
        self.ocr = OCRReader() if self.ocr_enabled else None
        if not self.ocr_enabled:
            print("   ⚠️  Tesseract not found, OCR disabled")

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
        detections = raw_troops["detections"]

        # 2. Filter ally troops by deck if specified
        if self.deck:
            ally_dets = [d for d in detections if d.get("team") == "ally"]
            other_dets = [d for d in detections if d.get("team") != "ally"]
            detections = other_dets + self._filter_troops_by_deck(ally_dets)

        # Store debug frame if available
        if debug and "debug_frame" in raw_troops:
            state["debug_frame"] = raw_troops["debug_frame"]

        # 3. Apply tracking (single pass so ally/enemy cannot cross-match)
        tracked = self.troop_tracker.update(detections)

        ally_troops = [d for d in tracked if d.get("team") == "ally"]
        enemy_troops = [d for d in tracked if d.get("team") == "enemy"]
        unknown_troops = [d for d in tracked if d.get("team") not in ("ally", "enemy")]

        state["troops"] = {
            "ally": ally_troops,
            "enemy": enemy_troops,
            "unknown": unknown_troops,
            "total_ally": len(ally_troops),
            "total_enemy": len(enemy_troops),
        }

        # 4. Detect cards in hand (with debug info if enabled)
        if debug and hasattr(self.card_detector, "detect_cards_with_debug"):
            cards_debug = self.card_detector.detect_cards_with_debug(
                frame, state.get("debug_frame")
            )
            state["cards_in_hand"] = [
                self._normalize_card(c["name"]) for c in cards_debug
            ]
            state["cards_debug"] = cards_debug
        else:
            state["cards_in_hand"] = [
                self._normalize_card(c)
                for c in self.card_detector.detect_cards_in_hand(frame)
            ]

        # Deck cycle tracking snapshot (rotation queue + next-card prediction)
        deck_tracker = getattr(self.card_detector, "deck_tracker", None)
        state["deck"] = deck_tracker.snapshot() if deck_tracker is not None else None

        # 5. Detect towers from YOLO detections
        tower_info = self._detect_towers_from_yolo(raw_troops)
        state["towers"] = tower_info

        # 6. Elixir (purple bar, pixel math) + timer (OCR)
        state["elixir"] = None
        state["match_time"] = None
        state["match_time_str"] = None

        if self.ocr_enabled and self.ocr is not None:
            try:
                state["elixir"] = self._read_elixir(frame)
            except Exception as exc:
                logger.debug("elixir read failed: %s", exc)

            try:
                timer_str = self.ocr.read_timer(frame)
                state["match_time_str"] = timer_str
                state["match_time"] = self._parse_timer(timer_str)
            except Exception as exc:
                logger.debug("timer read failed: %s", exc)

        # 7. Derived metrics
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
            "troops": {
                "ally": [],
                "enemy": [],
                "unknown": [],
                "total_ally": 0,
                "total_enemy": 0,
            },
            "cards_in_hand": ["unknown", "unknown", "unknown", "unknown"],
            "elixir": None,
            "match_time": None,
            "match_time_str": None,
            "towers": {},
            "deck": None,
            "is_double_elixir": False,
            "is_overtime": False,
        }

    # ------------------------------------------------------------------ troops
    def _detect_troops(self, frame: np.ndarray, debug: bool = False) -> Dict:
        """Detect troops using YOLO; resolve team from blue/red tint."""

        try:
            # NOTE: imgsz must match the detector's training resolution
            # (YOLO11/yolo11l/args.yaml: imgsz=640).  Running inference at
            # 1280 silently dropped real units (knight, giant) while keeping
            # UI/decor false positives -- verified Sept 2026.
            results = self.yolo.predict(frame, conf=0.5, imgsz=640, verbose=False)
        except Exception as exc:
            logger.warning("YOLO predict failed: %s", exc)
            return {"detections": []}

        detections = []

        # Create debug frame if needed
        debug_frame = frame.copy() if debug else None

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]

                # Skip detections that live in the bottom UI strip (hand cards,
                # elixir bar, buttons).  Everything overlapping the arena stays.
                if y1 >= ARENA_BOTTOM_Y:
                    continue

                # Skip meta / decorative classes by name (bars, text, emotes,
                # static map decorations, ...)
                if is_junk_class(class_name):
                    continue

                # Team hint from the class prefix (current 296-class model)
                class_lower = class_name.lower()
                if class_lower.startswith("ally_"):
                    team_hint, troop_type = "ally", class_name[5:]
                elif class_lower.startswith("enemy_"):
                    team_hint, troop_type = "enemy", class_name[6:]
                else:
                    # Taxonomy v2: no side prefix -- tint decides.
                    team_hint, troop_type = None, class_name

                # Team from the rendered blue/red tint (the game's own label).
                tint_team, tint_score = classify_team(frame, (x1, y1, x2, y2))
                if tint_team is not None:
                    team = tint_team
                    source = "tint"
                    if team_hint is not None and team_hint != team:
                        self.team_mismatches += 1
                        logger.debug(
                            "team-hint mismatch: %s classified %s (score %.2f)",
                            class_name,
                            team,
                            tint_score if tint_score is not None else 0.0,
                        )
                elif team_hint is not None:
                    team = team_hint
                    source = "prefix"
                else:
                    # Last resort for unprefixed classes with ambiguous tint:
                    # position prior (top half = enemy).  Rare; logged.
                    center_y = (y1 + y2) / 2
                    team = "ally" if center_y > frame.shape[0] / 2 else "enemy"
                    source = "position"
                    self.skipped_unknown_team += 1

                # Try to detect level using OCR (useful for enemy troops)
                level = None
                if self.read_levels and not self._is_tower(troop_type):
                    level = self._detect_level_ocr(
                        frame, int(x1), int(y1), int(x2), int(y2)
                    )

                troop_data = {
                    "type": troop_type,
                    "position": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "team": team,
                    "team_score": tint_score,
                    "team_source": source,
                    "level": level,
                }
                detections.append(troop_data)

                # Draw debug box if enabled
                if debug_frame is not None:
                    color = (
                        (0, 255, 0)
                        if team == "ally"
                        else (0, 0, 255)
                        if team == "enemy"
                        else (160, 160, 160)
                    )
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
                    if tint_score is not None:
                        label += f" t{tint_score:+.2f}"
                    cv2.putText(
                        debug_frame,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        result = {"detections": detections}
        if debug:
            result["debug_frame"] = debug_frame
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_card(name) -> str:
        """Strip folder prefixes; map placeholders to 'unknown'."""
        if not name:
            return "unknown"
        n = str(name).split("/")[-1]
        if n.startswith("waiting_for_card"):
            return "unknown"
        if n in ("unknown",):
            return "unknown"
        return n

    @staticmethod
    def _parse_timer(timer_str: Optional[str]) -> Optional[int]:
        """'M:SS' or 'MM:SS' -> seconds since start of the match clock."""
        if not timer_str or ":" not in timer_str:
            return None
        try:
            mins, secs = timer_str.split(":")
            return int(mins) * 60 + int(secs)
        except (ValueError, TypeError):
            return None

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
        """Redraw debug frame with only the filtered troops (legacy helper)."""
        debug_frame = frame.copy()

        for troops, color, tag in (
            (ally_troops, (0, 255, 0), "A"),
            (enemy_troops, (0, 0, 255), "E"),
        ):
            for troop in troops:
                bbox = troop.get("bbox", (0, 0, 0, 0))
                x1, y1, x2, y2 = bbox
                level = troop.get("level")
                conf = troop.get("confidence", 0)
                troop_type = troop.get("type", "")

                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{tag}: {troop_type}"
                if level:
                    label += f" L{level}"
                label += f" {conf:.2f}"
                cv2.putText(
                    debug_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        return debug_frame

    # ------------------------------------------------------------------ towers
    def _detect_towers_from_yolo(self, raw_troops: Dict) -> Dict:
        """
        Detect tower status from YOLO detections

        Instead of OCR, we check for presence of tower detections:
        - princess_tower (left/right based on x position)
        - king tower

        Returns dict with tower status (alive/down) for each position, plus
        ally_towers_alive / enemy_towers_alive counts for report compatibility.
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

        # Process all detections (combined list)
        all_detections = raw_troops.get("detections", [])

        for detection in all_detections:
            troop_type = detection.get("type", "").lower()
            team = detection.get("team", "")
            bbox = detection.get("bbox", (0, 0, 0, 0))
            level = detection.get("level")

            if team not in ("ally", "enemy"):
                continue

            # Calculate center x position
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2

            # Check if it's a princess-side tower (the 296-class taxonomy also
            # fires `building_tower` on the same objects -- treat as the same).
            if "princess_tower" in troop_type or troop_type.endswith("building_tower"):
                # Determine left/right based on x position
                # Screen center is roughly 360 (720/2)
                if center_x < 360:
                    position = "left_princess"
                else:
                    position = "right_princess"

                towers[team][position]["status"] = "alive"
                if level:
                    towers[team][position]["level"] = level

            elif (
                "king" in troop_type
                and "bar" not in troop_type
                and "skill" not in troop_type
                and "tower_bar" not in troop_type
            ):
                # King tower detected ('king' / 'king_tower' classes)
                towers[team]["king"]["status"] = "alive"
                if level:
                    towers[team]["king"]["level"] = level

        # Report-compatible alive counts
        for side in ("ally", "enemy"):
            towers[f"{side}_towers_alive"] = sum(
                1
                for pos in ("left_princess", "right_princess", "king")
                if towers[side][pos]["status"] == "alive"
            )

        return towers

    # --------------------------------------------------------------------- OCR
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
                    if 3 <= level <= 23:
                        return level

            return None

        except Exception:
            return None

    def _read_elixir(self, frame: np.ndarray) -> Optional[float]:
        """Read elixir from the purple bar (pixel math, no OCR)."""
        if self.ocr is None:
            return None
        return self.ocr.read_elixir(frame)

    def _read_timer(self, frame: np.ndarray) -> Optional[int]:
        """Read match timer via OCR; returns seconds as int (legacy API)."""
        if self.ocr is None:
            return None
        return self._parse_timer(self.ocr.read_timer(frame))

    # ------------------------------------------------------------------- debug
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

        # Draw unknown-team troops (gray)
        for troop in state["troops"].get("unknown", []):
            x1, y1, x2, y2 = troop["bbox"]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (160, 160, 160), 2)
            cv2.putText(
                vis_frame,
                f"? {troop['type']}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (160, 160, 160),
                1,
            )

        # Card slot visualization fallback
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

        # Deck cycle info (next card prediction)
        deck = state.get("deck")
        if deck and deck.get("next"):
            cv2.putText(
                vis_frame,
                f"Next: {deck['next']}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        return vis_frame
