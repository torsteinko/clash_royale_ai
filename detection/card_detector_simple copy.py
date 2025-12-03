# detection/card_detector_simple.py
"""Simple card detector using only color template matching"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from config.game_config import CARD_SLOTS
from detection.ocr_reader import OCRReader


class CardDetectorSimple:
    """Detect cards in hand using simple color template matching"""

    # CROP SETTINGS - Adjust these to change what area is matched
    CROP_TOP = 0.17  # Remove top border
    CROP_BOTTOM = 0.28  # Remove elixir + bottom border
    CROP_SIDE = 0.12  # Remove side borders

    # Card selection lift offset (when player taps a card)
    CARD_LIFT_OFFSET = 21

    # Persistence settings
    CONFIDENCE_THRESHOLD = 5  # Frames before we lock in a card
    MAX_WAITING_FRAMES = 25  # Maximum frames waiting_for_card can appear

    # Special card names (can be with or without folder prefix)
    WAITING_CARD_NAMES = ["waiting_for_card", "other/waiting_for_card"]

    def __init__(self, templates_dir="data/card_templates"):
        self.templates_dir = Path(templates_dir)
        print(
            f"🔍 Initializing CardDetectorSimple with templates from: {self.templates_dir}"
        )
        print(f"   Full path: {self.templates_dir.resolve()}")

        self.templates = {}  # Color templates
        self.templates_gray = {}  # Grayscale templates
        self._load_templates()

        # Initialize OCR reader
        self.ocr_reader = OCRReader()

        # Card persistence - remember locked cards per slot
        self._locked_cards = ["unknown", "unknown", "unknown", "unknown"]

        # Track consecutive detections per slot
        self._detection_counts = [
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
        ]

        # Track waiting_for_card state per slot
        self._waiting_state = [
            {"active": False, "frames": 0},
            {"active": False, "frames": 0},
            {"active": False, "frames": 0},
            {"active": False, "frames": 0},
        ]

        # Cache/resizing/histogram helpers
        self._template_resize_cache = (
            {}
        )  # key: (name,w,h,use_gray) -> resized np.ndarray
        self._template_histograms = {}  # key: name -> cv2 hist (color)
        self._precomputed = False

        # Deck inference (speed optimization)
        self._seen_card_names = []  # ordered seen non-waiting names (unique)
        self._seen_card_set = set()
        self.active_deck = None  # list of 8 base names when inferred
        self.deck_confirmed = False
        self.allowed_template_names = None  # dict of templates to restrict matching to

        # OCR throttle
        self._frame_count = 0
        self.ocr_read_every_n_frames = 2
        self._current_elixir = None  # cached per-frame elixir read

        # Precompute histograms after templates loaded
        self._precompute_template_histograms()

    def _load_templates(self):
        """Load all card templates from subdirectories"""
        if not self.templates_dir.exists():
            print(f"⚠️  Templates directory not found: {self.templates_dir}")
            return

        print(f"🔍 Loading card templates from: {self.templates_dir}")

        template_files = (
            list(self.templates_dir.rglob("*.png"))
            + list(self.templates_dir.rglob("*.jpg"))
            + list(self.templates_dir.rglob("*.jpeg"))
        )

        if not template_files:
            print(f"⚠️  No image files found in {self.templates_dir} or subdirectories")
            return

        print(f"   Found {len(template_files)} template files")

        for template_path in template_files:
            relative_path = template_path.relative_to(self.templates_dir)
            parent = relative_path.parent
            if str(parent) != ".":
                card_name = f"{parent}/{template_path.stem}"
            else:
                card_name = template_path.stem

            template = cv2.imread(str(template_path))
            if template is not None:
                self.templates[card_name] = template
                # Also create grayscale version
                self.templates_gray[card_name] = cv2.cvtColor(
                    template, cv2.COLOR_BGR2GRAY
                )

        # Alias waiting_for_card
        for name in list(self.templates.keys()):
            if "waiting_for_card" in name and "waiting_for_card" not in self.templates:
                self.templates["waiting_for_card"] = self.templates[name]
                self.templates_gray["waiting_for_card"] = self.templates_gray[name]
                print(f"   ✓ Aliased '{name}' as 'waiting_for_card'")

        print(f"   ✅ Loaded {len(self.templates)} templates")

        # Print organized summary
        if self.templates:
            print(f"\n📊 Template categories:")
            categories = {}
            for card_name in self.templates.keys():
                category = card_name.split("/")[0] if "/" in card_name else "root"
                categories[category] = categories.get(category, 0) + 1

            for category, count in sorted(categories.items()):
                print(f"   • {category}: {count} cards")

    def _precompute_template_histograms(self):
        """Compute small color histograms for each color template to allow fast shortlist."""
        if self._precomputed or not self.templates:
            return

        for name, tpl in self.templates.items():
            try:
                # Resize small for histogram (keep aspect but small)
                small = cv2.resize(tpl, (64, 64), interpolation=cv2.INTER_AREA)
                hist = cv2.calcHist(
                    [small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                cv2.normalize(hist, hist)
                self._template_histograms[name] = hist.flatten()
            except Exception:
                continue

        self._precomputed = True

    def _compute_color_hist(self, image: np.ndarray):
        """Compute same histogram for a queried card region (small & normalized)."""
        if image is None or image.size == 0:
            return None
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist(
            [small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _shortlist_templates_by_hist(self, card_artwork: np.ndarray, top_k: int = 12):
        """Return top_k template names most similar by histogram (fast)."""
        query_hist = self._compute_color_hist(card_artwork)
        if query_hist is None or not self._template_histograms:
            return list(self.templates.keys())

        # Compare via correlation (higher is more similar)
        scores = []
        # choose candidates source (allowed vs all)
        candidate_names = (
            self.allowed_template_names
            if self.deck_confirmed and self.allowed_template_names
            else list(self.templates.keys())
        )

        for name in candidate_names:
            tpl_hist = self._template_histograms.get(name)
            if tpl_hist is None:
                continue
            # use dot product as proxy (histograms are normalized)
            score = float(np.dot(query_hist, tpl_hist))
            scores.append((score, name))

        if not scores:
            return candidate_names

        scores.sort(reverse=True)
        shortlisted = [n for _, n in scores[:top_k]]
        return shortlisted

    def _cache_resize_template(
        self, name: str, target_w: int, target_h: int, use_gray: bool = False
    ):
        """Return cached resized template or resize+cache it."""
        key = (name, target_w, target_h, bool(use_gray))
        if key in self._template_resize_cache:
            return self._template_resize_cache[key]

        tpl = (
            self.templates_gray[name]
            if use_gray and name in self.templates_gray
            else self.templates[name]
        )
        if tpl is None:
            return None

        try:
            resized = cv2.resize(
                tpl, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            if use_gray and len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            self._template_resize_cache[key] = resized
            return resized
        except Exception:
            return None

    def detect_cards_with_debug(
        self, frame: np.ndarray, debug_frame: np.ndarray = None
    ) -> List[dict]:
        """Detect cards with debug visualization"""

        detected_cards = []

        # Throttle OCR reads: read elixir only every N frames
        self._frame_count += 1
        if (self._frame_count % self.ocr_read_every_n_frames) == 0:
            self._current_elixir = self.ocr_reader.read_elixir(frame)

        for slot_idx, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
            h, w = frame.shape[:2]

            if x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                detected_cards.append(
                    {"name": "unknown", "score": 0.0, "method": "invalid"}
                )
                continue

            # Check if card is grayed out (unaffordable)
            is_gray = self.ocr_reader.detect_gray_card(frame, (x1, y1, x2, y2))

            # Try normal position first
            result = self._detect_card_at_position(
                frame, x1, y1, x2, y2, use_gray=is_gray
            )

            # If low score, try lifted position (card selected by player)
            if result["score"] < 0.5:
                lifted_y1 = max(0, y1 - self.CARD_LIFT_OFFSET)
                lifted_y2 = max(0, y2 - self.CARD_LIFT_OFFSET)

                if lifted_y1 < lifted_y2:
                    lifted_result = self._detect_card_at_position(
                        frame, x1, lifted_y1, x2, lifted_y2, use_gray=is_gray
                    )
                    if lifted_result["score"] > result["score"]:
                        result = lifted_result
                        result["method"] += "_lifted"

            # Apply persistence and transition validation
            result = self._apply_persistence_logic(slot_idx, result)

            # Add elixir info to result
            result["is_gray"] = is_gray
            result["current_elixir"] = self._current_elixir

            detected_cards.append(result)

            # Draw debug visualization
            if debug_frame is not None:
                self._draw_debug_visualization(
                    debug_frame, frame, slot_idx, x1, y1, x2, y2, result
                )

        return detected_cards

    def _detect_card_at_position(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        use_gray: bool = False,
    ) -> dict:
        """Detect a card at a specific position"""

        h, w = frame.shape[:2]

        if x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return {"name": "unknown", "score": 0.0, "method": "invalid"}

        card_region = frame[y1:y2, x1:x2]

        if card_region.size == 0:
            return {"name": "unknown", "score": 0.0, "method": "empty"}

        # TIGHT CROP: Remove borders and elixir
        card_height = card_region.shape[0]
        card_width = card_region.shape[1]

        crop_top = int(card_height * self.CROP_TOP)
        crop_bottom = int(card_height * (1 - self.CROP_BOTTOM))
        crop_left = int(card_width * self.CROP_SIDE)
        crop_right = int(card_width * (1 - self.CROP_SIDE))

        card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]

        if card_artwork.size == 0:
            return {"name": "unknown", "score": 0.0, "method": "empty"}

        # Find best matching template
        best_name, best_score = self._find_best_match(card_artwork, use_gray=use_gray)

        method = "gray" if use_gray else "color"
        return {"name": best_name, "score": best_score, "method": method}

    def _find_best_match(
        self, card_artwork: np.ndarray, use_gray: bool = False
    ) -> Tuple[str, float]:
        """Find the best matching template using histogram shortlist + template matching"""

        if len(self.templates) == 0:
            return "unknown", 0.0

        best_match = "unknown"
        best_score = 0.0

        # Shortlist templates by histogram similarity (fast)
        shortlist = self._shortlist_templates_by_hist(card_artwork, top_k=12)

        # If deck confirmed, ensure shortlist comes from allowed templates
        if self.deck_confirmed and self.allowed_template_names:
            shortlist = [n for n in shortlist if n in self.allowed_template_names]

        # If shortlist is empty fallback to some allowed/all
        if not shortlist:
            shortlist = (
                self.allowed_template_names
                if self.deck_confirmed and self.allowed_template_names
                else list(self.templates.keys())
            )

        target_h, target_w = card_artwork.shape[:2]

        for card_name in shortlist:
            try:
                template_resized = self._cache_resize_template(
                    card_name, target_w, target_h, use_gray=use_gray
                )
                if template_resized is None:
                    continue

                # Template matching (same-size -> 1x1 result)
                result = cv2.matchTemplate(
                    (
                        card_artwork
                        if not use_gray
                        else (
                            cv2.cvtColor(card_artwork, cv2.COLOR_BGR2GRAY)
                            if len(card_artwork.shape) == 3
                            else card_artwork
                        )
                    ),
                    template_resized,
                    cv2.TM_CCOEFF_NORMED,
                )
                score = float(np.max(result))

                # quick accept if extremely good match
                if score > 0.87:
                    return card_name, score

                if score > best_score:
                    best_score = score
                    best_match = card_name

            except Exception:
                continue

        return best_match, best_score

    def _is_waiting_for_card(self, card_name: str) -> bool:
        """Check if the detected card is a waiting_for_card placeholder"""
        return card_name in self.WAITING_CARD_NAMES or card_name.endswith(
            "waiting_for_card"
        )

    def _apply_persistence_logic(self, slot_idx: int, result: dict) -> dict:
        """
        Apply persistence logic with transition validation:
        - Need 5 consecutive frames to lock in a card
        - Card can only change via waiting_for_card
        - waiting_for_card lasts max 25 frames
        """
        detected_name = result["name"]
        locked_card = self._locked_cards[slot_idx]
        detection_info = self._detection_counts[slot_idx]
        waiting_state = self._waiting_state[slot_idx]

        # Handle waiting_for_card detection
        if self._is_waiting_for_card(detected_name):
            waiting_state["active"] = True
            waiting_state["frames"] = 0
            self._locked_cards[slot_idx] = "unknown"
            detection_info["card"] = "waiting_for_card"
            detection_info["count"] = 1
            return {
                "name": "waiting_for_card",
                "score": result["score"],
                "method": "waiting",
            }

        # If in waiting state, track frames
        if waiting_state["active"]:
            waiting_state["frames"] += 1

            if detected_name != "unknown" and not self._is_waiting_for_card(
                detected_name
            ):
                if detection_info["card"] == detected_name:
                    detection_info["count"] += 1
                else:
                    detection_info["card"] = detected_name
                    detection_info["count"] = 1

                if detection_info["count"] >= self.CONFIDENCE_THRESHOLD:
                    self._locked_cards[slot_idx] = detected_name
                    waiting_state["active"] = False
                    waiting_state["frames"] = 0
                    return {
                        "name": detected_name,
                        "score": result["score"],
                        "method": result["method"] + "_confirmed",
                    }
                else:
                    return {
                        "name": "waiting_for_card",
                        "score": 0.0,
                        "method": f"confirming_{detection_info['count']}/{self.CONFIDENCE_THRESHOLD}",
                    }

            if waiting_state["frames"] >= self.MAX_WAITING_FRAMES:
                waiting_state["active"] = False
                waiting_state["frames"] = 0
                if detected_name != "unknown":
                    self._locked_cards[slot_idx] = detected_name
                    detection_info["card"] = detected_name
                    detection_info["count"] = 1
                    return {
                        "name": detected_name,
                        "score": result["score"],
                        "method": result["method"] + "_forced",
                    }

            return {
                "name": "waiting_for_card",
                "score": 0.0,
                "method": f"waiting_{waiting_state['frames']}/{self.MAX_WAITING_FRAMES}",
            }

        # Not in waiting state - normal detection
        if locked_card != "unknown" and not self._is_waiting_for_card(locked_card):
            if detected_name == locked_card:
                detection_info["count"] = min(
                    detection_info["count"] + 1, self.CONFIDENCE_THRESHOLD + 5
                )
                return {
                    "name": locked_card,
                    "score": result["score"],
                    "method": result["method"] + "_locked",
                }

            elif (
                detected_name != "unknown"
                and detected_name != locked_card
                and not self._is_waiting_for_card(detected_name)
            ):
                if detection_info["card"] == detected_name:
                    detection_info["count"] += 1
                else:
                    detection_info["card"] = detected_name
                    detection_info["count"] = 1

                if detection_info["count"] >= self.CONFIDENCE_THRESHOLD:
                    waiting_state["active"] = True
                    waiting_state["frames"] = 0
                    self._locked_cards[slot_idx] = "unknown"
                    return {
                        "name": "waiting_for_card",
                        "score": 0.0,
                        "method": "inferred_waiting",
                    }

                return {
                    "name": locked_card,
                    "score": 0.0,
                    "method": "persisted_over_" + detected_name.split("/")[-1],
                }

            else:
                return {"name": locked_card, "score": 0.0, "method": "persisted"}

        else:
            if detected_name != "unknown" and not self._is_waiting_for_card(
                detected_name
            ):
                if detection_info["card"] == detected_name:
                    detection_info["count"] += 1
                else:
                    detection_info["card"] = detected_name
                    detection_info["count"] = 1

                if detection_info["count"] >= self.CONFIDENCE_THRESHOLD:
                    self._locked_cards[slot_idx] = detected_name
                    return {
                        "name": detected_name,
                        "score": result["score"],
                        "method": result["method"] + "_confirmed",
                    }
                else:
                    return {
                        "name": detected_name,
                        "score": result["score"],
                        "method": f"building_{detection_info['count']}/{self.CONFIDENCE_THRESHOLD}",
                    }

            return result

    def _draw_debug_visualization(
        self,
        debug_frame: np.ndarray,
        frame: np.ndarray,
        slot_idx: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        result: dict,
    ):
        """Draw debug visualization for a card slot"""
        card_region = frame[y1:y2, x1:x2]
        card_height = card_region.shape[0]
        card_width = card_region.shape[1]

        crop_top = int(card_height * self.CROP_TOP)
        crop_bottom = int(card_height * (1 - self.CROP_BOTTOM))
        crop_left = int(card_width * self.CROP_SIDE)
        crop_right = int(card_width * (1 - self.CROP_SIDE))

        # Draw FULL card slot (yellow for color, gray for grayed out)
        is_gray = result.get("is_gray", False)
        slot_color = (128, 128, 128) if is_gray else (0, 255, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), slot_color, 2)

        # Draw lifted card slot (cyan)
        lifted_y1 = max(0, y1 - self.CARD_LIFT_OFFSET)
        lifted_y2 = max(0, y2 - self.CARD_LIFT_OFFSET)
        cv2.rectangle(debug_frame, (x1, lifted_y1), (x2, lifted_y2), (255, 255, 0), 1)

        # Draw TIGHT CROP region (green)
        tight_x1 = x1 + crop_left
        tight_y1 = y1 + crop_top
        tight_x2 = x1 + crop_right
        tight_y2 = y1 + crop_bottom
        cv2.rectangle(
            debug_frame,
            (tight_x1, tight_y1),
            (tight_x2, tight_y2),
            (0, 255, 0),
            2,
        )

        # Color based on method
        method = result.get("method", "")
        if "persisted" in method:
            color = (255, 165, 0)
        elif "waiting" in method or "confirming" in method or "inferred" in method:
            color = (255, 0, 255)
        elif "locked" in method or "confirmed" in method:
            color = (0, 255, 0)
        elif "building" in method:
            color = (0, 255, 255)
        elif result["name"] != "unknown":
            if result["score"] >= 0.70:
                color = (0, 255, 0)
            elif result["score"] >= 0.60:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        # Get short card name (remove folder prefix)
        card_name = result["name"]
        if "/" in card_name:
            card_name = card_name.split("/")[-1]

        # Draw background for better readability
        text = f"{card_name}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(
            debug_frame,
            (x1, y1 - 18),
            (x1 + text_size[0] + 4, y1 - 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            debug_frame,
            text,
            (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

        # Show score, method, and gray status below card
        gray_indicator = "🔘" if is_gray else "🟢"
        info_text = f"{result['score']:.2f} {method}"
        text_size2 = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.rectangle(
            debug_frame,
            (x1, y2 + 2),
            (x1 + text_size2[0] + 4, y2 + 16),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            debug_frame,
            info_text,
            (x1 + 2, y2 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
        )

    def detect_cards_in_hand(self, frame: np.ndarray) -> List[str]:
        """Detect cards - returns list of card names"""
        results = self.detect_cards_with_debug(frame, debug_frame=None)
        return [r["name"] for r in results]

    def reset_persistence(self):
        """Reset card persistence - call when starting a new match"""
        self._locked_cards = ["unknown", "unknown", "unknown", "unknown"]
        self._detection_counts = [
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
        ]
        self._waiting_state = [
            {"active": False, "frames": 0},
            {"active": False, "frames": 0},
            {"active": False, "frames": 0},
            {"active": False, "frames": 0},
        ]

    def get_current_elixir(self) -> Optional[float]:
        """Get the current elixir value from last frame"""
        return self._current_elixir

    # ensure alias at module level
    CardDetector = CardDetectorSimple
