# detection/card_detector_simple.py
"""Simple card detector with GPU acceleration and deck inference optimization"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from config.game_config import CARD_SLOTS
from detection.ocr_reader import OCRReader

# Try to import PyTorch for GPU acceleration
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        TORCH_DEVICE = torch.device("cuda")
        print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    else:
        TORCH_DEVICE = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None
    print("⚠️  PyTorch not installed, using CPU-only matching")


class CardDetectorSimple:
    """Detect cards in hand using simple color template matching with GPU acceleration"""

    # CROP SETTINGS - Adjust these to change what area is matched
    CROP_TOP = 0.17  # Remove top border
    CROP_BOTTOM = 0.28  # Remove elixir + bottom border
    CROP_SIDE = 0.12  # Remove side borders

    # Card selection lift offset (when player taps a card)
    CARD_LIFT_OFFSET = 21

    # Persistence settings
    CONFIDENCE_THRESHOLD = 1  # Frames before we lock in a card (Was 5, changed to 1 for faster locking since we rotate cards fast)
    MAX_WAITING_FRAMES = 40  # Maximum frames waiting_for_card can appear

    # Special card names (can be with or without folder prefix)
    WAITING_CARD_NAMES = [
        "waiting_for_card_1",
        "waiting_for_card_2",
        "waiting_for_card_3",
        "waiting_for_card_4",
        "other/waiting_for_card_1",
        "other/waiting_for_card_2",
        "other/waiting_for_card_3",
        "other/waiting_for_card_4",
    ]

    # Speed optimization settings
    HISTOGRAM_SHORTLIST_SIZE = 10  # Number of templates to check after histogram filter
    EARLY_ACCEPT_THRESHOLD = 0.85  # If score exceeds this, accept immediately

    # OCR is now rarely needed due to card cost lookup table
    # Only used as fallback when card name can't be matched to known costs

    def __init__(
        self, templates_dir="data/card_templates", deck: Optional[List[str]] = None
    ):
        """
        Args:
            templates_dir: Path to card templates directory
            deck: Optional list of 8 card names to filter detections
                  Should be base card names WITHOUT "ally_" or "enemy_" prefix
                  (e.g., ["knight", "archer", "musketeer", ...])
        """
        self.templates_dir = Path(templates_dir)

        # Normalize deck names: lowercase, strip any "ally_"/"enemy_" prefix
        if deck:
            normalized_deck = []
            for card in deck:
                card_lower = card.lower()
                # Remove team prefixes if user accidentally included them
                card_lower = card_lower.replace("ally_", "").replace("enemy_", "")
                normalized_deck.append(card_lower)
            self.deck = normalized_deck
        else:
            self.deck = None

        print(
            f"🔍 Initializing CardDetectorSimple with templates from: {self.templates_dir}"
        )
        print(f"   Full path: {self.templates_dir.resolve()}")

        if self.deck:
            print(f"   🎴 Deck filter active: {len(self.deck)} cards")

        self.templates = {}  # Color templates
        self.templates_gray = {}  # Grayscale templates
        self._load_templates()

        # Initialize OCR reader
        self.ocr_reader = OCRReader()

        # Load card costs from API data (eliminates need for OCR in most cases)
        self._card_costs = self._load_card_costs()

        # Card persistence - remember locked cards per slot
        self._locked_cards = ["unknown", "unknown", "unknown", "unknown"]
        self._card_confidence = [0, 0, 0, 0]  # Match confidence per slot

        # Track consecutive detections per slot
        self._detection_counts = [
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
            {"card": "unknown", "count": 0},
        ]

        # Track waiting_for_card state per slot
        # play_recorded: True if we've already added this play to the rotation queue
        self._waiting_state = [
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
        ]

        # === SPEED OPTIMIZATION STATE ===
        # Template resize cache
        self._template_resize_cache: Dict[tuple, np.ndarray] = {}

        # Histogram-based fast filtering
        self._template_histograms: Dict[str, np.ndarray] = {}
        self._precompute_template_histograms()

        # GPU tensors (if available)
        self._gpu_templates: Dict[str, torch.Tensor] = {}
        if TORCH_AVAILABLE:
            self._prepare_gpu_templates()

        # Deck inference (speed optimization)
        self._seen_card_names: List[str] = []  # ordered seen non-waiting names (unique)
        self._seen_card_set: set = set()
        self.active_deck: Optional[List[str]] = (
            None  # list of 8 base names when inferred
        )
        self.deck_confirmed: bool = False
        self.allowed_template_names: Optional[List[str]] = None

        # === CARD ROTATION QUEUE ===
        # In Clash Royale: 8 cards total, 4 in hand, 4 in queue
        # After you play a card, it goes to position 4 of the queue.
        # The card at position 1 of the queue comes to your hand.
        #
        # To predict the NEXT card, we need to know what card is currently
        # at position 1 of the queue. Since we start with unknown queue,
        # we need to track 5 card plays to know position 1:
        #
        # Initial: Hand=[A,B,C,D], Queue=[?,?,?,?] (unknown)
        # Play A: Queue=[?,?,?,A], next=? (still unknown)
        # Play B: Queue=[?,?,A,B], next=?
        # Play C: Queue=[?,A,B,C], next=?
        # Play D: Queue=[A,B,C,D], next=? (A went to hand, but we didn't know queue[1])
        # Play E: Queue=[B,C,D,E], next=B (NOW we know! B is coming next)
        #
        # So we need 5 plays before we can predict.
        #
        self._card_queue: List[str] = []  # Last 5 cards played (max length 5)
        self._hand_cards: List[str] = ["unknown"] * 4  # Current 4 cards in hand
        self._rotation_ready: bool = (
            False  # True when queue has 5 cards (we know next card)
        )
        self._last_played_slot: Optional[int] = (
            None  # Track which slot just had waiting_for_card
        )

        # Elixir tracking (fast bar detection, no OCR needed)
        self._current_elixir: Optional[float] = None

        # Per-slot elixir cost cache: only read OCR once per card, reset on waiting_for_card
        # Format: {slot_idx: {"cost": int, "needs_read": bool, "pending": bool, "wait_frames": int}}
        self._slot_elixir_cache = [
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
        ]

        # Gray state is derived from: current_elixir < card_cost
        self._cached_gray_states = [False, False, False, False]

    def _load_card_costs(self) -> Dict[str, int]:
        """Load card elixir costs from the API JSON file"""
        costs = {}
        api_path = Path("data/decks/cards_api.json")

        if not api_path.exists():
            print("   ⚠️  Card costs file not found, will use OCR fallback")
            return costs

        try:
            with open(api_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Process main cards
            for item in data.get("items", []):
                name = (
                    item.get("name", "")
                    .lower()
                    .replace(" ", "_")
                    .replace(".", "")
                    .replace("-", "_")
                )
                cost = item.get("elixirCost")
                if name and cost is not None:
                    # Store with various possible prefixes
                    costs[name] = cost
                    costs[f"base/{name}"] = cost
                    costs[f"evolution/{name}"] = cost
                    costs[f"evolution/{name}_evo"] = cost

            # Process support items (heroes)
            for item in data.get("supportItems", []):
                name = (
                    item.get("name", "")
                    .lower()
                    .replace(" ", "_")
                    .replace(".", "")
                    .replace("-", "_")
                )
                cost = item.get("elixirCost")
                if name and cost is not None:
                    costs[name] = cost
                    costs[f"hero/{name}"] = cost

            print(f"   ✅ Loaded {len(data.get('items', []))} card costs from API")

        except Exception as e:
            print(f"   ⚠️  Failed to load card costs: {e}")

        return costs

    def _get_card_cost(self, card_name: str) -> Optional[int]:
        """Get elixir cost for a card by name, returns None if unknown"""
        if not card_name or card_name in ("unknown", "waiting_for_card"):
            return None

        # Try exact match first
        if card_name in self._card_costs:
            return self._card_costs[card_name]

        # Try without prefix
        base_name = card_name.split("/")[-1] if "/" in card_name else card_name
        if base_name in self._card_costs:
            return self._card_costs[base_name]

        # Try normalizing the name
        normalized = (
            base_name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
        )
        if normalized in self._card_costs:
            return self._card_costs[normalized]

        return None

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

            # Filter by deck if specified
            if self.deck and not self._is_card_in_deck(card_name):
                continue

            template = cv2.imread(str(template_path))
            if template is not None:
                self.templates[card_name] = template
                self.templates_gray[card_name] = cv2.cvtColor(
                    template, cv2.COLOR_BGR2GRAY
                )

        # Create slot-specific waiting_for_card aliases
        # Templates are named waiting_for_card_1, waiting_for_card_2, etc.
        for slot_num in range(1, 5):
            slot_template_name = f"waiting_for_card_{slot_num}"
            # Check both with and without 'other/' prefix
            for prefix in ["", "other/"]:
                full_name = f"{prefix}{slot_template_name}"
                if (
                    full_name in self.templates
                    and slot_template_name not in self.templates
                ):
                    self.templates[slot_template_name] = self.templates[full_name]
                    self.templates_gray[slot_template_name] = self.templates_gray[
                        full_name
                    ]
                    print(f"   ✓ Aliased '{full_name}' as '{slot_template_name}'")
                    break

        print(f"   ✅ Loaded {len(self.templates)} templates")

        if self.templates:
            print(f"\n📊 Template categories:")
            categories = {}
            for card_name in self.templates.keys():
                category = card_name.split("/")[0] if "/" in card_name else "root"
                categories[category] = categories.get(category, 0) + 1
            for category, count in sorted(categories.items()):
                print(f"   • {category}: {count} cards")

    def _is_card_in_deck(self, card_name: str) -> bool:
        """Check if a card template should be loaded based on deck filter"""
        if not self.deck:
            return True

        # Always include waiting_for_card templates
        if "waiting_for_card" in card_name.lower():
            return True

        # Get base card name without folder prefix
        base_name = card_name.split("/")[-1].lower()

        # Remove evolution suffix for matching
        base_name_no_evo = base_name.replace("_evolution", "").replace("_evo", "")

        # Check if this card is in the deck
        for deck_card in self.deck:
            deck_card_lower = deck_card.lower()
            # Match exact, with evolution, or partial
            if (
                deck_card_lower == base_name
                or deck_card_lower == base_name_no_evo
                or deck_card_lower in base_name
                or base_name_no_evo in deck_card_lower
            ):
                return True

        return False

    def _precompute_template_histograms(self):
        """Compute color histograms for fast template pre-filtering"""
        print("   📊 Precomputing template histograms...")
        for name, tpl in self.templates.items():
            try:
                # Resize to small size for fast histogram
                small = cv2.resize(tpl, (32, 32), interpolation=cv2.INTER_AREA)
                # Compute simplified color histogram (fewer bins = faster)
                hist = cv2.calcHist(
                    [small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]
                )
                cv2.normalize(hist, hist)
                self._template_histograms[name] = hist.flatten().astype(np.float32)
            except Exception:
                continue
        print(f"   ✅ Histograms ready for {len(self._template_histograms)} templates")

    def _prepare_gpu_templates(self):
        """Prepare templates as GPU tensors for fast matching"""
        if not TORCH_AVAILABLE:
            return
        print("   🚀 Preparing GPU templates...")
        # We'll prepare these on-demand when we know the target size
        self._gpu_templates_prepared = False

    def _get_shortlist_by_histogram(
        self, card_artwork: np.ndarray, top_k: int = None
    ) -> List[str]:
        """Get top-k most similar templates by histogram comparison (very fast)"""
        if top_k is None:
            top_k = self.HISTOGRAM_SHORTLIST_SIZE

        if not self._template_histograms:
            return list(self.templates.keys())

        try:
            # Compute query histogram
            small = cv2.resize(card_artwork, (32, 32), interpolation=cv2.INTER_AREA)
            query_hist = cv2.calcHist(
                [small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]
            )
            cv2.normalize(query_hist, query_hist)
            query_flat = query_hist.flatten().astype(np.float32)
        except Exception:
            return list(self.templates.keys())

        # Get candidate templates (restricted if deck confirmed)
        if self.deck_confirmed and self.allowed_template_names:
            candidates = self.allowed_template_names
        else:
            # During initial detection (before deck is confirmed) we should
            # avoid matching against evolution variants. These only appear
            # after at least one evolution cycle and including them early
            # causes false positives (e.g. `electro_dragon_evo`). Limit the
            # candidate set to `base/` and `hero/` templates plus any
            # waiting_for_card templates and root templates.
            candidates = []
            for name in self.templates.keys():
                # Always allow waiting templates (they can be in other/)
                if name in self.WAITING_CARD_NAMES or "waiting_for_card" in name:
                    candidates.append(name)
                    continue

                # If template has no folder prefix, allow it
                if "/" not in name:
                    candidates.append(name)
                    continue

                prefix = name.split("/")[0].lower()
                if prefix in ("base", "hero"):
                    candidates.append(name)

        # Score by histogram similarity (dot product of normalized histograms)
        scores = []
        for name in candidates:
            tpl_hist = self._template_histograms.get(name)
            if tpl_hist is not None:
                score = np.dot(query_flat, tpl_hist)
                scores.append((score, name))

        if not scores:
            return candidates

        # Sort by score descending and take top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [name for _, name in scores[:top_k]]

    def _cache_resize_template(
        self, name: str, target_w: int, target_h: int, use_gray: bool = False
    ) -> Optional[np.ndarray]:
        """Get cached resized template or create and cache it"""
        key = (name, target_w, target_h, use_gray)
        if key in self._template_resize_cache:
            return self._template_resize_cache[key]

        source = self.templates_gray if use_gray else self.templates
        tpl = source.get(name)
        if tpl is None:
            return None

        try:
            resized = cv2.resize(
                tpl, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            # Limit cache size
            if len(self._template_resize_cache) > 500:
                # Clear half the cache
                keys_to_remove = list(self._template_resize_cache.keys())[:250]
                for k in keys_to_remove:
                    del self._template_resize_cache[k]
            self._template_resize_cache[key] = resized
            return resized
        except Exception:
            return None

    def _match_template_gpu(
        self, card_artwork: np.ndarray, template: np.ndarray
    ) -> float:
        """GPU-accelerated template matching using PyTorch"""
        if not TORCH_AVAILABLE:
            return self._match_template_cpu(card_artwork, template)

        try:
            # Ensure same size
            if card_artwork.shape != template.shape:
                template = cv2.resize(
                    template, (card_artwork.shape[1], card_artwork.shape[0])
                )

            # Convert to tensors
            card_tensor = torch.from_numpy(card_artwork.astype(np.float32)).to(
                TORCH_DEVICE
            )
            tpl_tensor = torch.from_numpy(template.astype(np.float32)).to(TORCH_DEVICE)

            # Normalize
            card_norm = card_tensor - card_tensor.mean()
            tpl_norm = tpl_tensor - tpl_tensor.mean()

            # Compute normalized cross-correlation
            numerator = (card_norm * tpl_norm).sum()
            denominator = torch.sqrt((card_norm**2).sum() * (tpl_norm**2).sum())

            if denominator > 0:
                score = (numerator / denominator).item()
            else:
                score = 0.0

            return score

        except Exception:
            return self._match_template_cpu(card_artwork, template)

    def _match_template_cpu(
        self, card_artwork: np.ndarray, template: np.ndarray
    ) -> float:
        """CPU template matching"""
        try:
            if card_artwork.shape != template.shape:
                template = cv2.resize(
                    template, (card_artwork.shape[1], card_artwork.shape[0])
                )
            result = cv2.matchTemplate(card_artwork, template, cv2.TM_CCOEFF_NORMED)
            return float(np.max(result))
        except Exception:
            return 0.0

    def detect_cards_with_debug(
        self, frame: np.ndarray, debug_frame: np.ndarray = None
    ) -> List[dict]:
        """Detect cards with debug visualization"""
        detected_cards = []

        # Read current elixir using fast bar detection (every frame is fine, it's fast)
        self._current_elixir = self.ocr_reader.read_elixir(frame)

        for slot_idx, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
            h, w = frame.shape[:2]

            if x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                detected_cards.append(
                    {"name": "unknown", "score": 0.0, "method": "invalid"}
                )
                continue

            slot_cache = self._slot_elixir_cache[slot_idx]
            locked_card = self._locked_cards[slot_idx]
            waiting_state = self._waiting_state[slot_idx]

            # Decrement cooldown if active (prevents re-triggering right after confirmation)
            if waiting_state.get("cooldown", 0) > 0:
                waiting_state["cooldown"] -= 1

            # === FAST CHECK: Detect waiting_for_card even if locked ===
            # This is crucial - when a card is played, we need to detect
            # the "waiting_for_card" state to unlock the slot
            slot_roi = frame[y1:y2, x1:x2]
            is_waiting_visual = self._detect_waiting_for_card_fast(slot_roi, slot_idx)

            if is_waiting_visual and locked_card != "unknown":
                # Card was just played! Trigger waiting state
                # But only if not in cooldown (prevents false re-trigger after confirmation)
                if (
                    not waiting_state["active"]
                    and waiting_state.get("cooldown", 0) == 0
                ):
                    print(
                        f"   ⏳ Slot {slot_idx}: Detected waiting_for_card (was {locked_card})"
                    )
                    waiting_state["active"] = True
                    waiting_state["frame_count"] = 0
                    waiting_state["detected_card"] = locked_card
                    # Record wall-clock time so we can assign predicted
                    # cards after a short delay (e.g. 300ms) when deck is
                    # confirmed and rotation inference is active.
                    waiting_state["started_at"] = time.time()
                    waiting_state["predicted_assigned"] = False

                    # Only record the play ONCE
                    if not waiting_state.get("play_recorded", False):
                        self._on_card_played(slot_idx, locked_card)
                        waiting_state["play_recorded"] = True

                    # Reset lock for this slot
                    self._locked_cards[slot_idx] = "unknown"
                    self._card_confidence[slot_idx] = 0
                    self._slot_elixir_cache[slot_idx] = {
                        "cost": None,
                        "needs_read": False,
                        "pending": False,
                        "wait_frames": 0,
                    }

            # === FAST PATH: Skip detection for locked cards ===
            # If card is locked and not in waiting state, skip expensive detection
            if (
                locked_card != "unknown"
                and not self._is_waiting_for_card(locked_card)
                and not waiting_state["active"]
                and not is_waiting_visual
            ):

                # Use cached gray state based on elixir cost
                card_cost = slot_cache["cost"]
                if card_cost is not None and self._current_elixir is not None:
                    is_gray = self._current_elixir < card_cost
                else:
                    is_gray = self._cached_gray_states[slot_idx]

                self._cached_gray_states[slot_idx] = is_gray

                # Return cached result immediately (skip template matching!)
                result = {
                    "name": locked_card,
                    "score": 1.0,  # High confidence since locked
                    "method": "cached_locked",
                    "is_gray": is_gray,
                    "current_elixir": self._current_elixir,
                    "card_cost": card_cost,
                }
                detected_cards.append(result)

                # Still draw debug if needed
                if debug_frame is not None:
                    self._draw_debug_visualization(
                        debug_frame, frame, slot_idx, x1, y1, x2, y2, result
                    )
                continue

            # === SLOW PATH: Full detection needed ===
            # === PHASE 1: Determine gray state ===
            # Use cached cost if available, otherwise use fast saturation detection
            card_cost = slot_cache["cost"]
            if card_cost is not None and self._current_elixir is not None:
                is_gray = self._current_elixir < card_cost
            else:
                # Fallback to saturation-based detection (fast, ~0.05ms)
                is_gray = self.ocr_reader.detect_gray_card(frame, (x1, y1, x2, y2))

            self._cached_gray_states[slot_idx] = is_gray

            # If the deck is confirmed and rotation tracking is ready,
            # we can avoid expensive template matching entirely. Instead
            # we infer the slot contents from our rotation/queue state and
            # only keep looking for the `waiting_for_card` visual to know
            # when a card was played and a new one arrives.
            if self.deck_confirmed and self._rotation_ready:
                # If the slot visually shows waiting_for_card, report that
                if is_waiting_visual:
                    predicted = self._get_predicted_next_card(slot_idx)
                    result = {
                        "name": "waiting_for_card",
                        "score": 0.0,
                        "method": "waiting",
                        "predicted_next": predicted,
                    }
                else:
                    # Otherwise, infer the current card from our hand tracking
                    inferred = self._hand_cards[slot_idx]
                    result = {
                        "name": inferred if inferred is not None else "unknown",
                        "score": 1.0,
                        "method": "inferred_rotation",
                    }

                # Apply the same persistence/confirmation logic as usual
                result = self._apply_persistence_logic(slot_idx, result)

                # Append and draw debug as usual
                detected_cards.append(result)
                if debug_frame is not None:
                    self._draw_debug_visualization(
                        debug_frame, frame, slot_idx, x1, y1, x2, y2, result
                    )
                continue

            # === PHASE 2: Detect card ===
            # Try normal position first
            result = self._detect_card_at_position(
                frame, x1, y1, x2, y2, use_gray=is_gray
            )

            # If low score, try lifted position
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

            # === PHASE 3: Get elixir cost (lookup first, OCR only as last resort) ===
            detected_name = result["name"]

            # Try to get cost from card name lookup (instant, no OCR needed)
            if slot_cache["cost"] is None and detected_name not in (
                "unknown",
                "waiting_for_card",
            ):
                lookup_cost = self._get_card_cost(detected_name)
                if lookup_cost is not None:
                    slot_cache["cost"] = lookup_cost
                    slot_cache["needs_read"] = False
                    slot_cache["pending"] = False

            # OCR fallback: only if lookup failed and we have a confirmed card
            # This should rarely happen since we have costs for all known cards
            if (
                slot_cache["needs_read"]
                and slot_cache["cost"] is None
                and detected_name not in ("unknown", "waiting_for_card")
            ):

                # Card detected but cost not in lookup table - use OCR
                cost = self.ocr_reader.read_card_elixir_cost(frame, (x1, y1, x2, y2))
                if cost is not None:
                    slot_cache["cost"] = cost
                    slot_cache["needs_read"] = False
                    slot_cache["pending"] = False

            # Update deck inference
            self._update_deck_inference(result)

            # Add extra info
            result["is_gray"] = is_gray
            result["current_elixir"] = self._current_elixir
            result["card_cost"] = slot_cache["cost"]

            detected_cards.append(result)

            # Draw debug visualization
            if debug_frame is not None:
                self._draw_debug_visualization(
                    debug_frame, frame, slot_idx, x1, y1, x2, y2, result
                )

        # Try to initialize rotation tracking after processing all slots
        if not self._rotation_ready:
            self._try_initialize_rotation()

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

        # TIGHT CROP
        card_height, card_width = card_region.shape[:2]
        crop_top = int(card_height * self.CROP_TOP)
        crop_bottom = int(card_height * (1 - self.CROP_BOTTOM))
        crop_left = int(card_width * self.CROP_SIDE)
        crop_right = int(card_width * (1 - self.CROP_SIDE))

        card_artwork = card_region[crop_top:crop_bottom, crop_left:crop_right]
        if card_artwork.size == 0:
            return {"name": "unknown", "score": 0.0, "method": "empty"}

        # Convert to grayscale if needed
        if use_gray and len(card_artwork.shape) == 3:
            card_to_match = cv2.cvtColor(card_artwork, cv2.COLOR_BGR2GRAY)
        else:
            card_to_match = card_artwork

        # Find best match
        best_name, best_score = self._find_best_match(card_to_match, use_gray=use_gray)

        # SAFETY CHECK: If deck is confirmed, validate that matched card is in deck
        # This shouldn't happen if histogram shortlist is working correctly, but acts as failsafe
        # Skip check for waiting_for_card variants (they're not real cards)
        is_waiting = best_name == "unknown" or "waiting_for_card" in best_name
        if self.deck_confirmed and not is_waiting:
            base_name = best_name.split("/")[-1].replace("_evo", "")
            if self.active_deck and base_name not in self.active_deck:
                # Card not in deck - this is a detection error, return unknown
                print(f"   ⚠️ SAFETY: Rejected {best_name} (not in locked deck)")
                best_name = "unknown"
                best_score = 0.0

        method = "gray" if use_gray else "color"
        return {"name": best_name, "score": best_score, "method": method}

    def _find_best_match(
        self, card_artwork: np.ndarray, use_gray: bool = False
    ) -> Tuple[str, float]:
        """Find best matching template using histogram shortlist + template matching"""
        if len(self.templates) == 0:
            return "unknown", 0.0

        # Get shortlist by histogram (fast pre-filter)
        shortlist = self._get_shortlist_by_histogram(card_artwork)

        best_match = "unknown"
        best_score = 0.0
        target_h, target_w = card_artwork.shape[:2]

        for card_name in shortlist:
            template = self._cache_resize_template(
                card_name, target_w, target_h, use_gray=use_gray
            )
            if template is None:
                continue

            # Match using GPU if available
            if TORCH_AVAILABLE:
                score = self._match_template_gpu(card_artwork, template)
            else:
                score = self._match_template_cpu(card_artwork, template)

            # Early exit for very good matches
            if score > self.EARLY_ACCEPT_THRESHOLD:
                return card_name, score

            if score > best_score:
                best_score = score
                best_match = card_name

        return best_match, best_score

    def _update_deck_inference(self, result: dict):
        """Track observed cards to infer the 8-card deck for faster matching"""
        if self.deck_confirmed:
            return

        name = result.get("name", "unknown")
        score = result.get("score", 0.0)
        method = result.get("method", "")

        # Add card if it's locked/confirmed (regardless of score) OR has high confidence
        is_locked = "locked" in method or "confirmed" in method
        if name == "unknown" or self._is_waiting_for_card(name):
            return
        if not is_locked and score < 0.4:
            return

        # Get base name without folder prefix
        base_name = name.split("/")[-1]

        if base_name not in self._seen_card_set:
            self._seen_card_set.add(base_name)
            self._seen_card_names.append(base_name)
            print(
                f"   🎴 Deck card {len(self._seen_card_names)}/8: {base_name}, Confidence-score: {score:.2f}"
            )

            # Once we have 8 unique cards, confirm deck
            if len(self._seen_card_set) >= 8:
                self._confirm_deck()

    def _confirm_deck(self):
        """Confirm the deck and restrict future matching to only these cards"""
        self.active_deck = list(self._seen_card_names)[:8]

        # Build allowed template names (include evolution variants)
        allowed = set()
        for tpl_name in self.templates.keys():
            base = tpl_name.split("/")[-1]
            # Include if exact match
            if base in self._seen_card_set:
                allowed.add(tpl_name)
            # Include evolution variants
            elif "evolution" in tpl_name.lower():
                for seen in self._seen_card_set:
                    if seen in tpl_name:
                        allowed.add(tpl_name)
                        break

        # Always include waiting_for_card
        for name in self.WAITING_CARD_NAMES:
            if name in self.templates:
                allowed.add(name)

        self.allowed_template_names = list(allowed)
        self.deck_confirmed = True

        # Clear caches since we'll be using fewer templates
        self._template_resize_cache.clear()

        # Try to initialize rotation tracking
        self._try_initialize_rotation()

        print(
            f"\n✅ Deck confirmed! Restricting to {len(self.allowed_template_names)} templates:"
        )
        print(f"   Deck: {', '.join(self.active_deck)}")

    def _detect_waiting_for_card_fast(
        self, slot_roi: np.ndarray, slot_idx: int
    ) -> bool:
        """
        Detect 'waiting_for_card' state using template matching.
        Uses slot-specific templates (waiting_for_card_1, waiting_for_card_2, etc.)
        since each slot has slightly different lighting/appearance.

        Args:
            slot_roi: The image region of the card slot (full slot, will be cropped)
            slot_idx: The slot index (0-3)

        Returns True only if the slot matches the waiting_for_card template
        with high confidence.
        """
        try:
            if slot_roi.size == 0:
                return False

            # Crop slot_roi to artwork area (same as how templates are extracted)
            card_height, card_width = slot_roi.shape[:2]
            crop_top = int(card_height * self.CROP_TOP)
            crop_bottom = int(card_height * (1 - self.CROP_BOTTOM))
            crop_left = int(card_width * self.CROP_SIDE)
            crop_right = int(card_width * (1 - self.CROP_SIDE))
            card_artwork = slot_roi[crop_top:crop_bottom, crop_left:crop_right]

            if card_artwork.size == 0:
                return False

            # Get the slot-specific waiting_for_card template (1-indexed)
            slot_num = slot_idx + 1
            waiting_template = self.templates.get(f"waiting_for_card_{slot_num}")

            # Fallback: try generic waiting_for_card or any waiting template
            if waiting_template is None:
                waiting_template = self.templates.get("waiting_for_card")
            if waiting_template is None:
                for name in self.templates:
                    if "waiting_for_card" in name:
                        waiting_template = self.templates[name]
                        break

            if waiting_template is None:
                return False

            # Resize template to match artwork size (should be very close already)
            h, w = card_artwork.shape[:2]
            template_resized = cv2.resize(
                waiting_template, (w, h), interpolation=cv2.INTER_AREA
            )

            # Template matching on the cropped artwork area
            result = cv2.matchTemplate(
                card_artwork, template_resized, cv2.TM_CCOEFF_NORMED
            )
            score = result[0, 0] if result.size > 0 else 0.0

            # Only return True if we have a HIGH match (>0.7 means very confident)
            # This prevents false positives from lifted/dragged cards
            return score > 0.70

        except Exception:
            return False

    def _is_waiting_for_card(self, card_name: str) -> bool:
        """Check if the detected card is a waiting_for_card placeholder"""
        return card_name in self.WAITING_CARD_NAMES or card_name.endswith(
            "waiting_for_card"
        )

    # === CARD ROTATION QUEUE METHODS ===

    def _get_base_name(self, card_name: str) -> str:
        """Get base card name without folder prefix"""
        if not card_name or card_name in ("unknown", "waiting_for_card"):
            return card_name
        return card_name.split("/")[-1].replace("_evo", "")

    def _on_card_played(self, slot_idx: int, played_card: str):
        """
        Called when a card is played (waiting_for_card detected).

        Simple queue logic:
        1. The played card goes to the BACK of the queue
        2. The card at the FRONT of the queue comes to hand
        """
        base_name = self._get_base_name(played_card)
        if base_name in ("unknown", "waiting_for_card"):
            return

        if self._rotation_ready and len(self._card_queue) >= 1:
            # Queue is active - the front card will come to hand
            next_card = self._card_queue.pop(0)  # Remove from front
            self._card_queue.append(base_name)  # Add played card to back

            # Update hand tracking - the played card leaves, next_card arrives
            self._hand_cards[slot_idx] = next_card

            queue_display = [f"[{i+1}]{c}" for i, c in enumerate(self._card_queue)]
            print(
                f"   🔄 Played {base_name} → {next_card} coming → Queue: {' → '.join(queue_display)}"
            )
        else:
            # Queue not ready yet, just track plays
            self._card_queue.append(base_name)
            if len(self._card_queue) > 4:
                self._card_queue.pop(0)
            queue_display = [f"[{i+1}]{c}" for i, c in enumerate(self._card_queue)]
            print(f"   🔄 Played {base_name} → Queue: {' → '.join(queue_display)}")
            print("Queue list: " + str(self._card_queue))

    def _on_card_confirmed(self, slot_idx: int, card_name: str):
        """Called when a new card appears in a slot (confirmed detection)"""
        base_name = self._get_base_name(card_name)
        if base_name in ("unknown", "waiting_for_card") or self._is_waiting_for_card(
            card_name
        ):
            return

        # SAFETY CHECK: If deck is confirmed, validate card is in deck
        if self.deck_confirmed and self.active_deck:
            if base_name not in self.active_deck:
                print(f"   ⚠️ SAFETY: Rejected confirmed card {base_name} (not in deck)")
                return

    def _get_predicted_next_card(self, slot_idx: int) -> Optional[str]:
        """Get the predicted next card for a slot based on rotation queue"""
        if not self._rotation_ready or len(self._card_queue) < 1:
            return None

        # Front of queue is the next card that will arrive
        return self._card_queue[0]

    def _try_initialize_rotation(self):
        """
        Initialize rotation tracking when the deck is confirmed.

        Simply find the 4 cards NOT in hand - those are the queue.
        """
        if self._rotation_ready:
            return

        if not self.deck_confirmed or not self.active_deck:
            return

        # Get current hand as base names
        hand_base = set()
        for card in self._hand_cards:
            if card and card != "unknown":
                base = card.split("/")[-1].replace("_evo", "")
                hand_base.add(base)

        if len(hand_base) < 4:
            return

        # Queue = cards from deck that are NOT in hand
        # We don't know the exact order, but we'll figure it out as cards are played
        queue = []
        for card in self.active_deck:
            base = card.replace("_evo", "")
            if base not in hand_base:
                queue.append(base)

        if len(queue) == 4:
            self._card_queue = queue
            self._rotation_ready = True
            print(f"   🔁 Rotation initialized!")
            print(f"      Hand: {list(hand_base)}")
            print(f"      Queue: {self._card_queue}")

    def _apply_persistence_logic(self, slot_idx: int, result: dict) -> dict:
        """Apply persistence logic with transition validation"""
        detected_name = result["name"]
        locked_card = self._locked_cards[slot_idx]
        detection_info = self._detection_counts[slot_idx]
        waiting_state = self._waiting_state[slot_idx]

        # Handle waiting_for_card detection (card was played)
        if self._is_waiting_for_card(detected_name):
            # Note: _on_card_played is called in detect_cards_with_debug when waiting is first detected
            # Don't call it again here to avoid duplicates

            waiting_state["active"] = True
            waiting_state["frames"] = 0
            self._locked_cards[slot_idx] = "unknown"
            self._hand_cards[slot_idx] = "unknown"  # Mark slot as empty
            detection_info["card"] = "waiting_for_card"
            detection_info["count"] = 1

            # Reset elixir cost cache for this slot - new card incoming
            # But if we can predict the next card, use it immediately!
            predicted_card = self._get_predicted_next_card(slot_idx)
            if predicted_card:
                # We know what card is coming - use prediction!
                predicted_cost = self._get_card_cost(predicted_card)
                self._slot_elixir_cache[slot_idx] = {
                    "cost": predicted_cost,
                    "needs_read": False,  # No OCR needed!
                    "pending": False,
                    "wait_frames": 0,
                }
            else:
                # No prediction available, will need to detect/OCR
                self._slot_elixir_cache[slot_idx] = {
                    "cost": None,
                    "needs_read": True,
                    "pending": False,
                    "wait_frames": 0,
                }

            return {
                "name": "waiting_for_card",
                "score": result["score"],
                "method": "waiting",
                "predicted_next": predicted_card,  # Include prediction in result
            }

        # If in waiting state, track frames
        if waiting_state["active"]:
            # If deck is confirmed and rotation is ready, assign the predicted
            # next card to this slot after a short delay (300ms) so we don't
            # need to run template matching to confirm the new card.
            if (
                waiting_state.get("started_at")
                and self.deck_confirmed
                and self._rotation_ready
                and not waiting_state.get("predicted_assigned", False)
            ):
                if time.time() - waiting_state["started_at"] >= 0.3:
                    predicted = self._get_predicted_next_card(slot_idx)
                    if predicted:
                        # Assign predicted card into slot
                        self._locked_cards[slot_idx] = predicted
                        self._hand_cards[slot_idx] = predicted
                        waiting_state["predicted_assigned"] = True
                        waiting_state["active"] = False
                        waiting_state["frames"] = 0
                        waiting_state["play_recorded"] = False
                        waiting_state["cooldown"] = 5
                        # Notify confirmation hooks
                        self._on_card_confirmed(slot_idx, predicted)
                        return {
                            "name": predicted,
                            "score": 1.0,
                            "method": "inferred_rotation_assigned",
                        }

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
                    self._hand_cards[slot_idx] = detected_name  # Update hand tracking
                    self._on_card_confirmed(slot_idx, detected_name)
                    waiting_state["active"] = False
                    waiting_state["frames"] = 0
                    waiting_state["play_recorded"] = False  # Reset for next play
                    waiting_state["cooldown"] = 5  # Prevent re-trigger for 5 frames
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
                waiting_state["play_recorded"] = False  # Reset for next play
                waiting_state["cooldown"] = 5  # Prevent re-trigger for 5 frames
                if detected_name != "unknown":
                    self._locked_cards[slot_idx] = detected_name
                    self._hand_cards[slot_idx] = detected_name  # Update hand tracking
                    self._on_card_confirmed(slot_idx, detected_name)
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

        # Not in waiting state
        if locked_card != "unknown" and not self._is_waiting_for_card(locked_card):
            if detected_name == locked_card:
                detection_info["count"] = 1
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
                # Immediately lock in the new card
                self._locked_cards[slot_idx] = detected_name
                self._hand_cards[slot_idx] = detected_name
                self._on_card_confirmed(slot_idx, detected_name)
                detection_info["card"] = detected_name
                detection_info["count"] = 1
                return {
                    "name": detected_name,
                    "score": result["score"],
                    "method": result["method"] + "_confirmed_immediate",
                }
            else:
                return {"name": locked_card, "score": 0.0, "method": "persisted"}
        else:
            # No locked card yet - initial detection
            if detected_name != "unknown" and not self._is_waiting_for_card(
                detected_name
            ):
                # Immediately lock in the detected card
                self._locked_cards[slot_idx] = detected_name
                self._hand_cards[slot_idx] = detected_name  # Track initial hand
                self._on_card_confirmed(slot_idx, detected_name)
                detection_info["card"] = detected_name
                detection_info["count"] = 1
                return {
                    "name": detected_name,
                    "score": result["score"],
                    "method": result["method"] + "_confirmed_immediate",
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
        card_height, card_width = card_region.shape[:2]

        crop_top = int(card_height * self.CROP_TOP)
        crop_bottom = int(card_height * (1 - self.CROP_BOTTOM))
        crop_left = int(card_width * self.CROP_SIDE)
        crop_right = int(card_width * (1 - self.CROP_SIDE))

        # Draw slot rectangle
        is_gray = result.get("is_gray", False)
        slot_color = (128, 128, 128) if is_gray else (0, 255, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), slot_color, 2)

        # Draw lifted slot
        lifted_y1 = max(0, y1 - self.CARD_LIFT_OFFSET)
        lifted_y2 = max(0, y2 - self.CARD_LIFT_OFFSET)
        cv2.rectangle(debug_frame, (x1, lifted_y1), (x2, lifted_y2), (255, 255, 0), 1)

        # Draw crop region
        tight_x1 = x1 + crop_left
        tight_y1 = y1 + crop_top
        tight_x2 = x1 + crop_right
        tight_y2 = y1 + crop_bottom
        cv2.rectangle(
            debug_frame, (tight_x1, tight_y1), (tight_x2, tight_y2), (0, 255, 0), 2
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
            color = (
                (0, 255, 0)
                if result["score"] >= 0.70
                else (0, 255, 255) if result["score"] >= 0.60 else (0, 165, 255)
            )
        else:
            color = (0, 0, 255)

        # Card name
        card_name = (
            result["name"].split("/")[-1] if "/" in result["name"] else result["name"]
        )
        text_size = cv2.getTextSize(card_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(
            debug_frame, (x1, y1 - 18), (x1 + text_size[0] + 4, y1 - 2), (0, 0, 0), -1
        )
        cv2.putText(
            debug_frame,
            card_name,
            (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

        # Elixir cost (show below card name)
        card_cost = result.get("card_cost", None)
        if card_cost is not None:
            cost_text = f"{card_cost} elixir"
            cost_size = cv2.getTextSize(cost_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.rectangle(
                debug_frame,
                (x1, y2 + 18),
                (x1 + cost_size[0] + 4, y2 + 32),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                debug_frame,
                cost_text,
                (x1 + 2, y2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 120, 255),  # Light purple
                1,
            )

        # Score and method
        info_text = f"{result['score']:.2f} {method}"
        text_size2 = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.rectangle(
            debug_frame, (x1, y2 + 2), (x1 + text_size2[0] + 4, y2 + 16), (0, 0, 0), -1
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
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
            {"active": False, "frames": 0, "play_recorded": False, "cooldown": 0},
        ]
        # Reset elixir cost cache
        self._slot_elixir_cache = [
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
            {"cost": None, "needs_read": False, "pending": False, "wait_frames": 0},
        ]
        self._cached_gray_states = [False, False, False, False]
        # Also reset deck inference for new match
        self._seen_card_names = []
        self._seen_card_set = set()
        self.active_deck = None
        self.deck_confirmed = False
        # Reset rotation tracking
        self._card_queue = []  # Sliding window of last 4 cards played
        self._hand_cards = ["unknown"] * 4
        self._rotation_ready = False
        self.allowed_template_names = None
        self._template_resize_cache.clear()

    def get_current_elixir(self) -> Optional[float]:
        """Get the current elixir value from last frame"""
        return self._current_elixir


# Backwards-compatible alias
CardDetector = CardDetectorSimple
