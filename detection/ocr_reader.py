"""OCR for elixir, timer, tower HP - optimized with GPU-accelerated EasyOCR"""

import cv2
import numpy as np
import re

# Try to import EasyOCR for GPU acceleration, fall back to Tesseract
try:
    import easyocr
    import torch

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OCRReader:
    def __init__(self, use_gpu=True):
        # Initialize OCR engine
        self._easyocr_reader = None
        self._use_easyocr = False

        if EASYOCR_AVAILABLE and use_gpu:
            try:
                gpu_available = torch.cuda.is_available()
                print(f"🔧 Initializing EasyOCR (GPU: {gpu_available})...")
                self._easyocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=gpu_available,
                    verbose=False,
                    model_storage_directory="data/models/easyocr",
                )
                self._use_easyocr = True
                if gpu_available:
                    print(
                        f"   ✅ EasyOCR ready with GPU: {torch.cuda.get_device_name(0)}"
                    )
                else:
                    print("   ✅ EasyOCR ready (CPU mode)")
            except Exception as e:
                print(f"   ⚠️ EasyOCR init failed: {e}, falling back to Tesseract")
                self._use_easyocr = False

        if not self._use_easyocr:
            if TESSERACT_AVAILABLE:
                print("🔧 Using Tesseract OCR (CPU)")
            else:
                print("❌ No OCR engine available! Install easyocr or pytesseract")

        # Elixir bar region (the purple/pink bar itself, not the number)
        self.ELIXIR_BAR_REGION = (192, 1264, 691, 1235)
        self.ELIXIR_BAR_EMPTY_X = 192
        self.ELIXIR_BAR_FULL_X = 691

        # Purple/pink color range in HSV for detecting elixir bar fill
        self.ELIXIR_COLOR_LOW = np.array([140, 80, 100])
        self.ELIXIR_COLOR_HIGH = np.array([170, 255, 255])

        # Card elixir cost - tight centered region at bottom of card
        self.CARD_ELIXIR_WIDTH = 30
        self.CARD_ELIXIR_HEIGHT = 30
        self.CARD_ELIXIR_Y_OFFSET = -1

    def read_elixir(self, frame: np.ndarray) -> float:
        """
        Read current elixir by measuring purple bar + recharge progress.

        Bar structure at Y=1245:
        - Bar starts at x=195, each elixir = 50 pixels
        - Three color zones (in BGR/HSV):
          1. Purple (filled) - bright magenta
          2. Recharge (partial) - higher brightness, LOWER saturation than empty
          3. Empty - dark blue, HIGH saturation, LOW brightness

        Photoshop values (H:0-360, S:0-100, V:0-100):
        - Empty:    H=215, S=97%, V=47%, RGB(4,53,120)
        - Recharge: H=221, S=67%, V=64%, RGB(54,88,162)

        OpenCV HSV (H:0-179, S:0-255, V:0-255):
        - Empty:    H=107, S=247, V=120
        - Recharge: H=110, S=171, V=163

        Returns float 0.0-10.0 with decimal precision from recharge bar.
        """
        try:
            # Single Y coordinate to sample (middle of bar)
            BAR_Y = 1245

            # Bar dimensions
            BAR_START_X = 195
            BAR_END_X = 695
            PIXELS_PER_ELIXIR = 50

            h, w = frame.shape[:2]

            # Bounds check
            if BAR_END_X > w or BAR_Y >= h:
                return None

            # Get single row of pixels at BAR_Y
            row_bgr = frame[BAR_Y, BAR_START_X:BAR_END_X]

            if row_bgr.size == 0:
                return None

            # Convert to HSV
            row_reshaped = row_bgr.reshape(1, -1, 3)
            hsv_row = cv2.cvtColor(row_reshaped, cv2.COLOR_BGR2HSV)[0]

            # Extract channels
            hue = hsv_row[:, 0].astype(np.int16)
            sat = hsv_row[:, 1].astype(np.int16)
            val = hsv_row[:, 2].astype(np.int16)

            # Also get blue channel directly (BGR)
            blue_channel = row_bgr[:, 0].astype(np.int16)

            # === COLOR DEFINITIONS (OpenCV scale) ===
            # Purple (filled elixir): magenta/pink hue, high saturation
            # Hue around 140-170 in OpenCV (280-340 in Photoshop)
            purple_mask = (
                (hue >= 130)
                & (hue <= 175)  # Purple/magenta hue
                & (sat >= 100)  # Good saturation
                & (val >= 100)  # Visible brightness
            )

            # Empty: H=107 (OpenCV), S=247, V=120, Blue=120
            # Recharge: H=110 (OpenCV), S=171, V=163, Blue=162
            #
            # Key insight: Recharge has LOWER saturation and HIGHER value than empty!
            # Also recharge has higher blue channel value

            # Empty detection: high saturation (>200), lower value (<140), lower blue (<140)
            empty_mask = (
                (sat >= 200)  # High saturation (empty is very saturated)
                & (val <= 140)  # Lower brightness
                & (blue_channel <= 140)  # Lower blue channel
            )

            # Recharge detection: lower saturation (<200), higher value (>140), higher blue (>140)
            # Also must be in the blue hue range (not purple)
            recharge_mask = (
                (hue >= 100)
                & (hue <= 130)  # Blue hue range (not purple)
                & (sat < 220)  # Lower saturation than empty
                & (val >= 130)  # Brighter than empty
                & (blue_channel >= 130)  # Higher blue than empty
                & ~purple_mask  # Not purple
            )

            # Find rightmost purple pixel (full elixir bars)
            purple_indices = np.where(purple_mask)[0]

            if len(purple_indices) == 0:
                # No purple - check if there's recharge (between 0 and 1)
                recharge_indices = np.where(recharge_mask)[0]
                if len(recharge_indices) > 0:
                    rightmost_recharge = recharge_indices[-1]
                    elixir = rightmost_recharge / PIXELS_PER_ELIXIR
                    return round(max(0.0, min(10.0, elixir)), 1)
                return 0.0

            rightmost_purple = purple_indices[-1]

            # Calculate full elixir from purple
            full_elixir = rightmost_purple / PIXELS_PER_ELIXIR

            # Now check for recharge bar AFTER the purple
            if rightmost_purple < len(row_bgr) - 1:
                # Check pixels after purple for recharge
                after_purple_mask = recharge_mask[rightmost_purple + 1 :]
                recharge_indices_after = np.where(after_purple_mask)[0]

                if len(recharge_indices_after) > 0:
                    # Find how far the recharge extends
                    rightmost_recharge_relative = recharge_indices_after[-1]

                    # Add 1 because indices are 0-based
                    partial_pixels = rightmost_recharge_relative + 1
                    partial_elixir = partial_pixels / PIXELS_PER_ELIXIR

                    total_elixir = full_elixir + partial_elixir
                    return round(max(0.0, min(10.0, total_elixir)), 1)

            # No recharge found, just return full elixir
            return round(max(0.0, min(10.0, full_elixir)), 1)

        except Exception as e:
            return None

    def read_elixir_ocr(self, frame: np.ndarray) -> float:
        """
        Read current elixir using OCR (slower, but more precise).
        Kept as fallback if bar detection fails.
        """
        try:
            # Use a region near the elixir number (left side of bar)
            x1, y1 = 295, 1225
            x2, y2 = 360, 1260
            h, w = frame.shape[:2]

            if x2 > w or y2 > h:
                return None

            elixir_region = frame[y1:y2, x1:x2]
            if elixir_region.size == 0:
                return None

            gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            scaled = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(scaled, config=config).strip()

            if text:
                if "10" in text:
                    return 10.0
                digits = re.findall(r"\d+", text)
                if digits:
                    val = int(digits[0])
                    if 0 <= val <= 10:
                        return float(val)

            return None

        except Exception:
            return None

    def get_card_elixir_region(self, card_slot: tuple) -> tuple:
        """
        Get the pixel region for a card's elixir cost number.
        Returns (x1, y1, x2, y2) or None if invalid.
        """
        x1, y1, x2, y2 = card_slot
        card_width = x2 - x1
        center_x = x1 + (card_width // 2)

        half_w = self.CARD_ELIXIR_WIDTH // 2
        elixir_x1 = center_x - half_w
        elixir_x2 = center_x + half_w
        elixir_y2 = y2 - self.CARD_ELIXIR_Y_OFFSET
        elixir_y1 = elixir_y2 - self.CARD_ELIXIR_HEIGHT

        return (elixir_x1, elixir_y1, elixir_x2, elixir_y2)

    def read_card_elixir_cost(self, frame: np.ndarray, card_slot: tuple) -> int:
        """
        Read the elixir cost from a specific card slot.
        Uses EasyOCR (GPU) if available, else Tesseract.
        Only digits 1-10 are valid elixir costs.
        """
        print(f"[OCR] Reading elixir cost for slot {card_slot}")
        try:
            h, w = frame.shape[:2]
            elixir_x1, elixir_y1, elixir_x2, elixir_y2 = self.get_card_elixir_region(
                card_slot
            )

            # Bounds check
            if elixir_x2 > w or elixir_y2 > h or elixir_x1 < 0 or elixir_y1 < 0:
                return None

            elixir_region = frame[elixir_y1:elixir_y2, elixir_x1:elixir_x2]

            if elixir_region.size == 0:
                return None

            # Use EasyOCR if available (GPU accelerated, ~8x faster)
            if self._use_easyocr and self._easyocr_reader is not None:
                return self._read_elixir_easyocr(elixir_region)
            elif TESSERACT_AVAILABLE:
                return self._read_elixir_tesseract(elixir_region)
            else:
                return None

        except Exception as e:
            print(f"[OCR] Error: {e}")
            return None

    def _read_elixir_easyocr(self, roi: np.ndarray) -> int:
        """Read elixir cost using EasyOCR (GPU accelerated)"""
        # EasyOCR works best with raw scaled images (no threshold)
        scaled = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Only allow digits 1-9 and 0 (for "10")
        results = self._easyocr_reader.readtext(
            scaled, allowlist="0123456789", detail=0
        )

        if results:
            text = "".join(results)
            digits = re.findall(r"\d+", text)
            if digits:
                val = int(digits[0])
                if 1 <= val <= 10:
                    return val
        return None

    def _read_elixir_tesseract(self, roi: np.ndarray) -> int:
        """Read elixir cost using Tesseract (CPU fallback)"""
        # Preprocess - elixir cost is white number on colored background
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Scale up for better OCR
        scaled = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # OCR - single character mode
        config = "--psm 10 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(scaled, config=config).strip()

        if text:
            digits = re.findall(r"\d+", text)
            if digits:
                val = int(digits[0])
                if 1 <= val <= 10:
                    return val
        return None

    def draw_debug_regions(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw debug rectangles showing where OCR regions are located.
        Returns a copy of frame with debug overlays.
        """
        debug = frame.copy()

        # Draw elixir bar region (cyan)
        x1, y1, x2, y2 = self.ELIXIR_BAR_REGION
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            debug,
            "ELIXIR BAR",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

        # Draw each card elixir region (magenta)
        from config.game_config import CARD_SLOTS

        for idx, slot in enumerate(CARD_SLOTS):
            ex1, ey1, ex2, ey2 = self.get_card_elixir_region(slot)
            cv2.rectangle(debug, (ex1, ey1), (ex2, ey2), (255, 0, 255), 2)
            cv2.putText(
                debug,
                f"C{idx+1}",
                (ex1, ey1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 255),
                1,
            )

        return debug

    def is_card_affordable(
        self, frame: np.ndarray, card_slot: tuple, current_elixir: float = None
    ) -> bool:
        """
        Check if a card is affordable (has enough elixir).
        If current_elixir is None, will try to read it from frame.
        Returns True if affordable, False if not, None if unknown.
        """
        if current_elixir is None:
            current_elixir = self.read_elixir(frame)

        if current_elixir is None:
            return None  # Can't determine

        card_cost = self.read_card_elixir_cost(frame, card_slot)

        if card_cost is None:
            return None  # Can't determine

        return current_elixir >= card_cost

    def detect_gray_card(self, frame: np.ndarray, card_slot: tuple) -> bool:
        """
        Detect if a card is grayed out (unaffordable) by checking saturation.
        Gray cards have very low saturation.
        Returns True if gray, False if colored.
        """
        try:
            x1, y1, x2, y2 = card_slot

            # Get center region of card (avoid borders)
            margin_x = (x2 - x1) // 4
            margin_y = (y2 - y1) // 4

            center_region = frame[
                y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x
            ]

            if center_region.size == 0:
                return False

            # Convert to HSV and check saturation
            hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])

            # Gray cards have saturation < 50 typically
            return avg_saturation < 50

        except Exception:
            return False

    def read_timer(self, frame: np.ndarray) -> str:
        """
        Read match timer from top-right corner.
        Returns string like "2:43" or None.
        """
        try:
            # Timer region (top right) - adjust based on resolution
            h, w = frame.shape[:2]
            x1, y1 = w - 120, 20
            x2, y2 = w - 20, 70

            timer_region = frame[y1:y2, x1:x2]

            if timer_region.size == 0:
                return None

            if self._use_easyocr and self._easyocr_reader is not None:
                scaled = cv2.resize(
                    timer_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                )
                results = self._easyocr_reader.readtext(
                    scaled, allowlist="0123456789:", detail=0
                )
                if results:
                    text = "".join(results)
                    if re.match(r"^\d{1,2}:\d{2}$", text):
                        return text
            elif TESSERACT_AVAILABLE:
                gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                scaled = cv2.resize(
                    thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                )
                config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
                text = pytesseract.image_to_string(scaled, config=config).strip()
                if re.match(r"^\d{1,2}:\d{2}$", text):
                    return text

            return None

        except Exception:
            return None

    def read_tower_hp(self, frame: np.ndarray, region: tuple) -> int:
        """
        Read tower HP from a specific region.
        Uses EasyOCR (GPU) if available.

        Args:
            frame: Full game frame
            region: (x1, y1, x2, y2) tuple for HP region

        Returns:
            HP value (1000-8000 range) or None if failed
        """
        try:
            x1, y1, x2, y2 = region
            h, w = frame.shape[:2]

            # Bounds check
            if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                return None

            hp_region = frame[y1:y2, x1:x2]

            if hp_region.size == 0:
                return None

            if self._use_easyocr and self._easyocr_reader is not None:
                return self._read_hp_easyocr(hp_region)
            elif TESSERACT_AVAILABLE:
                return self._read_hp_tesseract(hp_region)
            else:
                return None

        except Exception as e:
            return None

    def _read_hp_easyocr(self, roi: np.ndarray) -> int:
        """Read tower HP using EasyOCR (GPU accelerated)"""
        # Scale up for better accuracy
        scaled = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        results = self._easyocr_reader.readtext(
            scaled, allowlist="0123456789", detail=0
        )

        if results:
            text = "".join(results)
            digits = "".join(filter(str.isdigit, text))
            if digits:
                hp = int(digits)
                # Valid tower HP range
                if 100 <= hp <= 8000:
                    return hp
        return None

    def _read_hp_tesseract(self, roi: np.ndarray) -> int:
        """Read tower HP using Tesseract (CPU fallback)"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        scaled = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(scaled, config=config).strip()

        digits = "".join(filter(str.isdigit, text))
        if digits:
            hp = int(digits)
            if 100 <= hp <= 8000:
                return hp
        return None

    def read_tower_level(self, frame: np.ndarray, region: tuple) -> int:
        """
        Read tower level from a specific region.

        Args:
            frame: Full game frame
            region: (x1, y1, x2, y2) tuple for level region

        Returns:
            Level (1-15 range) or None if failed
        """
        try:
            x1, y1, x2, y2 = region
            h, w = frame.shape[:2]

            if x1 >= w or y1 >= h or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                return None

            level_region = frame[y1:y2, x1:x2]

            if level_region.size == 0:
                return None

            if self._use_easyocr and self._easyocr_reader is not None:
                scaled = cv2.resize(
                    level_region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                )
                results = self._easyocr_reader.readtext(
                    scaled, allowlist="0123456789", detail=0
                )
                if results:
                    text = "".join(results)
                    digits = "".join(filter(str.isdigit, text))
                    if digits:
                        level = int(digits)
                        if 1 <= level <= 15:
                            return level
            elif TESSERACT_AVAILABLE:
                gray = cv2.cvtColor(level_region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                scaled = cv2.resize(
                    binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                )
                config = "--psm 10 -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(scaled, config=config).strip()
                digits = "".join(filter(str.isdigit, text))
                if digits:
                    level = int(digits)
                    if 1 <= level <= 15:
                        return level

            return None

        except Exception:
            return None
