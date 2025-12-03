"""OCR for elixir, timer, tower HP"""

import cv2
import numpy as np
import pytesseract
import re


class OCRReader:
    def __init__(self):
        # pytesseract.pytesseract.tesseract_cmd = (
        #     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        # )

        # Elixir bar region (where the purple bar and number are)
        # Adjust these based on your screen resolution
        self.ELIXIR_NUMBER_REGION = (305, 1227, 380, 1270)  # (x1, y1, x2, y2)

        # Card elixir cost regions (bottom of each card slot)
        # These are relative offsets from card slot bottom-left
        self.CARD_ELIXIR_OFFSET = {
            "x_offset": 45,  # Center of card
            "y_offset": -25,  # From bottom of card slot
            "width": 30,
            "height": 30,
        }

    def read_elixir(self, frame: np.ndarray) -> float:
        """
        Read current elixir from the elixir bar.
        Returns float 0-10, or None if cannot read.
        """
        try:
            x1, y1, x2, y2 = self.ELIXIR_NUMBER_REGION
            h, w = frame.shape[:2]

            # Bounds check
            if x2 > w or y2 > h:
                return None

            elixir_region = frame[y1:y2, x1:x2]

            if elixir_region.size == 0:
                return None

            # Preprocess for OCR
            # The elixir number is white/pink on purple background
            gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)

            # Threshold to get white text
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            # Scale up for better OCR
            scaled = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # OCR with digit-only config
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(scaled, config=config).strip()

            # Parse number
            if text:
                # Handle "10" specially
                if "10" in text:
                    return 10.0
                # Get first digit
                digits = re.findall(r"\d+", text)
                if digits:
                    val = int(digits[0])
                    if 0 <= val <= 10:
                        return float(val)

            return None

        except Exception as e:
            return None

    def read_card_elixir_cost(self, frame: np.ndarray, card_slot: tuple) -> int:
        """
        Read the elixir cost from a specific card slot.
        card_slot: (x1, y1, x2, y2) of the card slot
        Returns int 1-10, or None if cannot read.
        """
        try:
            x1, y1, x2, y2 = card_slot
            h, w = frame.shape[:2]

            # Calculate elixir number position (bottom center of card)
            card_width = x2 - x1
            elixir_x1 = x1 + (card_width // 2) - 15
            elixir_x2 = elixir_x1 + 30
            elixir_y1 = y2 - 35
            elixir_y2 = y2 - 5

            # Bounds check
            if elixir_x2 > w or elixir_y2 > h or elixir_x1 < 0 or elixir_y1 < 0:
                return None

            elixir_region = frame[elixir_y1:elixir_y2, elixir_x1:elixir_x2]

            if elixir_region.size == 0:
                return None

            # Preprocess - elixir cost is white number on colored background
            gray = cv2.cvtColor(elixir_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Scale up
            scaled = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # OCR
            config = "--psm 10 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(scaled, config=config).strip()

            if text:
                digits = re.findall(r"\d+", text)
                if digits:
                    val = int(digits[0])
                    if 1 <= val <= 10:
                        return val

            return None

        except Exception:
            return None

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

            # Preprocess
            gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # OCR
            config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
            text = pytesseract.image_to_string(scaled, config=config).strip()

            # Validate format (M:SS or MM:SS)
            if re.match(r"^\d{1,2}:\d{2}$", text):
                return text

            return None

        except Exception:
            return None
