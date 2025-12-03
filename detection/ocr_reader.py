"""OCR for elixir, timer, tower HP"""

import cv2
import pytesseract

class OCRReader:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def read_elixir(self, frame, region):
        """Read elixir count"""
        # TODO: Implement OCR
        return None
    
    def read_timer(self, frame, region):
        """Read match timer"""
        # TODO: Implement OCR
        return None
