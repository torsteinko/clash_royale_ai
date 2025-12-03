# detection/card_detector.py
"""Card detection using template matching"""

import cv2
import numpy as np
from pathlib import Path
from config.game_config import CARD_SLOTS


class CardDetector:
    """Detect cards in hand using normalized templates"""
    
    def __init__(self, templates_dir='data/card_templates'):
        self.templates = self._load_templates(templates_dir)
        print(f"   ✅ Loaded {len(self.templates)} card templates")
    
    def _load_templates(self, templates_dir):
        """Load all normalized card templates"""
        templates = {}
        templates_path = Path(templates_dir)
        
        for subfolder in ['base', 'evolution', 'hero']:
            folder = templates_path / subfolder
            if not folder.exists():
                continue
            
            for img_path in folder.glob('*.png'):
                card_name = img_path.stem
                template = cv2.imread(str(img_path))
                if template is not None:
                    templates[card_name] = template
        
        return templates
    
    def detect_cards_in_hand(self, frame: np.ndarray) -> list:
        """Detect which 4 cards are in hand"""
        
        cards = []
        
        for slot_idx, (x1, y1, x2, y2) in enumerate(CARD_SLOTS):
            card_region = frame[y1:y2, x1:x2]
            
            if card_region.size == 0:
                cards.append('unknown')
                continue
            
            # Normalize card (remove border, resize)
            normalized = self._normalize_card(card_region)
            
            # Match template
            best_match = self._match_template(normalized)
            cards.append(best_match)
        
        return cards
    
    def _normalize_card(self, card_region: np.ndarray) -> np.ndarray:
        """Normalize card region (same as templates)"""
        
        height, width = card_region.shape[:2]
        
        # Crop border (15%)
        crop = 0.15
        x = int(width * crop)
        y = int(height * crop)
        w = int(width * (1 - 2 * crop))
        h = int(height * (1 - 2 * crop))
        
        cropped = card_region[y:y+h, x:x+w]
        
        # Resize to template size (120x90)
        resized = cv2.resize(cropped, (120, 90), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _match_template(self, card_image: np.ndarray) -> str:
        """Find best matching template"""
        
        best_score = 0
        best_card = 'unknown'
        
        for card_name, template in self.templates.items():
            result = cv2.matchTemplate(card_image, template, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score:
                best_score = score
                best_card = card_name
        
        # Confidence threshold
        if best_score < 0.55:  # Adjust as needed
            return 'unknown'
        
        return best_card
