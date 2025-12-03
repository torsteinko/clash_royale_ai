# game_state/state_extractor.py
"""Complete game state extractor with YOLO + OCR + Card detection"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from typing import Dict, List, Optional

from detection.card_detector import CardDetector
from config.game_config import CARD_SLOTS, ELIXIR_REGION, TIMER_REGION, TOWER_REGIONS


class GameStateExtractor:
    """Extract complete game state from screenshot"""
    
    def __init__(self, yolo_model_path='runs/detect/clash_royale_FINAL_1280px/weights/best.pt'):
        print("\n" + "="*80)
        print("🎮 INITIALIZING GAME STATE EXTRACTOR")
        print("="*80)
        
        # Load YOLO model
        print("📦 Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        print(f"   ✅ YOLO loaded")
        
        # Load card detector
        print("🎴 Loading card templates...")
        self.card_detector = CardDetector()
        print(f"   ✅ Card detector ready")
        
        # Configure OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        print("="*80)
        print("✅ Ready to detect!")
        print("="*80 + "\n")
    
    def extract_state(self, frame: np.ndarray) -> Dict:
        """Extract complete game state"""
        
        state = {}
        
        # 1. Detect troops/buildings
        troops = self._detect_troops(frame)
        state['troops'] = troops
        
        # 2. Detect cards in hand
        state['cards_in_hand'] = self.card_detector.detect_cards_in_hand(frame)
        
        # 3. Read elixir
        state['elixir'] = self._read_elixir(frame)
        
        # 4. Read timer
        state['match_time'] = self._read_timer(frame)
        
        # 5. Read tower health
        state['towers'] = self._read_tower_health(frame)
        
        # 6. Derived metrics
        state['is_double_elixir'] = state['match_time'] is not None and state['match_time'] <= 60
        state['is_overtime'] = state['match_time'] is not None and state['match_time'] <= 0
        
        return state
    
    def _detect_troops(self, frame: np.ndarray) -> Dict:
        """Detect troops using YOLO"""
        
        results = self.yolo.predict(frame, conf=0.5, imgsz=1280, verbose=False)
        
        ally_troops = []
        enemy_troops = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                
                # Determine side (ally = bottom half, enemy = top half)
                center_y = (y1 + y2) / 2
                side = 'ally' if center_y > 360 else 'enemy'
                
                troop_data = {
                    'type': class_name,
                    'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf
                }
                
                if side == 'ally':
                    ally_troops.append(troop_data)
                else:
                    enemy_troops.append(troop_data)
        
        return {
            'ally': ally_troops,
            'enemy': enemy_troops,
            'total_ally': len(ally_troops),
            'total_enemy': len(enemy_troops)
        }
    
    def _read_elixir(self, frame: np.ndarray) -> Optional[float]:
        """Read elixir count"""
        
        x1, y1, x2, y2 = ELIXIR_REGION
        elixir_img = frame[y1:y2, x1:x2]
        
        # Preprocess
        gray = cv2.cvtColor(elixir_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, None, fx=3, fy=3)
        
        # OCR
        try:
            config = '--psm 7 --oem 3 -c tesseract_char_whitelist=0123456789./'
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            if '/' in text:
                current = float(text.split('/')[0])
            else:
                current = float(text)
            
            return min(10.0, max(0.0, current))
        except:
            return None
    
    def _read_timer(self, frame: np.ndarray) -> Optional[int]:
        """Read match timer"""
        
        x1, y1, x2, y2 = TIMER_REGION
        timer_img = frame[y1:y2, x1:x2]
        
        # Preprocess
        gray = cv2.cvtColor(timer_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, None, fx=3, fy=3)
        
        # OCR
        try:
            config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:'
            text = pytesseract.image_to_string(binary, config=config).strip()
            
            if ':' in text:
                parts = text.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            return int(text)
        except:
            return None
    
    def _read_tower_health(self, frame: np.ndarray) -> Dict[str, Optional[int]]:
        """Read tower health"""
        
        towers = {}
        
        for tower_name, (x1, y1, x2, y2) in TOWER_REGIONS.items():
            hp_img = frame[y1:y2, x1:x2]
            
            try:
                gray = cv2.cvtColor(hp_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                binary = cv2.resize(binary, None, fx=2, fy=2)
                
                config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
                text = pytesseract.image_to_string(binary, config=config).strip()
                
                towers[tower_name] = int(text)
            except:
                towers[tower_name] = None
        
        return towers
    
    def visualize_state(self, frame: np.ndarray, state: Dict) -> np.ndarray:
        """Draw detections on frame"""
        
        vis_frame = frame.copy()
        
        # Draw ally troops (green)
        for troop in state['troops']['ally']:
            x1, y1, x2, y2 = troop['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{troop['type']} {troop['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw enemy troops (red)
        for troop in state['troops']['enemy']:
            x1, y1, x2, y2 = troop['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{troop['type']} {troop['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw cards
        for idx, card in enumerate(state['cards_in_hand']):
            x1, y1, x2, y2 = CARD_SLOTS[idx]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis_frame, card, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw elixir
        if state['elixir'] is not None:
            cv2.putText(vis_frame, f"Elixir: {state['elixir']:.1f}", (550, 640),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Draw timer
        if state['match_time'] is not None:
            mins = state['match_time'] // 60
            secs = state['match_time'] % 60
            cv2.putText(vis_frame, f"Time: {mins}:{secs:02d}", (1100, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        return vis_frame
