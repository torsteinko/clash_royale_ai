"""YOLO troop detection wrapper"""

from ultralytics import YOLO
import numpy as np

class TroopDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        """Detect troops using YOLO"""
        results = self.model.predict(frame, conf=0.5, imgsz=1280, verbose=False)
        return results
