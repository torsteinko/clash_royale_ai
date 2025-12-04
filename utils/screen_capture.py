"""Screen capture utilities"""

import pyautogui
import cv2
import numpy as np


def capture_screen():
    """Capture screenshot"""
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    # Save to file for debugging
    cv2.imwrite("debug_screenshot.png", frame)
    return frame
