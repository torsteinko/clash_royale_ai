"""Screen capture utilities — ADB (headless) or desktop fallback.

Backends:
  * "adb": capture the framebuffer of an Android emulator / device via
    `adb exec-out screencap -p` (see detection/adb_capture.py).  This is the
    headless, cross-platform replacement for the old Windows-only win32 MEmu
    window capture.  Device selection: `serial` argument, ADB_SERIAL env var,
    or the first device from `adb devices`.
  * "pyautogui": screenshot the local desktop (legacy interactive use).

`ScreenCapture` picks ADB when a device is reachable, otherwise falls back to
pyautogui if installed.  `capture_screen()` is kept for backwards
compatibility with existing call sites.
"""

import os

import cv2
import numpy as np


class ScreenCapture:
    def __init__(self, serial: str = None, mode: str = "auto", crop=None):
        """
        Args:
            serial: ADB device serial (e.g. "emulator-5554", "100.x.y.z:5555").
                    Falls back to $ADB_SERIAL, then the first attached device.
            mode: "auto" | "adb" | "pyautogui"
            crop: optional (x1, y1, x2, y2) crop applied to every frame
        """
        self.serial = serial or os.environ.get("ADB_SERIAL") or None
        self.mode = mode
        self.crop = crop
        self._backend = None

        if mode in ("auto", "adb"):
            try:
                from detection.adb_capture import list_devices

                devices = list_devices()
                if devices:
                    if self.serial is None:
                        self.serial = devices[0]
                    if self.serial in devices:
                        self._backend = "adb"
            except Exception:
                pass

        if self._backend is None and mode in ("auto", "pyautogui"):
            try:
                import pyautogui  # noqa: F401

                self._backend = "pyautogui"
            except ImportError:
                self._backend = None

    # ------------------------------------------------------------------ API
    def available(self) -> bool:
        return self._backend is not None

    def describe(self) -> str:
        if self._backend == "adb":
            return f"adb (serial={self.serial})"
        if self._backend == "pyautogui":
            return "pyautogui (local desktop)"
        return "none"

    def capture(self):
        """Return a BGR frame (np.ndarray) or None when capture fails."""
        if self._backend == "adb":
            from detection.adb_capture import capture_adb

            frame = capture_adb(self.serial)
        elif self._backend == "pyautogui":
            import pyautogui

            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            return None

        if frame is not None and self.crop:
            x1, y1, x2, y2 = self.crop
            frame = frame[y1:y2, x1:x2]
        return frame


def capture_screen():
    """Backwards-compatible helper: capture one frame via the best backend."""
    return ScreenCapture().capture()
