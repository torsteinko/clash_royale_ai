"""ADB-based screen capture — headless cross-platform replacement for the
Windows-only win32 MEmu window capture (win32gui/win32ui/win32con).

Works anywhere ADB can reach a device or emulator:
  - Linux CI / headless VMs:  emulator with -no-window, or physical device over USB/net
  - Android emulator on GCP (nested virtualization): serial like "emulator-5554"
  - Physical phone over Tailscale: `adb connect <tailscale-ip>:5555`

Same return contract as `capture_background_window(hwnd)` in
policy/background_controller.py: a BGR uint8 numpy array (H, W, 3), or None
when the device is unreachable / the capture failed.

Usage:
    from detection.adb_capture import list_devices, capture_adb, tap, swipe

    serial = list_devices()[0]
    frame = capture_adb(serial)           # np.ndarray BGR or None
    tap(serial, 540, 1600)                # card slot tap, etc.
"""

from __future__ import annotations

import subprocess
import time

import cv2
import numpy as np

ADB_PATH = "adb"

# A valid adb screencap PNG is always > few hundred bytes; anything smaller is
# a truncated/garbage read from a wedged device.
_MIN_PNG_BYTES = 100


def _adb(args: list[str], serial: str | None = None, timeout: float = 20.0) -> str | None:
    """Run an adb command, targeting `serial` when given. Returns stdout as
    text, or None on failure."""
    cmd = [ADB_PATH]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def _adb_bytes(args: list[str], serial: str | None = None,
               timeout: float = 20.0) -> bytes | None:
    """Like `_adb` but returns raw stdout bytes (for screencap PNG streams)."""
    cmd = [ADB_PATH]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def list_devices() -> list[str]:
    """Serials of devices in `device` state (excludes unauthorized/offline)."""
    out = _adb(["devices"])
    if not out:
        return []
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def wait_for_device(serial: str | None = None, timeout_s: float = 60.0) -> bool:
    """Block until a device is available (or timeout)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if list_devices():
            return True
        time.sleep(2)
    return False


def capture_adb(serial: str | None = None) -> np.ndarray | None:
    """Capture the device screen via `adb exec-out screencap -p`.

    Returns BGR uint8 (H, W, 3) numpy array — same contract as the win32
    capture in background_controller.py — or None if capture failed.

    Note: the win32 path crops MEmu window borders (crop_top=35 etc.);
    screencap returns the RAW device framebuffer, so no border crop is
    needed. If your emulator window adds chrome, crop via the device
    resolution instead (the emulator renders pure 1080x1920, etc.).
    """
    raw = _adb_bytes(["exec-out", "screencap", "-p"], serial=serial, timeout=15.0)
    if not raw or len(raw) < _MIN_PNG_BYTES:
        return None
    img = cv2.imdecode(np.frombuffer(raw, dtype="uint8"), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return np.ascontiguousarray(img)


def tap(serial: str, x: int, y: int) -> bool:
    """Send a tap. Coordinates are in DEVICE pixels (same space as capture)."""
    return _adb(["shell", "input", "tap", str(int(x)), str(int(y))],
                serial=serial) is not None


def swipe(serial: str, x1: int, y1: int, x2: int, y2: int,
          duration_ms: int = 200) -> bool:
    return _adb(["shell", "input", "swipe", str(int(x1)), str(int(y1)),
                 str(int(x2)), str(int(y2)), str(int(duration_ms))],
                serial=serial) is not None


def start_emulator_capture(serial: str, warmup: bool = True) -> bool:
    """Sanity check before a play loop: device reachable and screencap works."""
    if not list_devices():
        return False
    frame = capture_adb(serial)
    if frame is None:
        return False
    if warmup:
        # Some emulators return a black first frame right after boot; a second
        # read confirms the pipeline is hot.
        time.sleep(0.5)
        frame = capture_adb(serial)
    return frame is not None and frame.size > 0
