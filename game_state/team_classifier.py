"""Team ownership from the rendered blue/red tint.

Clash Royale renders the recording player's units with blue accents and the
opponent's with red accents.  This is the game's own team label: it is always
present, survives tower losses and bridge crossings, and is far more reliable
than position/spawn-zone heuristics (crossings happen constantly; direction
flips when troops chase) or the learned ally_/enemy_ detector heads
(measured ~20% side-label errors on real frames - see
skills/gaming/clash-royale-ai/references/detection-analysis.md).

Metric per detection: central crop (15% padding), count strongly
blue-dominant (B > R+15 and B > G+8) vs red-dominant (R > B+15 and R > G+8)
pixels; score = (n_blue - n_red) / (n_blue + n_red), in [-1, +1];
+1 = clearly blue team, -1 = clearly red team.  Grass/dirt/gray pixels are
excluded by construction (green-dominant or low-saturation pixels match
neither mask).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional, Tuple

import numpy as np

# Channel margins (0-255 BGR scale).
_BLUE_MARGIN = 15   # B must exceed R by this much
_BLUE_GREEN = 8     # ... and G by this much
_RED_MARGIN = 15
_RED_GREEN = 8

# |score| below this = ambiguous (occlusion, spell flash, tiny sprite).
AMBIGUOUS = 0.25
# |score| at/above this = trust it over the detector's class prefix.
STRONG = 0.35


def tint_score(frame: np.ndarray, bbox, pad: float = 0.15) -> Optional[float]:
    """Blue-minus-red tint score in [-1, +1] for one detection bbox."""
    if frame is None or frame.size == 0:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 6:
        return None
    x1 += int(w * pad)
    x2 -= int(w * pad)
    y1 += int(h * pad)
    y2 -= int(h * pad)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size < 30:
        return None
    crop = crop.astype(np.int16)
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    n_blue = int(((b > r + _BLUE_MARGIN) & (b > g + _BLUE_GREEN)).sum())
    n_red = int(((r > b + _RED_MARGIN) & (r > g + _RED_GREEN)).sum())
    total = n_blue + n_red
    if total == 0:
        return 0.0
    return (n_blue - n_red) / total


def classify_team(
    frame: np.ndarray, bbox, min_abs: float = STRONG
) -> Tuple[Optional[str], Optional[float]]:
    """Return ("ally"|"enemy"|None, score).  None team = ambiguous."""
    score = tint_score(frame, bbox)
    if score is None:
        return None, None
    if score >= min_abs:
        return "ally", score
    if score <= -min_abs:
        return "enemy", score
    return None, score


class TeamVoter:
    """Temporal majority vote per track, so spell flashes / occlusions can't
    flip a unit's team mid-track."""

    def __init__(self, window: int = 5):
        self.window = window
        self._votes = defaultdict(lambda: deque(maxlen=window))

    def vote(self, track_id: int, score: Optional[float]) -> Tuple[Optional[str], float]:
        """Add an observation; return (team or None, mean score)."""
        if score is not None:
            self._votes[track_id].append(float(score))
        scores = self._votes[track_id]
        if not scores:
            return None, 0.0
        mean = sum(scores) / len(scores)
        if mean >= AMBIGUOUS:
            return "ally", mean
        if mean <= -AMBIGUOUS:
            return "enemy", mean
        return None, mean

    def forget(self, track_id: int) -> None:
        self._votes.pop(track_id, None)

    def reset(self) -> None:
        self._votes.clear()
