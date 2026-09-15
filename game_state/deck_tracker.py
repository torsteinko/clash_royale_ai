"""Deck cycle (card rotation) tracker for Clash Royale.

How the in-match card cycle works (matches in-game behaviour):
  * A deck holds 8 cards: 4 in hand, 4 waiting in an ordered queue.
  * When you play the card from a hand slot, the played card is appended to
    the BACK of the waiting queue, and the FRONT of the queue slides into the
    freed slot (it is the card shown as "next" during the 1s waiting
    animation).

Tracking strategy (key property, easy to get wrong):
  The queue update on a play only needs the IDENTITY of the played card:

      queue = queue[1:] + [played]      # front consumed, played appended

  The queue starts as [?, ?, ?, ?].  Every observed play appends a *known*
  card to the tail and consumes one head entry, so after at most 4 observed
  plays the queue is completely known and every future arrival can be
  predicted exactly.  Arrival observations (the card confirmed in the slot
  after the waiting animation) are used to validate predictions and to
  resync when a play was missed.

What is NOT needed: the relative order of the 4 hand cards.  Removing the
played card from the hand and appending it at the tail yields the same queue
for any position of the played card inside the hand block, so hand slot
order never affects predictions.

This module is pure logic (no OpenCV / no game knowledge beyond the cycle).
All names are normalized base names ("knight", "elixir_golem") or None.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WAITING = {"unknown", "waiting_for_card"}


def base_name(name: Optional[str]) -> Optional[str]:
    """Normalize a card name: drop folder prefixes / evolution suffixes."""
    if name is None:
        return None
    n = str(name).strip().lower()
    if not n or n in _WAITING or n.startswith("waiting_for_card"):
        return None
    n = n.split("/")[-1]
    for suffix in ("_evolution", "-evolution", "_evo"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n


class DeckTracker:
    """Track the 4 hand slots and the ordered 4-card waiting queue."""

    def __init__(self, deck: Optional[List[str]] = None):
        self.deck: Optional[List[str]] = None          # 8 base names, if known
        if deck:
            self.deck = [b for b in (base_name(c) for c in deck) if b]
        self.hand: List[Optional[str]] = [None] * 4    # observed per slot
        self.queue: List[Optional[str]] = [None] * 4   # arrival order; [0] arrives next
        self.play_log: List[Dict] = []                 # debug / audit trail
        self.mismatches: int = 0
        self.resyncs: int = 0
        self.unknown_plays: int = 0
        self._pending: Optional[Dict] = None

    # ------------------------------------------------------------------ deck
    def set_deck(self, deck: Optional[List[str]]) -> None:
        """Provide the 8-card deck (base names) once inferred."""
        if not deck:
            return
        self.deck = [b for b in (base_name(c) for c in deck) if b]
        self._eliminate()

    # ---------------------------------------------------------- observations
    def note_hand(self, slot: int, card: Optional[str]) -> None:
        """Observe a card in a hand slot (from the card detector)."""
        card = base_name(card)
        if card is None:
            return
        slot = int(slot)
        if not 0 <= slot < 4:
            return
        if self.hand[slot] == card:
            return
        if self._pending and self._pending.get("slot") == slot:
            self.note_arrival(slot, card)
            return
        # Resync: a card that was still waiting is suddenly in hand -> we
        # missed a play event.  Consume the queue up to and including it.
        if card in self.queue:
            idx = self.queue.index(card)
            self.resyncs += 1
            logger.warning(
                "DeckTracker resync: %s appeared in slot %d while at queue[%d] "
                "(missed play event?)", card, slot, idx,
            )
            self.queue = self.queue[idx + 1:] + [None]
        self.hand[slot] = card
        self._eliminate()

    def note_play(self, slot: int) -> None:
        """A play was detected in `slot` (waiting_for_card appeared)."""
        slot = int(slot)
        if not 0 <= slot < 4:
            return
        if self._pending and self._pending.get("slot") == slot:
            return  # idempotent: play already recorded for this slot
        played = self.hand[slot]
        expected = self.queue[0]
        if played is None:
            self.unknown_plays += 1
        self._pending = {"slot": slot, "played": played, "expected": expected}
        self.play_log.append(
            {"slot": slot, "played": played, "expected_next": expected}
        )
        self.hand[slot] = None
        # front consumed, played appended (unknown stays unknown)
        self.queue = self.queue[1:] + [played]
        self._eliminate()
        logger.debug(
            "Play in slot %d: played=%s | expected arrival=%s | queue=%s",
            slot, played, expected, self.queue,
        )

    def note_arrival(self, slot: int, card: Optional[str]) -> None:
        """A new card was confirmed in `slot` after the waiting animation."""
        card = base_name(card)
        if card is None:
            return
        slot = int(slot)
        if not 0 <= slot < 4:
            return
        expected = None
        if self._pending and self._pending.get("slot") == slot:
            expected = self._pending.get("expected")
        if expected is not None and expected != card:
            self.mismatches += 1
            logger.warning(
                "Deck cycle mismatch: expected %s, observed %s (slot %d). "
                "Prediction for this arrival was wrong.", expected, card, slot,
            )
        self.hand[slot] = card
        if self._pending and self._pending.get("slot") == slot:
            self._pending = None
        self._eliminate()

    # ------------------------------------------------------------ knowledge
    def _eliminate(self) -> None:
        """Fill the last unknown queue slot when the deck is fully known."""
        if not self.deck:
            return
        if any(h is None for h in self.hand):
            return  # a hand slot is mid-arrival; multiset reasoning incomplete
        known = Counter(c for c in self.hand if c) + Counter(
            c for c in self.queue if c
        )
        missing = list((Counter(self.deck) - known).elements())
        none_positions = [i for i, q in enumerate(self.queue) if q is None]
        if len(missing) == 1 and len(none_positions) == 1:
            self.queue[none_positions[0]] = missing[0]

    @property
    def ready(self) -> bool:
        """True when the next arriving card is known."""
        return self.queue[0] is not None

    @property
    def complete(self) -> bool:
        """True when the whole waiting queue is known (steady-state tracking)."""
        return all(q is not None for q in self.queue)

    def predict_next(self) -> Optional[str]:
        """The card that will slide into the hand on the next play."""
        return self.queue[0]

    def predict_hand(self) -> List[Optional[str]]:
        """Current hand contents per slot (observations only)."""
        return list(self.hand)

    def snapshot(self) -> Dict:
        """Compact status dict for state/debug output."""
        return {
            "hand": list(self.hand),
            "queue": list(self.queue),
            "next": self.queue[0],
            "ready": self.ready,
            "complete": self.complete,
            "plays": len(self.play_log),
            "mismatches": self.mismatches,
            "resyncs": self.resyncs,
            "unknown_plays": self.unknown_plays,
        }

    def reset(self) -> None:
        self.hand = [None] * 4
        self.queue = [None] * 4
        self.play_log = []
        self.mismatches = 0
        self.resyncs = 0
        self.unknown_plays = 0
        self._pending = None
