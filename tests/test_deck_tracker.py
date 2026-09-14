"""Tests for game_state.deck_tracker.DeckTracker (plain script, no pytest).

Run:  ~/venvs/clash/bin/python tests/test_deck_tracker.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_state.deck_tracker import DeckTracker, base_name  # noqa: E402


class GameSim:
    """Reference implementation of the in-game cycle (the ground truth)."""

    def __init__(self, hand, queue):
        self.hand = list(hand)          # 4 cards by slot
        self.queue = list(queue)        # 4 cards in arrival order

    def play(self, slot):
        played = self.hand[slot]
        arrival = self.queue.pop(0)
        self.hand[slot] = arrival
        self.queue.append(played)
        return played, arrival

    def next_card(self):
        return self.queue[0]


def test_base_name():
    assert base_name("base/fireball") == "fireball"
    assert base_name("evolution/knight_evo") == "knight"
    assert base_name("Knight-Evolution") == "knight"
    assert base_name("unknown") is None
    assert base_name("waiting_for_card_2") is None
    assert base_name(None) is None
    print("ok  base_name")


def test_learning_sequence():
    dt = DeckTracker(deck=list("abcdefgh"))
    for i, c in enumerate("abcd"):
        dt.note_hand(i, c)
    assert not dt.ready and not dt.complete, dt.snapshot()

    sim = GameSim("abcd", list("efgh"))
    # Play slot 1 (b) -> arrival e
    played, arrival = sim.play(1)
    dt.note_play(1)
    assert dt.queue == [None, None, None, "b"], dt.queue
    assert not dt.ready
    dt.note_arrival(1, arrival)
    assert dt.hand[1] == "e"

    # Play slot 2 (c) -> f ; slot 3 (d) -> g
    for slot in (2, 3):
        played, arrival = sim.play(slot)
        dt.note_play(slot)
        dt.note_arrival(slot, arrival)

    # Play slot 0 (a) -> h ; after this the tracker queue should be complete
    played, arrival = sim.play(0)
    dt.note_play(0)
    dt.note_arrival(0, arrival)

    assert dt.complete, dt.snapshot()
    assert dt.queue == ["b", "c", "d", "a"], dt.queue
    assert dt.predict_next() == sim.next_card() == "b"

    # Track 30 more plays; prediction must always match the simulator.
    for step in range(30):
        assert dt.predict_next() == sim.next_card(), (step, dt.snapshot())
        slot = random.randrange(4)
        played, arrival = sim.play(slot)
        dt.note_play(slot)
        dt.note_arrival(slot, arrival)
    print("ok  learning_sequence (queue completes in <=4 plays, exact afterwards)")


def test_random_decks():
    rng = random.Random(7)
    cards = [f"card{i}" for i in range(8)]
    for trial in range(20):
        rng.shuffle(cards)
        hand = cards[:4]
        queue = cards[4:]
        dt = DeckTracker(deck=list(cards))
        for i, c in enumerate(hand):
            dt.note_hand(i, c)
        sim = GameSim(hand, queue)
        wrong = 0
        for step in range(60):
            if dt.ready and dt.predict_next() != sim.next_card():
                wrong += 1
            slot = rng.randrange(4)
            played, arrival = sim.play(slot)
            dt.note_play(slot)
            dt.note_arrival(slot, arrival)
        assert wrong == 0, (trial, wrong, dt.snapshot())
        assert dt.complete
        assert dt.queue == sim.queue, (dt.queue, sim.queue)
    print("ok  random_decks (20 decks x 60 plays, predictions exact once ready)")


def test_no_deck_known():
    """Predictions must also become correct without a known deck list."""
    dt = DeckTracker()
    sim = GameSim("abcd", "efgh")
    for i, c in enumerate("abcd"):
        dt.note_hand(i, c)
    for _ in range(4):
        slot = random.randrange(4)
        played, arrival = sim.play(slot)
        dt.note_play(slot)
        dt.note_arrival(slot, arrival)
    assert dt.complete, dt.snapshot()
    for _ in range(20):
        assert dt.predict_next() == sim.next_card()
        slot = random.randrange(4)
        played, arrival = sim.play(slot)
        dt.note_play(slot)
        dt.note_arrival(slot, arrival)
    print("ok  no_deck_known")


def test_elimination():
    dt = DeckTracker(deck=list("abcdefgh"))
    for i, c in enumerate("abcd"):
        dt.note_hand(i, c)
    sim = GameSim("abcd", list("efgh"))
    for slot in (0, 1, 2):
        played, arrival = sim.play(slot)
        dt.note_play(slot)
        dt.note_arrival(slot, arrival)
    # queue should now be [h, a, b, c] -- h filled by elimination
    assert dt.queue == ["h", "a", "b", "c"], dt.queue
    print("ok  elimination")


def test_mismatch_and_resync():
    dt = DeckTracker(deck=list("abcdefgh"))
    for i, c in enumerate("abcd"):
        dt.note_hand(i, c)
    sim = GameSim("abcd", list("efgh"))
    # Learn the full queue first (4 observed plays in the same slot).
    for _ in range(4):
        played, arrival = sim.play(0)
        dt.note_play(0)
        dt.note_arrival(0, arrival)
    assert dt.complete and dt.queue == ["a", "e", "f", "g"], dt.snapshot()
    # Now a play with a wrong arrival must be flagged as a mismatch.
    sim.play(1)
    dt.note_play(1)
    dt.note_arrival(1, "x_wrong")
    assert dt.mismatches == 1, dt.snapshot()
    # Missed-play resync: a card we KNEW was waiting appears in hand without
    # a play event -> queue must be consumed up to and including it.
    dt2 = DeckTracker(deck=list("abcdefgh"))
    for i, c in enumerate("abcd"):
        dt2.note_hand(i, c)
    sim2 = GameSim("abcd", list("efgh"))
    for _ in range(4):
        played, arrival = sim2.play(0)
        dt2.note_play(0)
        dt2.note_arrival(0, arrival)
    assert dt2.complete, dt2.snapshot()
    played, arrival = sim2.play(1)          # we miss this play event entirely
    dt2.note_hand(1, arrival)               # ... but observe the arrival card
    assert dt2.resyncs == 1, dt2.snapshot()
    assert dt2.queue[0] == "e", dt2.snapshot()
    print("ok  mismatch_and_resync")


if __name__ == "__main__":
    test_base_name()
    test_learning_sequence()
    test_random_decks()
    test_no_deck_known()
    test_elimination()
    test_mismatch_and_resync()
    print("\nALL DECK TRACKER TESTS PASSED")
