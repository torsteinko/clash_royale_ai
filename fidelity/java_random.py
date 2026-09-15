"""Exact java.util.Random + Collections.shuffle replication.

Why this exists: the Java reference shuffles each player's 8-card deck at reset
(`Hand.java`: `Collections.shuffle(deckCards, random)`), with
`GameSession.reset`: blue gets `new Random(seed)`, red `new Random(seed + 1)`
when an explicit reset seed is given (which the replay recorder does:
`env.reset(seed=...)`). To replay a recorded match action-for-action in gpusim,
the gpusim deck order must equal the *shuffled* Java order — so we replicate the
JDK RNG bit-exactly instead of approximating.

`java_shuffle` matches `Collections.shuffle(ArrayList, Random)` (ArrayList is
RandomAccess -> the simple Fisher-Yates loop) and `JavaRandom` matches the JDK
LCG (`nextInt(bound)` incl. the 32-bit overflow rejection loop).
"""
from __future__ import annotations

_MULT = 0x5DEECE66D
_ADD = 0xB
_MASK = (1 << 48) - 1
_INT32 = 1 << 31


class JavaRandom:
    """java.util.Random (48-bit LCG)."""

    def __init__(self, seed: int):
        self._seed = (int(seed) ^ _MULT) & _MASK

    def _next(self, bits: int) -> int:
        self._seed = (self._seed * _MULT + _ADD) & _MASK
        return self._seed >> (48 - bits)

    def next_int(self, bound: int) -> int:
        assert bound > 0
        if bound & (bound - 1) == 0:  # power of two fast path
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            # Java: while (bits - val + (bound-1) < 0) retry  (32-bit signed overflow)
            if bits - val + (bound - 1) < _INT32:
                return val

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) / float(1 << 53)


def java_shuffle(items, seed: int) -> list:
    """Replicate Collections.shuffle(list, new Random(seed)) for an 8-element list."""
    rnd = JavaRandom(seed)
    lst = list(items)
    for i in range(len(lst), 1, -1):
        j = rnd.next_int(i)
        lst[i - 1], lst[j] = lst[j], lst[i - 1]
    return lst


def shuffled_hand_order(deck: list, seed: int) -> list:
    """The Java hand/cycle order after reset: play order is the shuffled list
    (hand = first 4, 'next' = 5th, queue = rest; gpusim's hand+cycle model is an
    exact FIFO equivalent — see gpusim/DIVERGENCES.md #7)."""
    return java_shuffle(deck, seed)
