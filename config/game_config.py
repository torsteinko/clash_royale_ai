# config/game_config.py
"""Game configuration for portrait mode (720x1280)"""

# Screen resolution (PORTRAIT)
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1280

# Bottom UI strip: everything whose top edge starts at/below this y in a
# 720x1280 portrait capture is an interface element (hand cards, elixir bar,
# buttons), not gameplay.  Detections overlapping the arena (y1 < this line)
# are kept.  (Sept 2026: state_extractor previously had this check inverted,
# which discarded essentially all real gameplay detections.)
ARENA_BOTTOM_Y = 1060

# Meta / decorative detector classes that never contribute gameplay state.
# Exact names:
JUNK_CLASS_EXACT = {
    "bar",
    "elixir",
    "clock",
    "emote",
    "big-text",
    "background-items",
    "miner_dirt",
    "evolution_symbol",
}
# Substring matches (health bars, static map decorations, ...):
JUNK_CLASS_SUBSTRINGS = (
    "tower_bar",
    "king_tower_bar",
    "bar-level",
    "skeleton_king_bar",
    "dagger-duchess-tower-bar",
    "snowman",
    "goblin_bush",
)


def is_junk_class(class_name: str) -> bool:
    """True for meta/decorative classes that should never enter game state."""
    n = (class_name or "").lower()
    if not n:
        return True
    if n in JUNK_CLASS_EXACT:
        return True
    return any(s in n for s in JUNK_CLASS_SUBSTRINGS)

# Card hand positions (bottom of screen)
CARD_SLOTS = [
    (169, 1069, 283, 1215),   # Card 1
    (305, 1069, 418, 1215),   # Card 2
    (440, 1069, 553, 1215),   # Card 3
    (575, 1069, 688, 1215),   # Card 4
]


# Elixir region (bottom center)
ELIXIR_REGION = (280, 1150, 440, 1200)

# Timer region (top center)
TIMER_REGION = (280, 30, 440, 70)

# Tower regions (portrait layout)
TOWER_REGIONS = {
    'ally_left': (50, 800, 150, 830),
    'ally_right': (570, 800, 670, 830),
    'ally_king': (260, 900, 460, 930),
    'enemy_left': (50, 200, 150, 230),
    'enemy_right': (570, 200, 670, 230),
    'enemy_king': (260, 100, 460, 130)
}
