# config/game_config.py
"""Game configuration for portrait mode (720x1280)"""

# Screen resolution (PORTRAIT)
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1280

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
