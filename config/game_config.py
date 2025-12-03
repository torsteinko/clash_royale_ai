"""Game configuration and constants"""

# Screen resolution
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Card hand positions
CARD_SLOTS = [
    (200, 620, 340, 710),
    (360, 620, 500, 710),
    (520, 620, 660, 710),
    (680, 620, 820, 710)
]

# Elixir region
ELIXIR_REGION = (550, 650, 730, 700)

# Timer region
TIMER_REGION = (1100, 10, 1250, 50)

# Tower regions
TOWER_REGIONS = {
    'ally_left': (120, 480, 220, 510),
    'ally_right': (1060, 480, 1160, 510),
    'ally_king': (540, 550, 740, 580),
    'enemy_left': (120, 80, 220, 110),
    'enemy_right': (1060, 80, 1160, 110),
    'enemy_king': (540, 140, 740, 170)
}
