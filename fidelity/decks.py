"""Deck table for the M4 fidelity tooling.

The names map to the training box's deck definitions (`~/multi_selfplay_train.py`,
`~/record_replays.py` on clash-training) — the same decks the pool12 replays were
recorded with. Card names are lowercase Java/gpusim keys.
"""

DECKS: dict[str, list[str]] = {
    "hog": ["hogrider", "musketeer", "cannon", "skeletons", "icespirits", "fireball", "log", "zap"],
    "giant": ["giant", "witch", "minions", "musketeer", "fireball", "arrows", "valkyrie", "knight"],
    "logb": ["goblinbarrel", "princess", "rocket", "knight", "goblingang", "log", "infernotower", "icespirits"],
    "xbow": ["xbow", "tesla", "log", "icespirits", "skeletons", "archer", "fireball", "knight"],
    "lava": ["lavahound", "balloon", "megaminion", "minions", "tombstone", "fireball", "zap", "arrows"],
    "yard": ["graveyard", "poison", "babydragon", "tornado", "knight", "skeletons", "arrows", "icewizard"],
    "rg": ["royalgiant", "fisherman", "hunter", "skeletons", "electrospirit", "lightning", "log", "ghost"],
    "golem": ["golem", "babydragon", "darkwitch", "tornado", "lightning", "skeletons", "minions", "valkyrie"],
    "miner": ["miner", "poison", "bats", "skeletons", "speargoblins", "valkyrie", "tesla", "log"],
    "mortar": ["mortar", "goblinbarrel", "knight", "bats", "log", "arrows", "princess", "goblins"],
    "balloon": ["balloon", "freeze", "bats", "skeletons", "valkyrie", "arrows", "tesla", "miner"],
    "pekka": ["pekka", "battleram", "ghost", "electrowizard", "poison", "zap", "skeletons", "minions"],
    "default": ["knight", "archer", "fireball", "arrows", "giant", "musketeer", "minions", "valkyrie"],
}
