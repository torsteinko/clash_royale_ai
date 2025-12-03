# config/tower_stats.py
"""Tower identification database with accurate stats"""

# Tower stats at each level
TOWER_STATS = {
    'princess': {
        1:  {'hp': 1400, 'damage': 50, 'dps': 62},
        2:  {'hp': 1512, 'damage': 54, 'dps': 67},
        3:  {'hp': 1624, 'damage': 58, 'dps': 72},
        4:  {'hp': 1750, 'damage': 62, 'dps': 77},
        5:  {'hp': 1890, 'damage': 67, 'dps': 83},
        6:  {'hp': 2030, 'damage': 72, 'dps': 90},
        7:  {'hp': 2184, 'damage': 78, 'dps': 97},
        8:  {'hp': 2352, 'damage': 84, 'dps': 105},
        9:  {'hp': 2534, 'damage': 90, 'dps': 112},
        10: {'hp': 2786, 'damage': 99, 'dps': 123},
        11: {'hp': 3052, 'damage': 109, 'dps': 136},
        12: {'hp': 3346, 'damage': 119, 'dps': 148},
        13: {'hp': 3668, 'damage': 131, 'dps': 163},
        14: {'hp': 4032, 'damage': 144, 'dps': 180},
        15: {'hp': 4424, 'damage': 158, 'dps': 197},
    },
    
    'king': {
        1:  {'hp': 2400, 'damage': 50, 'dps': 50},
        2:  {'hp': 2568, 'damage': 54, 'dps': 54},
        3:  {'hp': 2736, 'damage': 58, 'dps': 58},
        4:  {'hp': 2904, 'damage': 62, 'dps': 62},
        5:  {'hp': 3096, 'damage': 67, 'dps': 67},
        6:  {'hp': 3312, 'damage': 72, 'dps': 72},
        7:  {'hp': 3528, 'damage': 78, 'dps': 78},
        8:  {'hp': 3768, 'damage': 84, 'dps': 84},
        9:  {'hp': 4008, 'damage': 90, 'dps': 90},
        10: {'hp': 4392, 'damage': 99, 'dps': 99},
        11: {'hp': 4824, 'damage': 109, 'dps': 109},
        12: {'hp': 5304, 'damage': 119, 'dps': 119},
        13: {'hp': 5832, 'damage': 131, 'dps': 131},
        14: {'hp': 6408, 'damage': 144, 'dps': 144},
        15: {'hp': 7032, 'damage': 158, 'dps': 158},
    },
    
    'cannoneer': {
        6:  {'hp': 1740, 'damage': 189, 'dps': 85},
        7:  {'hp': 1872, 'damage': 210, 'dps': 95},
        8:  {'hp': 2016, 'damage': 233, 'dps': 105},
        9:  {'hp': 2172, 'damage': 259, 'dps': 117},
        10: {'hp': 2388, 'damage': 288, 'dps': 130},
        11: {'hp': 2616, 'damage': 320, 'dps': 145},
        12: {'hp': 2868, 'damage': 352, 'dps': 160},
        13: {'hp': 3144, 'damage': 387, 'dps': 175},
        14: {'hp': 3456, 'damage': 426, 'dps': 193},
        15: {'hp': 3792, 'damage': 469, 'dps': 213},
    },
    
    'dagger_duchess': {
        9:  {'hp': 2298, 'damage': 96, 'dps': 192},
        10: {'hp': 2527, 'damage': 102, 'dps': 204},
        11: {'hp': 2768, 'damage': 107, 'dps': 214},
        12: {'hp': 3035, 'damage': 118, 'dps': 236},
        13: {'hp': 3327, 'damage': 130, 'dps': 236},
        14: {'hp': 3657, 'damage': 142, 'dps': 284},
        15: {'hp': 4013, 'damage': 156, 'dps': 312},
    },
    
    'chef': {
        9:  {'hp': 2244, 'damage': 90, 'dps': 90},
        10: {'hp': 2467, 'damage': 99, 'dps': 99},
        11: {'hp': 2703, 'damage': 109, 'dps': 109},
        12: {'hp': 2963, 'damage': 119, 'dps': 119},
        13: {'hp': 3248, 'damage': 131, 'dps': 131},
        14: {'hp': 3571, 'damage': 144, 'dps': 144},
        15: {'hp': 3918, 'damage': 158, 'dps': 158},
    },
}


def identify_tower(hp: int, level: int, is_king: bool = False) -> str:
    """
    Identify tower type based on FULL HP at match start
    
    Called ONCE when tower is first seen at full HP.
    
    Args:
        hp: Current HP (should be max HP)
        level: Tower level
        is_king: True if king position, False if princess position
    
    Returns:
        'princess', 'king', 'cannoneer', 'dagger_duchess', 'chef', or 'unknown'
    """
    
    # Filter by position
    if is_king:
        candidates = ['king', 'chef']
    else:
        candidates = ['princess', 'cannoneer', 'dagger_duchess']
    
    # Exact match (1% tolerance for OCR errors)
    for tower_type in candidates:
        if level not in TOWER_STATS[tower_type]:
            continue
        
        max_hp = TOWER_STATS[tower_type][level]['hp']
        
        # Check if HP matches (within 1%)
        if abs(hp - max_hp) <= max_hp * 0.01:
            return tower_type
    
    return 'unknown'


def get_max_hp(tower_type: str, level: int) -> int:
    """Get max HP for tower type at level"""
    if tower_type in TOWER_STATS and level in TOWER_STATS[tower_type]:
        return TOWER_STATS[tower_type][level]['hp']
    return 0


def get_tower_damage(tower_type: str, level: int) -> int:
    """Get damage for tower type at level"""
    if tower_type in TOWER_STATS and level in TOWER_STATS[tower_type]:
        return TOWER_STATS[tower_type][level]['damage']
    return 0


# Display names
TOWER_DISPLAY_NAMES = {
    'princess': 'Princess Tower',
    'king': 'King Tower',
    'cannoneer': 'Cannoneer',
    'dagger_duchess': 'Dagger Duchess',
    'chef': 'Royal Chef'
}
