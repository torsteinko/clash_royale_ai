# Offline Reinforcement Learning Training

This directory contains the offline RL training pipeline for the Clash Royale AI, adapted from the KataCR project.

## Overview

The training system uses:
- **Decision Transformer**: Transformer-based policy network that predicts card selection and placement
- **Offline RL**: Train from pre-recorded gameplay (no need for live interaction)
- **YOLO-based State**: Uses your existing detection model for game state extraction

## Quick Start

### 1. Collect Replay Data

#### From Video Recording
```bash
python scripts/collect_replay_data.py --mode video --video recordings/gameplay.mp4 --deck knight archer fireball goblin giant musketeer zap skeleton_army
```

#### From Live Gameplay (5 FPS for 3 minutes)
```bash
python scripts/collect_replay_data.py --mode live --duration 180 --fps 5 --deck knight archer fireball goblin
```

This will save replay files to `replay_data/` directory.

### 2. Train the Policy

```bash
python policy/offline/train.py --replay-dir replay_data --batch-size 16 --epochs 50 --lr 1e-4
```

Training logs and checkpoints will be saved to `runs/policy_training/<timestamp>/`

### 3. Monitor Training

```bash
tensorboard --logdir runs/policy_training
```

Then open http://localhost:6006 in your browser.

## Directory Structure

```
policy/
├── __init__.py
├── builders/
│   ├── __init__.py
│   └── state_builder.py       # State/Action/Reward builders
├── offline/
│   ├── __init__.py
│   ├── dataset.py             # Replay dataset loader
│   ├── train.py               # Training script
│   ├── cnn/
│   │   └── cnn_blocks.py      # CNN feature extractors
│   └── models/
│       └── policy_transformer.py  # Decision Transformer model
└── utils/
    └── __init__.py            # Utility functions

replay_data/                   # Saved replay files (.pkl or .xz)
scripts/
└── collect_replay_data.py     # Replay collection script
```

## Data Format

### Replay Files
Each replay file contains:
- `states`: List of state dicts (troops, cards, elixir, time, towers)
- `actions`: List of action dicts (card_id, xy position)
- `rewards`: numpy array of rewards per frame
- `terminals`: numpy array of episode end markers

### State Dict
```python
{
    'troops': [
        {
            'xy': (x, y),           # Position
            'type': 'knight',       # Troop type
            'team': 0,              # 0=ally, 1=enemy
            'bbox': (x1,y1,x2,y2),
            'confidence': 0.9,
            'level': 11
        },
        ...
    ],
    'cards': ['knight', 'archer', 'fireball', 'goblin'],  # 4 cards in hand
    'elixir': 7,                    # Current elixir (0-10)
    'time': 45,                     # Elapsed seconds
    'towers': {                     # Tower status
        'ally_left': True,
        'ally_right': True,
        'ally_king': True,
        'enemy_left': True,
        'enemy_right': False,       # Destroyed
        'enemy_king': True
    }
}
```

### Action Dict
```python
{
    'card_id': 2,                   # 0=no action, 1-4=card slots
    'xy': (9, 16)                   # Grid position (18x32)
}
```

## Model Architecture

The `PolicyTransformer` uses:
- **Input**: Sequence of (return-to-go, state, previous_action) tuples
- **Transformer**: Multi-head self-attention with 4-8 layers
- **Outputs**:
  - Card selection: Softmax over 4 cards
  - Position: Softmax over 32×18 grid

### Training Objectives
1. **Card Loss**: Cross-entropy for card selection
2. **Position Loss**: Cross-entropy for placement location
3. **Total Loss**: Sum of both losses

## Configuration

Edit `TrainConfig` in `policy/offline/train.py`:

```python
class TrainConfig:
    # Model
    d_model = 256              # Transformer dimension
    n_head = 8                 # Attention heads
    n_layers = 4               # Transformer layers
    sequence_length = 16       # Sequence length for training
    
    # Training
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000
    
    # Data
    replay_dir = 'replay_data'
    num_workers = 4
```

## Reward Structure

The `RewardBuilder` calculates rewards based on:
- **Tower Destroyed**: +1.0 per enemy tower, +3.0 for king tower
- **Tower Lost**: -1.0 per ally tower, -3.0 for king tower
- **Win Bonus**: +5.0 for winning, -5.0 for losing
- **Crown Bonus**: +0.5 per crown won, -0.5 per crown lost

## Advanced Usage

### Custom Reward Function

Edit `policy/builders/state_builder.py`:

```python
class RewardBuilder:
    def calculate_reward(self, game_state: Dict) -> float:
        # Your custom reward logic
        reward = 0.0
        
        # Example: Reward for troop count advantage
        ally_troops = len(game_state['troops']['ally'])
        enemy_troops = len(game_state['troops']['enemy'])
        reward += 0.1 * (ally_troops - enemy_troops)
        
        return reward
```

### Custom State Features

Edit `state_builder.build_state()` to add features:

```python
def build_state(self, game_state: Dict) -> Dict:
    state = {}
    
    # Add your custom features
    state['elixir_advantage'] = state['elixir'] - opponent_elixir
    state['tower_count'] = count_standing_towers(game_state)
    
    return state
```

## Tips for Better Performance

1. **Collect Diverse Data**: Record games with different decks and strategies
2. **Balance Wins/Losses**: Include both winning and losing games
3. **Data Augmentation**: Mirror arena left/right for 2x data
4. **Sequence Length**: Longer sequences (16-32) capture better context
5. **Model Size**: Increase `d_model` and `n_layers` if you have lots of data

## Troubleshooting

### No replay data found
```bash
# Check replay directory exists and contains .pkl or .xz files
ls replay_data/
```

### Out of memory during training
```bash
# Reduce batch size or sequence length
python policy/offline/train.py --batch-size 8 --sequence-length 12
```

### Low accuracy
- Collect more diverse training data
- Increase model capacity (more layers/dimensions)
- Train for more epochs
- Check that your YOLO model is detecting troops accurately

## Next Steps

1. **Collect Data**: Record 10+ games (wins and losses)
2. **Train Model**: Run for 50+ epochs
3. **Evaluate**: Test the policy in actual games
4. **Iterate**: Collect more data from policy rollouts, retrain

## References

Based on the KataCR project:
- Decision Transformer architecture
- Offline RL training pipeline
- State-Action-Reward builders

Adapted for your YOLO-based detection system and simplified to PyTorch.
