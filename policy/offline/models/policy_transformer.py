"""
Decision Transformer for Clash Royale
Adapted from KataCR's transformer models for PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input"""
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention"""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """Forward pass with residual connections"""
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class PolicyTransformer(nn.Module):
    """
    Transformer-based policy network for Clash Royale

    Predicts:
        - Which card to play (card selection)
        - Where to play it (position on arena)
    """

    def __init__(
        self,
        num_cards: int = 108,
        num_troops: int = 200,
        d_model: int = 256,
        n_head: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        arena_grid_size: Tuple[int, int] = (32, 18),
        state_dim: int = 6,  # Dimension of state vector (elixir, time, 4 cards)
    ):
        super().__init__()

        self.d_model = d_model
        self.arena_grid_size = arena_grid_size

        # State encoder: project state features to d_model
        self.state_projection = nn.Linear(state_dim, d_model)

        # Embeddings (kept for future use if needed)
        self.card_embed = nn.Embedding(
            num_cards + 1, d_model // 4
        )  # +1 for padding/empty
        self.troop_embed = nn.Embedding(num_troops + 1, d_model // 4)
        self.team_embed = nn.Embedding(2, d_model // 8)  # Ally/Enemy
        self.elixir_embed = nn.Embedding(11, d_model // 8)  # 0-10 elixir
        self.rtg_projection = nn.Linear(1, d_model // 4)  # Return-to-go

        # State encoder (combines all state features)
        self.state_encoder = nn.Linear(d_model, d_model)

        # Action encoder (previous actions)
        self.action_card_embed = nn.Embedding(
            num_cards + 1, d_model // 2
        )  # Card embeddings for actions (includes padding token)
        self.action_pos_embed = nn.Linear(2, d_model // 2)  # Position (x, y)
        self.action_encoder = nn.Linear(d_model, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model, max_seq_len * 3
        )  # *3 for (rtg, state, action)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output heads
        self.card_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_cards),  # Use num_cards parameter
        )

        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                d_model // 2, arena_grid_size[0] * arena_grid_size[1]
            ),  # Flat position logits
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_state(self, state_dict):
        """
        Encode state from dictionary format

        Args:
            state_dict: Dict with 'troops', 'cards', 'elixir'

        Returns:
            state_encoding: (B, d_model)
        """
        # This is a simplified version - you would customize based on your state format
        # For now, just create a dummy encoding
        batch_size = len(state_dict) if isinstance(state_dict, list) else 1
        return torch.zeros(batch_size, self.d_model)

    def forward(
        self,
        states,  # (B, T, state_dim) - State feature vectors
        actions,  # (B, T, 3) - Previous actions [card_id, position_x, position_y]
        rtg,  # (B, T) - Return-to-go values
        timesteps,  # (B, T) - Timestep indices
        attention_mask=None,
    ):
        """
        Forward pass

        Args:
            states: State tensors (B, T, state_dim)
            actions: Previous actions (B, T, 3) - [card_id, position_x, position_y]
            rtg: Return-to-go (B, T)
            timesteps: Timestep indices (B, T)
            attention_mask: Optional attention mask

        Returns:
            card_logits: (B, T, 4) - Card selection logits
            position_logits: (B, T, grid_size) - Position logits
        """
        B, T = rtg.shape

        # Encode states from feature vectors
        state_embeds = self.state_projection(states)  # (B, T, d_model)
        state_embeds = self.state_encoder(state_embeds)  # (B, T, d_model)

        # Encode RTG
        rtg_embeds = self.rtg_projection(rtg.unsqueeze(-1))  # (B, T, d_model//4)
        rtg_embeds = F.pad(
            rtg_embeds, (0, self.d_model - self.d_model // 4)
        )  # Pad to d_model

        # Encode previous actions
        action_card_embeds = self.action_card_embed(
            actions[:, :, 0].long()
        )  # (B, T, d_model//2)
        action_pos_embeds = self.action_pos_embed(
            actions[:, :, 1:].float()
        )  # (B, T, d_model//2)
        action_embeds = torch.cat(
            [action_card_embeds, action_pos_embeds], dim=-1
        )  # (B, T, d_model)
        action_embeds = self.action_encoder(action_embeds)

        # Interleave: (rtg, state, action) for each timestep
        sequence = torch.stack(
            [rtg_embeds, state_embeds, action_embeds], dim=2
        )  # (B, T, 3, d_model)
        sequence = sequence.reshape(B, T * 3, self.d_model)

        # Add positional encoding
        sequence = self.pos_encoding(sequence)

        # Apply transformer blocks
        x = sequence
        for block in self.transformer_blocks:
            x = block(x, mask=attention_mask)

        # Extract state predictions (every 3rd token starting from index 1)
        state_outputs = x[:, 1::3, :]  # (B, T, d_model)

        # Predict actions
        card_logits = self.card_head(state_outputs)  # (B, T, 4)
        position_logits = self.position_head(state_outputs)  # (B, T, grid_size)

        return card_logits, position_logits

    def predict_action(self, state_dict, rtg, timestep):
        """
        Predict single action given current state

        Args:
            state_dict: Current state dictionary
            rtg: Target return-to-go
            timestep: Current timestep

        Returns:
            card_idx: Predicted card index (0-3)
            position: Predicted position (x, y) in grid coordinates
        """
        self.eval()
        with torch.no_grad():
            # Prepare inputs (batch size 1, sequence length 1)
            rtg_tensor = torch.FloatTensor([[rtg]])
            timestep_tensor = torch.LongTensor([[timestep]])
            action_tensor = torch.LongTensor([[[0, 0]]])  # Dummy previous action

            # Forward pass
            card_logits, position_logits = self.forward(
                [state_dict], action_tensor, rtg_tensor, timestep_tensor
            )

            # Sample from distributions
            card_probs = F.softmax(card_logits[0, 0], dim=0)
            card_idx = torch.multinomial(card_probs, 1).item()

            position_probs = F.softmax(position_logits[0, 0], dim=0)
            position_idx = torch.multinomial(position_probs, 1).item()

            # Convert flat position to (x, y)
            rows, cols = self.arena_grid_size
            position_y = position_idx // cols
            position_x = position_idx % cols

            return card_idx, (position_x, position_y)


if __name__ == "__main__":
    # Test model
    model = PolicyTransformer(
        num_cards=108,
        num_troops=200,
        d_model=256,
        n_head=8,
        n_layers=4,
        arena_grid_size=(32, 18),
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    B, T = 4, 16  # Batch size, sequence length
    states = [None] * (B * T)  # Dummy states
    actions = torch.randint(
        0, 5, (B, T, 3)
    )  # Random previous actions: [card_id, pos_x, pos_y]
    rtg = torch.randn(B, T)  # Random return-to-go
    timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)

    card_logits, position_logits = model(states, actions, rtg, timesteps)

    print(f"Card logits shape: {card_logits.shape}")  # Should be (B, T, 4)
    print(f"Position logits shape: {position_logits.shape}")  # Should be (B, T, 32*18)

    # Test single prediction
    dummy_state = {
        "troops": [],
        "cards": ["knight", "archer", "fireball", "goblin"],
        "elixir": 7,
    }
    card_idx, position = model.predict_action(dummy_state, rtg=10.0, timestep=5)
    print(f"Predicted action: card={card_idx}, position={position}")
