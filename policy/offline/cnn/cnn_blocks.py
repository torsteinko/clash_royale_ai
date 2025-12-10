"""CNN backbone models for feature extraction (PyTorch version)"""

import torch
import torch.nn as nn


class CNNBlockConfig:
    """Configuration for CNN blocks"""

    def __init__(self, filters=[32, 64, 64], kernels=[8, 4, 3], strides=[4, 2, 1]):
        self.filters = filters
        self.kernels = kernels
        self.strides = strides


class CNNBlock(nn.Module):
    """Simple CNN feature extractor"""

    def __init__(self, cfg: CNNBlockConfig, in_channels=3):
        super().__init__()
        self.cfg = cfg
        layers = []

        in_ch = in_channels
        for f, k, s in zip(cfg.filters, cfg.kernels, cfg.strides):
            layers.append(nn.Conv2d(in_ch, f, kernel_size=k, stride=s, padding=k // 2))
            layers.append(nn.ReLU(inplace=True))
            in_ch = f

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x)


class ArenaEncoder(nn.Module):
    """Encode arena features from game state"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        # CNN for arena image features (if using visual input)
        self.arena_cnn = CNNBlock(
            CNNBlockConfig(
                filters=[64, 128, 128], kernels=[6, 3, 3], strides=[2, 2, 2]
            ),
            in_channels=3,
        )

        # Spatial positional encoding for troops
        self.troop_embed = nn.Embedding(200, 64)  # Max 200 troop classes
        self.team_embed = nn.Embedding(2, 16)  # Ally/Enemy

        self.hidden_dim = hidden_dim

    def forward(
        self, arena_image=None, troop_positions=None, troop_types=None, troop_teams=None
    ):
        """
        Args:
            arena_image: (B, 3, H, W) - Optional visual arena representation
            troop_positions: (B, N, 2) - Troop positions (x, y)
            troop_types: (B, N) - Troop class indices
            troop_teams: (B, N) - Team (0=ally, 1=enemy)

        Returns:
            features: (B, D) - Encoded arena features
        """
        features = []

        if arena_image is not None:
            visual_features = self.arena_cnn(arena_image)
            visual_features = visual_features.mean(dim=[2, 3])  # Global average pooling
            features.append(visual_features)

        if troop_types is not None and troop_teams is not None:
            troop_embeds = self.troop_embed(troop_types)
            team_embeds = self.team_embed(troop_teams)
            troop_features = torch.cat([troop_embeds, team_embeds], dim=-1)
            troop_features = troop_features.mean(dim=1)  # Average over troops
            features.append(troop_features)

        return torch.cat(features, dim=-1) if features else None


class CardEncoder(nn.Module):
    """Encode hand cards"""

    def __init__(self, num_cards=200, embed_dim=64):
        super().__init__()
        self.card_embed = nn.Embedding(num_cards + 1, embed_dim)  # +1 for padding
        self.elixir_embed = nn.Embedding(11, 16)  # 0-10 elixir

    def forward(self, card_ids, elixir):
        """
        Args:
            card_ids: (B, 4) - Cards in hand
            elixir: (B,) - Current elixir

        Returns:
            features: (B, D) - Encoded card features
        """
        card_features = self.card_embed(card_ids)  # (B, 4, embed_dim)
        card_features = card_features.flatten(1)  # (B, 4*embed_dim)

        elixir_features = self.elixir_embed(elixir)  # (B, 16)

        return torch.cat([card_features, elixir_features], dim=-1)


if __name__ == "__main__":
    # Test CNN block
    cfg = CNNBlockConfig(filters=[16, 32, 32], kernels=[6, 3, 3], strides=[2, 2, 2])
    model = CNNBlock(cfg, in_channels=3)
    x = torch.randn(8, 3, 64, 64)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test arena encoder
    arena_enc = ArenaEncoder(hidden_dim=128)
    arena_img = torch.randn(8, 3, 256, 256)
    troop_pos = torch.randn(8, 10, 2)
    troop_types = torch.randint(0, 100, (8, 10))
    troop_teams = torch.randint(0, 2, (8, 10))
    features = arena_enc(arena_img, troop_pos, troop_types, troop_teams)
    print(f"Arena features shape: {features.shape}")

    # Test card encoder
    card_enc = CardEncoder(num_cards=108, embed_dim=64)
    card_ids = torch.randint(0, 108, (8, 4))
    elixir = torch.randint(0, 11, (8,))
    card_features = card_enc(card_ids, elixir)
    print(f"Card features shape: {card_features.shape}")
