import sys
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


from policy.offline.models.policy_transformer import PolicyTransformer
from policy.offline.train import TrainConfig  # your TrainConfig class


def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore config
    cfg = TrainConfig()
    cfg_dict = ckpt.get("config", {})
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # IMPORTANT: make sure state_dim matches what you used (126)
    if not hasattr(cfg, "state_dim"):
        cfg.state_dim = 126

    model = PolicyTransformer(
        num_cards=cfg.num_cards,
        num_troops=cfg.num_troops,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.sequence_length,
        dropout=cfg.dropout,
        arena_grid_size=cfg.arena_grid_size,
        state_dim=cfg.state_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, device


@torch.no_grad()
def predict_single(
    model, cfg, device, state_vec, rtg=10.0, timestep=0, temperature=1.0
):
    """
    state_vec: numpy array of shape (state_dim,) = your 126-dim vector
    rtg: desired return-to-go (float)
    timestep: int, usually 0 for single-step
    """
    state = (
        torch.from_numpy(state_vec).float().unsqueeze(0).unsqueeze(0).to(device)
    )  # (1,1,126)
    actions_in = torch.zeros(1, 1, 3, device=device)  # dummy previous action
    rtg_tensor = torch.tensor([[rtg]], dtype=torch.float32, device=device)  # (1,1)
    t_tensor = torch.tensor([[timestep]], dtype=torch.long, device=device)  # (1,1)

    card_logits, pos_logits = model(state, actions_in, rtg_tensor, t_tensor)

    # Temperature sampling
    card_probs = torch.softmax(card_logits[0, 0] / temperature, dim=-1)
    pos_probs = torch.softmax(pos_logits[0, 0] / temperature, dim=-1)

    card_id = torch.multinomial(card_probs, 1).item()
    pos_idx = torch.multinomial(pos_probs, 1).item()

    H, W = cfg.arena_grid_size
    y = pos_idx // W
    x = pos_idx % W

    return {
        "card_id": card_id,
        "pos_idx": pos_idx,
        "pos_xy": (int(x), int(y)),
        "card_conf": float(card_probs[card_id]),
    }


if __name__ == "__main__":
    ckpt_path = "runs/policy_training/20251209_112531/checkpoints/best_model.pt"
    model, cfg, device = load_model(ckpt_path)

    print("Loaded model on", device)
    print(
        "num_cards:", cfg.num_cards, "state_dim:", getattr(cfg, "state_dim", "unknown")
    )

    # Dummy test state: you should replace this with a real 126-dim state from your loader
    state_dim = getattr(cfg, "state_dim", 126)
    dummy_state = np.zeros(state_dim, dtype=np.float32)
    dummy_state[0] = 0.5  # elixir ~5
    dummy_state[1] = 0.3  # time normalized
    dummy_state[2:6] = [1, 2, 3, 4]  # cards in hand

    action = predict_single(
        model, cfg, device, dummy_state, rtg=10.0, timestep=0, temperature=1.0
    )
    print("\nPredicted action:")
    print("  card_id:", action["card_id"])
    print("  pos_xy:", action["pos_xy"])
    print("  card_confidence:", f"{action['card_conf']:.3f}")
