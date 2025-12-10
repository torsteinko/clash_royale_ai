from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from ultralytics import YOLO
from PIL import Image
import cv2
from torchvision import models, transforms
import torch.nn as nn

# Add project root to sys.path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from detection.ocr_reader import OCRReader
from policy.offline.models.policy_transformer import PolicyTransformer
from policy.offline.train import TrainConfig
from policy.offline.card_list import card2idx, idx2card
from policy.offline.label_list import unit2idx, idx2unit

# Load YOLO class names
with open(ROOT / "data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)

YOLO_CLASS_NAMES = data_yaml["names"]
YOLO_CONF_THRESHOLD = 0.5


# --- Card Detector Class ---
class CardClassifier:
    def __init__(self, model_path="models/cards_cls.pt", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]

        # Build model
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Transform (same as training)
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, img_pil, valid_cards=None):
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_t)

            # --- DECK FILTERING ---
            if valid_cards:
                mask = torch.full_like(logits, float("-inf"))
                found_any = False
                for card_name in valid_cards:
                    if card_name in self.classes:
                        idx = self.classes.index(card_name)
                        mask[0, idx] = logits[0, idx]
                        found_any = True
                if found_any:
                    logits = mask
            # ----------------------

            pred_idx = torch.argmax(logits, dim=1).item()
        return self.classes[pred_idx]

    def detect_hand(self, img_cv2, valid_deck=None, save_crops=True):
        print("detect hand ran")
        """
        Detect 4 cards in hand.
        Uses precise coordinates focused on card faces (87x81).
        """
        h, w = img_cv2.shape[:2]

        if save_crops:
            debug_dir = Path("debug_hand")
            debug_dir.mkdir(exist_ok=True)

        # --- PRECISION CROP CONFIG (From your working snippet) ---
        W_PERC = 87 / 720
        H_PERC = 81 / 1280

        # Center Y Logic
        CY_PERC = (1069 + 20 + 40.5) / 1280

        # Center X Logic
        CX_1 = 226 / 720
        CX_2 = (226 + 136) / 720
        CX_3 = (226 + 136 * 2) / 720
        CX_4 = (226 + 136 * 3) / 720

        card_positions = [
            (CX_1, CY_PERC, W_PERC, H_PERC),
            (CX_2, CY_PERC, W_PERC, H_PERC),
            (CX_3, CY_PERC, W_PERC, H_PERC),
            (CX_4, CY_PERC, W_PERC, H_PERC),
        ]

        detected_cards = []
        for i, (cx, cy, cw, ch) in enumerate(card_positions):
            x1 = int((cx - cw / 2) * w)
            y1 = int((cy - ch / 2) * h)
            x2 = int((cx + cw / 2) * w)
            y2 = int((cy + ch / 2) * h)

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            card_crop = img_cv2[y1:y2, x1:x2]

            if card_crop.size == 0:
                detected_cards.append("unknown")
                continue

            # Convert to PIL RGB
            card_pil = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))

            # Predict (with deck filter)
            card_name = self.predict(card_pil, valid_cards=valid_deck)
            detected_cards.append(card_name)

            # Save Debug
            if save_crops:
                timestamp = int(time.time() * 1000)
                debug_crop = card_crop.copy()
                cv2.putText(
                    debug_crop,
                    card_name,
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
                fname = debug_dir / f"slot{i}_{card_name}_{timestamp}.jpg"
                cv2.imwrite(str(fname), debug_crop)

        return detected_cards

    def predict(self, img_pil, valid_cards=None):
        """
        Predict card from PIL image.
        Args:
            img_pil: PIL Image of the crop.
            valid_cards: List of allowed card names. If provided, prediction is restricted to these.
        """
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_t)

            # --- DECK FILTERING ---
            if valid_cards:
                # Create a mask of -inf
                mask = torch.full_like(logits, float("-inf"))

                found_any = False
                for card_name in valid_cards:
                    if card_name in self.classes:
                        idx = self.classes.index(card_name)
                        mask[0, idx] = logits[0, idx]
                        found_any = True

                # Only apply mask if we actually found corresponding IDs
                if found_any:
                    logits = mask
            # ----------------------

            pred_idx = torch.argmax(logits, dim=1).item()
        return self.classes[pred_idx]

    def detect_hand(self, img_cv2, valid_deck=None):
        """
        Detect 4 cards in hand from a screenshot (BGR format).
        Args:
            img_cv2: BGR image.
            valid_deck: List of allowable card names (e.g. your deck + evos).
        """
        h, w = img_cv2.shape[:2]
        debug_dir = Path("debug_crops")
        debug_dir.mkdir(exist_ok=True)

        # --- PRECISION CROP CONFIG ---
        W_PERC = 87 / 720
        H_PERC = 81 / 1280
        CY_PERC = (1069 + 4 + 40.5) / 1280

        CX_1 = 226 / 720
        CX_2 = (226 + 136) / 720
        CX_3 = (226 + 136 * 2) / 720
        CX_4 = (226 + 136 * 3) / 720

        card_positions = [
            (CX_1, CY_PERC, W_PERC, H_PERC),
            (CX_2, CY_PERC, W_PERC, H_PERC),
            (CX_3, CY_PERC, W_PERC, H_PERC),
            (CX_4, CY_PERC, W_PERC, H_PERC),
        ]

        detected_cards = []
        for i, (cx, cy, cw, ch) in enumerate(card_positions):
            x1 = int((cx - cw / 2) * w)
            y1 = int((cy - ch / 2) * h)
            x2 = int((cx + cw / 2) * w)
            y2 = int((cy + ch / 2) * h)

            # Crop card region
            card_crop = img_cv2[y1:y2, x1:x2]

            if card_crop.size == 0:
                detected_cards.append("unknown")
                continue

            # Convert to PIL RGB
            card_pil = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))

            # Predict (Pass valid_deck here)
            card_name = self.predict(card_pil, valid_cards=valid_deck)

            # print(f"Slot {i}: Predicted {card_name}")
            detected_cards.append(card_name)

        return detected_cards


# --- Helper Functions ---
def normalize_yolo_name(yolo_name):
    name = yolo_name.replace("ally_", "").replace("enemy_", "")
    name = name.replace("_", "-")
    return name


def load_models():
    # 1. YOLO
    yolo_path = "runs/synthetic/train_20251207_183426_single/weights/best.pt"
    if not Path(yolo_path).exists():
        raise FileNotFoundError(f"YOLO weights not found at {yolo_path}")
    yolo_model = YOLO(yolo_path)

    # 2. Policy
    policy_checkpoint = "runs/policy_training/20251210_005420/checkpoints/best_model.pt"
    if not Path(policy_checkpoint).exists():
        raise FileNotFoundError(f"Policy checkpoint not found at {policy_checkpoint}")
    checkpoint = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)

    config = TrainConfig()
    for k, v in checkpoint.get("config", {}).items():
        if hasattr(config, k):
            setattr(config, k, v)

    policy_model = PolicyTransformer(
        num_cards=config.num_cards,
        num_troops=config.num_troops,
        d_model=config.d_model,
        n_head=config.n_head,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.sequence_length,
        dropout=config.dropout,
        arena_grid_size=config.arena_grid_size,
        state_dim=config.state_dim,
    )
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.eval()

    # 3. Card Classifier
    card_model_path = ROOT / "models" / "cards_cls.pt"
    if not card_model_path.exists():
        print(
            f"⚠️ Card classifier not found at {card_model_path}. Using manual fallback."
        )
        card_classifier = None
    else:
        print(f"📦 Loading Card Classifier from {card_model_path}")
        card_classifier = CardClassifier(str(card_model_path))

    # 4. OCR Reader
    ocr_reader = OCRReader(use_gpu=torch.cuda.is_available())
    print(f"📦 OCR Reader initialized. Using gpu = {torch.cuda.is_available()}")

    return yolo_model, policy_model, config, card_classifier, ocr_reader


def build_state(detections, hand_cards, config, elixir=5, time_sec=None):
    """Build 126-dim state vector."""
    state = np.zeros(config.state_dim, dtype=np.float32)

    # 1. Elixir (Normalized 0-10)
    # Ensure it's within bounds
    state[0] = max(0.0, min(10.0, float(elixir)))

    # 2. Time (Normalized 0-1 or Seconds)
    # Usually standard games are 180s (3 min) + overtime.
    # If the model was trained on normalized time [0, 1], we should normalize.
    # If time_sec is None (not detected), default to 0.5 (mid-game).
    if time_sec is not None:
        # Normalize assuming standard 3-min game logic, or just pass seconds if training used seconds.
        # Based on previous context, let's normalize to [0, 1] range for 3 minutes.
        # If overtime, it might go > 1.0, which is fine for neural nets.
        state[1] = time_sec / 180.0
    else:
        state[1] = 0.5  # Default fallback

    # --- NAME MAPPING (Classifier -> Policy) ---
    NAME_MAP = {
        "skeleton": "skeletons",
        "skeleton-evo": "skeletons-evolution",
        "skeletons-evo": "skeletons-evolution",
        "bat": "bats",
        "bat-evo": "bats-evolution",
        "bats-evo": "bats-evolution",
        "goblin": "goblins",
        "spear-goblin": "spear-goblins",
        "minion": "minions",
        "archer": "archers",
        "wall-breaker": "wall-breakers",
        "musketeer-evo": "musketeer",
    }

    # --- HAND ---
    cards_found = []
    # (Print statements optional for production speed)
    # print("\n🎴 --- DETECTED HAND ---")

    for card_name in hand_cards:
        norm_name = card_name.lower().replace("_", "-")

        # 1. Apply Explicit Map
        if norm_name in NAME_MAP:
            norm_name = NAME_MAP[norm_name]

        # 2. Generic Logic
        elif norm_name.endswith("-evo"):
            base = norm_name.replace("-evo", "")
            if f"{base}-evolution" in card2idx:
                norm_name = f"{base}-evolution"
            elif f"{base}s-evolution" in card2idx:
                norm_name = f"{base}s-evolution"
            elif f"{base[:-1]}-evolution" in card2idx:
                norm_name = f"{base[:-1]}-evolution"
            elif base in card2idx:
                norm_name = base
            elif f"{base}s" in card2idx:
                norm_name = f"{base}s"

        # 3. Handle Empty/Waiting
        if "waiting" in norm_name or "empty" in norm_name:
            continue

        # 4. Final Check
        if norm_name in card2idx:
            cards_found.append(card2idx[norm_name])
        # else:
        # print(f"   ⚠️ Unknown card: {card_name}")

    # print(f"👉 HAND INPUT: {[idx2card.get(c, f'ID{c}') for c in cards_found]}")

    for i in range(4):
        state[2 + i] = cards_found[i] if i < len(cards_found) else 0

    # --- UNITS ---
    ally_units, enemy_units = [], []
    hand_threshold_y = 0.75

    for det in detections:
        label = YOLO_CLASS_NAMES[int(det["class"])]
        x_norm = ((det["bbox"][0] + det["bbox"][2]) / 2) / det["img_width"]
        y_norm = ((det["bbox"][1] + det["bbox"][3]) / 2) / det["img_height"]

        if y_norm <= hand_threshold_y:
            unit_name = normalize_yolo_name(label)
            if unit_name in unit2idx:
                unit_id = unit2idx[unit_name]
                is_enemy = "enemy" in label
                unit_info = [unit_id, x_norm, y_norm, 1.0]
                if is_enemy:
                    enemy_units.append(unit_info)
                else:
                    ally_units.append(unit_info)

    MAX = 15
    for i in range(MAX):
        base = 6 + i * 4
        state[base : base + 4] = ally_units[i] if i < len(ally_units) else 0
    for i in range(MAX):
        base = 6 + MAX * 4 + i * 4
        state[base : base + 4] = enemy_units[i] if i < len(enemy_units) else 0

    return state


def visualize(image_path, pos, card_id, detections, save_path="prediction.png"):
    img = Image.open(image_path)
    w, h = img.size

    ARENA_X_MIN, ARENA_X_MAX = w * 0.069, w * 0.920
    ARENA_Y_MIN, ARENA_Y_MAX = h * 0.085, h * 0.766
    GRID_W, GRID_H = 18, 32

    grid_x, grid_y = pos
    img_x = ARENA_X_MIN + (grid_x / GRID_W) * (ARENA_X_MAX - ARENA_X_MIN)
    img_y = ARENA_Y_MIN + (grid_y / GRID_H) * (ARENA_Y_MAX - ARENA_Y_MIN)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.xlim(0, w)
    plt.ylim(h, 0)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = YOLO_CLASS_NAMES[det["class"]]
        conf = det["conf"]
        color = "red" if "enemy" in label else "cyan"
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            zorder=2,
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1,
            y1 - 5,
            f"{label} {conf:.2f}",
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5),
            zorder=3,
        )

    plt.scatter(
        img_x,
        img_y,
        c="lime",
        s=300,
        marker="X",
        linewidths=3,
        edgecolors="black",
        zorder=10,
    )
    card_name = idx2card.get(card_id, f"ID {card_id}")
    plt.text(
        img_x,
        img_y - 40,
        f"Play: {card_name}",
        color="lime",
        fontsize=14,
        fontweight="bold",
        ha="center",
        bbox=dict(facecolor="black", alpha=0.7),
        zorder=10,
    )

    rect = plt.Rectangle(
        (ARENA_X_MIN, ARENA_Y_MIN),
        ARENA_X_MAX - ARENA_X_MIN,
        ARENA_Y_MAX - ARENA_Y_MIN,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none",
        linestyle="--",
        zorder=5,
    )
    plt.gca().add_patch(rect)

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"✅ Saved to {save_path}")
    plt.close()


def main(image_path):
    yolo, policy, config, card_classifier, ocr_reader = load_models()
    img_pil = Image.open(image_path)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # --- OCR (Elixir) ---
    elixir = ocr_reader.read_elixir(img_cv2)
    if elixir is None:
        print("⚠️ OCR failed to read elixir. Assuming 5.")
        elixir = 5
    print(f"💧 Elixir: {elixir}")

    # Test timer detection
    timer = ocr_reader.read_timer(img_cv2)
    if timer is not None:
        print(f"⏱️ Timer detected: {timer}")
    else:
        print("⚠️ OCR failed to read timer.")

    # --- YOLO Detection ---
    results = yolo(img_pil, conf=YOLO_CONF_THRESHOLD)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append(
                {
                    "class": int(box.cls),
                    "bbox": box.xyxy[0].tolist(),
                    "conf": float(box.conf),
                    "img_width": img_pil.width,
                    "img_height": img_pil.height,
                }
            )
    print(f"Found {len(detections)} objects (conf > {YOLO_CONF_THRESHOLD}).")

    if card_classifier:
        hand_cards = card_classifier.detect_hand(img_cv2)
    else:
        hand_cards = ["hog-rider", "musketeer", "cannon", "ice-spirit"]
        print("⚠️ Using hardcoded hand.")

    state = build_state(detections, hand_cards, config, elixir=elixir)
    state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        c_logits, p_logits = policy(
            state_t, torch.zeros(1, 1, 3), torch.tensor([[10.0]]), torch.tensor([[0]])
        )

    # Read timer
    try:
        big_text_id = YOLO_CLASS_NAMES.index("big-text")
    except ValueError:
        big_text_id = -1

    if big_text_id != -1:
        print(
            f"Big Text Card Logit: {c_logits[0,0,big_text_id]:.4f} (Idx {big_text_id})"
        )
    else:
        print("Big Text card not in card2idx mapping.")

    for idx, name in idx2card.items():
        if name == "big-text":
            print("^^^ Big Text Card ^^^")
            big_text_id = idx
            break

    big_text_boxes = [d for d in detections if d["class"] == big_text_id]

    # read timer
    timer = ocr_reader.read_timer(img_cv2, big_text_boxes)
    if timer is None:
        print("⚠️ OCR failed to read timer.")
    else:
        print(f"⏱️ Timer detected: {timer}")

    # --- ACTION MASKING ---
    # Retrieve the Card IDs in hand from the state vector
    # State indices [2, 3, 4, 5] correspond to the 4 cards in hand
    hand_ids = state[2:6].astype(int).tolist()

    # Create a mask of -inf
    masked_logits = torch.full_like(c_logits[0, 0], float("-inf"))

    # Allow only cards in hand (and maybe '0' if no card is playable, but usually we want a play)
    valid_move_exists = False
    for card_id in hand_ids:
        if card_id != 0:  # 0 is "Empty Slot"
            masked_logits[card_id] = c_logits[0, 0, card_id]
            valid_move_exists = True

    if valid_move_exists:
        # Pick best VALID card
        card_id = torch.argmax(masked_logits).item()
    else:
        # Fallback if hand is empty (shouldn't happen with full hand)
        print("⚠️ Hand empty/invalid, picking raw max")
        card_id = torch.argmax(c_logits[0, 0]).item()

    pos_idx = torch.argmax(p_logits[0, 0]).item()
    grid_x, grid_y = (
        pos_idx % config.arena_grid_size[1],
        pos_idx // config.arena_grid_size[1],
    )

    print(
        f"\n🤖 Bot plays: {idx2card.get(card_id, 'Unknown')} at ({grid_x}, {grid_y})\n"
    )
    visualize(image_path, (grid_x, grid_y), card_id, detections)


if __name__ == "__main__":
    screenshot = Path(__file__).parent / "image.jpg"
    if screenshot.exists():
        main(screenshot)
    else:
        print(f"Screenshot not found: {screenshot}")
