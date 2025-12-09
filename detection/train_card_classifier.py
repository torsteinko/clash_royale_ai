import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
import random

# --- CONFIG ---
DATA_DIR = Path("data/card_templates")
MODEL_SAVE_PATH = Path("models/cards_cls.pt")
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100  # 100-250 is fine

MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


class ElixirTimerAugment:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        mode = random.choice(["dark", "sweep"])
        gray_img = ImageOps.grayscale(img).convert("RGB")
        dark_img = ImageEnhance.Brightness(gray_img).enhance(0.5)
        if mode == "dark":
            return dark_img
        elif mode == "sweep":
            angle = random.uniform(0, 360)
            w, h = img.size
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            bbox = [-w, -h, 2 * w, 2 * h]
            draw.pieslice(bbox, start=-90, end=-90 + angle, fill=255)
            result = Image.composite(img, dark_img, mask)
            return result
        return img


class CardDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_paths = []
        self.labels = []

        # 1. Base (Applied to ALL) - Geometric Fixes
        self.base_transforms = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # +/- 10% shift
                    scale=(0.85, 1.15),  # +/- 15% scale
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

        # 2. Card Specifics (Gray/Timer) - Applied to Cards ONLY
        self.card_effects = transforms.Compose(
            [
                transforms.RandomGrayscale(p=0.5),  # Force shape learning
                ElixirTimerAugment(p=0.5),
            ]
        )

        # 3. Finalize
        self.finalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        subdirs = ["base", "evolution", "hero", "other"]
        all_files = []
        for sub in subdirs:
            p = self.root_dir / sub
            if p.exists():
                all_files.extend(list(p.glob("*.png")))

        self.classes = sorted(
            list(set([f.stem.lower().replace(" ", "-") for f in all_files]))
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for f in all_files:
            cls_name = f.stem.lower().replace(" ", "-")
            self.image_paths.append(f)
            self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        cls_name = self.classes[label]

        # Pipeline
        image = self.base_transforms(image)

        # Only augment visual style if it's a real card
        if "waiting" not in cls_name and "empty" not in cls_name:
            image = self.card_effects(image)

        image = self.finalize(image)
        return image, label


def train_model():
    print(f"🚀 Training Card Classifier on {DATA_DIR}...")

    # Initialize Dataset (No external transform passed!)
    dataset = CardDataset(DATA_DIR)

    if len(dataset) == 0:
        print("❌ No images found!")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(dataloader):.4f} | Acc: {100 * correct / total:.2f}%"
        )

    print(f"💾 Saving to {MODEL_SAVE_PATH}")
    torch.save(
        {"model_state_dict": model.state_dict(), "classes": dataset.classes},
        MODEL_SAVE_PATH,
    )
    print("✅ Done!")


if __name__ == "__main__":
    train_model()
