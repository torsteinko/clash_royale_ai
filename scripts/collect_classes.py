import json
import sys
from pathlib import Path

# This is a temporary script to collect all class names from the sprites dataset

sys.path.insert(0, "F:/clash_royale_ai")
from dataset.sprites_dataset.build_synthetic_dataset import SpriteCollector

c = SpriteCollector(Path("F:/clash_royale_ai/dataset/sprites_dataset"))
c.collect_all()
print(json.dumps(sorted(list(c.class_names.keys())), indent=2))
print("\n#sprites:", len(c.sprites))
