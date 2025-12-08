import lzma
import numpy as np

data = np.load(
    lzma.open("replay_data/lan77_20240406_episodes_1.npy.xz", "rb"), allow_pickle=True
).item()
actions = data["action"]

print("First 10 actions:")
for i, a in enumerate(actions[:10]):
    print(f'{i}: card_id={a["card_id"]}, xy={a["xy"]}')

print(
    f'\nCard ID range: min={min(a["card_id"] for a in actions)}, max={max(a["card_id"] for a in actions)}'
)
