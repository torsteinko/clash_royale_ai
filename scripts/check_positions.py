import lzma
import numpy as np

data = np.load(
    lzma.open("replay_data/lan77_20240406_episodes_1.npy.xz", "rb"), allow_pickle=True
).item()

actions = [a for a in data["action"] if a["xy"] is not None]

print("Sample actions with positions:")
for a in actions[:10]:
    print(f"  card_id={a['card_id']}, xy={a['xy']}")

print(f"\nTotal actions: {len(data['action'])}")
print(f"Actions with positions: {len(actions)}")
print(f"Actions without positions: {len(data['action']) - len(actions)}")
