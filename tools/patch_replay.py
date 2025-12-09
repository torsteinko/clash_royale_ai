import numpy as np
import os
import glob
import pickle

# --- CONFIGURATION ---
INPUT_DIR = "replay_data"
OUTPUT_DIR = "replay_data_patched"

# MAPPING: KataCR Local ID (from file) -> YOUR Global ID
deck_mapping = {
    0: 31,  # cannon
    1: 0,  # empty
    2: 73,  # fireball
    3: 84,  # hog-rider
    4: 18,  # ice-golem
    5: 4,  # ice-spirit
    6: 5,  # ice-spirit-evo
    7: 98,  # musketeer
    8: 6,  # skeletons
    9: 7,  # skeletons-evo
    10: 22,  # the-log
}


def patch_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(INPUT_DIR, "*.npy"))

    if not files:
        print("❌ No files found.")
        return

    count = 0
    for fpath in files:
        try:
            # Load
            data = np.load(fpath, allow_pickle=True)
            if data.shape == ():
                data = data.item()

            new_data = {"states": [], "actions": [], "rewards": [], "terminals": []}
            state_map = {}  # Map frame_idx -> [GlobalID_1, GlobalID_2, ...]

            # 1. PROCESS STATES & RESOLVE HAND
            if "state" in data:
                for i, s in enumerate(data["state"]):
                    if "cards" in s:
                        # Map local IDs -> Global IDs
                        raw = s["cards"]
                        mapped = [deck_mapping.get(c, 0) for c in raw]

                        # Fix List Structure: [Next, S1, S2, S3, S4] -> [S1, S2, S3, S4]
                        if len(mapped) == 5:
                            hand = mapped[1:]  # Drop Next Card
                        else:
                            hand = mapped[:4]  # Fallback

                        # Pad to 4
                        while len(hand) < 4:
                            hand.append(0)

                        s["cards"] = hand
                        state_map[i] = hand  # Store for Action lookup

                    new_data["states"].append(s)

            # 2. PROCESS ACTIONS & CONVERT TO GLOBAL ID
            if "action" in data:
                for i, a in enumerate(data["action"]):
                    slot = a.get("card_id", 0)
                    global_id = 0

                    if slot > 0:
                        # Retrieve the hand for this exact frame
                        hand = state_map.get(i, [0, 0, 0, 0])

                        # Convert Slot (1-4) -> Index (0-3)
                        idx = slot - 1
                        if 0 <= idx < len(hand):
                            global_id = hand[idx]
                        else:
                            # If slot is invalid (e.g. 5), ignore it
                            global_id = 0

                    # OVERWRITE with Global ID (e.g. 84 for Hog)
                    a["card_id"] = global_id

                    new_data["actions"].append(a)

            # 3. REWARDS & TERMINALS
            new_data["rewards"] = data.get("reward", [0.0] * len(new_data["states"]))
            new_data["terminals"] = data.get("done", [False] * len(new_data["states"]))
            if new_data["terminals"]:
                new_data["terminals"][-1] = True

            # Save
            fname = os.path.basename(fpath).replace(".npy", ".pkl")
            with open(os.path.join(OUTPUT_DIR, fname), "wb") as f:
                pickle.dump(new_data, f)
            count += 1
            print(f"✅ Patched {fname} (Action ID updated)", end="\r")

        except Exception as e:
            print(f"\n❌ Failed {fpath}: {e}")

    print(f"\n\n🎉 Done! {count} files patched to Global IDs.")


if __name__ == "__main__":
    patch_and_save()
