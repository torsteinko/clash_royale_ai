import numpy as np
import glob
import os


def inspect_ids(replay_dir):
    print(f"📂 Scanning files in: {replay_dir}")
    files = glob.glob(os.path.join(replay_dir, "*.npy"))

    if not files:
        print("❌ No .npy files found!")
        return

    # Sets to store unique IDs
    hand_ids = set()
    action_ids = set()

    total_files = 0

    for fpath in files:
        try:
            # Load the file
            data = np.load(fpath, allow_pickle=True)
            if data.shape == ():
                data = data.item()

            # Check States (Cards in Hand)
            if "states" in data:
                for state in data["states"]:
                    if "cards" in state:
                        # Add all non-zero IDs from the hand
                        for card_id in state["cards"]:
                            if card_id != 0:
                                hand_ids.add(int(card_id))

            # Check Actions (Cards Played)
            if "actions" in data:
                for action in data["actions"]:
                    if "card_id" in action:
                        cid = int(action["card_id"])
                        if cid != 0:
                            action_ids.add(cid)

            total_files += 1
            print(f"  Processed: {os.path.basename(fpath)}", end="\r")

        except Exception as e:
            print(f"\n❌ Error reading {fpath}: {e}")

    print(f"\n\n{'='*40}")
    print(f"📊 SUMMARY (Scanned {total_files} files)")
    print(f"{'='*40}")

    print(f"\n🃏 Unique IDs in HAND (State):")
    print(sorted(list(hand_ids)))

    print(f"\n⚔️ Unique IDs in ACTIONS (Played):")
    print(sorted(list(action_ids)))

    print(f"\n{'='*40}")


if __name__ == "__main__":
    # Change this if your folder name is different
    inspect_ids("replay_data")
