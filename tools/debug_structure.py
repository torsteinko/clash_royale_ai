import numpy as np
import glob
import os


def debug_deep_structure(replay_dir):
    files = glob.glob(os.path.join(replay_dir, "*.npy"))
    if not files:
        print("❌ No .npy files found.")
        return

    fpath = files[0]
    print(f"🔍 Deep Inspection: {os.path.basename(fpath)}")

    try:
        data = np.load(fpath, allow_pickle=True)
        if data.shape == ():
            data = data.item()

        # Inspect 'state'
        if "state" in data:
            state_data = data["state"]
            print(f"\n📦 'state' is type: {type(state_data)}")

            # Check if it's a list/array (Sequence of frames)
            if isinstance(state_data, (list, np.ndarray)):
                print(f"  Length: {len(state_data)}")
                if len(state_data) > 0:
                    first_frame = state_data[0]
                    print(f"  First frame type: {type(first_frame)}")
                    if isinstance(first_frame, dict):
                        print(f"  First frame keys: {list(first_frame.keys())}")
                        if "cards" in first_frame:
                            print(f"  First frame 'cards': {first_frame['cards']}")
                        if "unit_infos" in first_frame:
                            print(
                                f"  First frame 'unit_infos' length: {len(first_frame['unit_infos'])}"
                            )
                        if "elixir" in first_frame:
                            print(f"  First frame 'elixir': {first_frame['elixir']}")
            # Check if it's a single dict (Single frame?)
            elif isinstance(state_data, dict):
                print(f"  Keys: {list(state_data.keys())}")
                if "cards" in state_data:
                    print(f"  'cards': {state_data['cards']}")

        # Inspect 'action'
        if "action" in data:
            action_data = data["action"]
            print(f"\n🎮 'action' is type: {type(action_data)}")

            if isinstance(action_data, (list, np.ndarray)):
                print(f"  Length: {len(action_data)}")
                if len(action_data) > 0:
                    first_action = action_data[0]
                    print(f"  First action: {first_action}")
            elif isinstance(action_data, dict):
                print(f"  Action data: {action_data}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    debug_deep_structure("replay_data")
