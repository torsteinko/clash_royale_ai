"""Inspect KataCR replay file format"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lzma
import numpy as np
from pathlib import Path


def inspect_replay(file_path):
    """Inspect a single replay file"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*80}")

    try:
        with lzma.open(file_path, "rb") as f:
            data = np.load(f, allow_pickle=True)

        print(f"Type: {type(data)}")
        print(f"Dtype: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # Try to get dict
        if hasattr(data, "item"):
            try:
                data_dict = data.item()
                if isinstance(data_dict, dict):
                    print(f"\nKeys: {list(data_dict.keys())}")

                    for key in data_dict.keys():
                        val = data_dict[key]
                        if isinstance(val, (list, np.ndarray)):
                            print(f"  {key}: length={len(val)}, type={type(val)}")
                            if len(val) > 0:
                                print(f"    First item type: {type(val[0])}")
                                if isinstance(val[0], dict):
                                    print(f"    First item keys: {list(val[0].keys())}")
                        else:
                            print(f"  {key}: {type(val)}")

                    return data_dict
            except:
                pass

        print(f"\nRaw data: {data}")
        return data

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    replay_dir = Path("replay_data")
    files = list(replay_dir.glob("*.xz"))

    if not files:
        print("No .xz files found in replay_data/")
    else:
        # Inspect first file
        inspect_replay(files[0])

        print(f"\n\nFound {len(files)} total replay files")
        print("To inspect a specific file:")
        print(f"  python scripts/inspect_replay.py path/to/file.xz")
