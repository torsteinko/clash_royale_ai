"""Clash Royale Bot - Main Entry Point"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print('='*80)
    print('CLASH ROYALE BOT v0.1')
    print('='*80)
    print()
    print('Options:')
    print('  1. Test detection (interactive live test)')
    print('  2. Run rule-based bot (not implemented)')
    print('  3. Train RL agent (see policy/offline/train.py)')
    print('  4. Deploy bot (see policy/background_controller.py)')
    print()

    choice = input('Select option: ')

    if choice == '1':
        # Import by file path: the repo's tests/ directory can be shadowed by
        # an unrelated site-packages "tests" package, breaking the
        # `from tests.test_live_detection import main` form (verified Sept
        # 2026: ModuleNotFoundError).
        import importlib.util

        test_path = Path(__file__).parent / 'tests' / 'test_live_detection.py'
        spec = importlib.util.spec_from_file_location('test_live_detection', test_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            print(f'⚠️  Live detection test unavailable here: {exc}')
            print('   (it uses the Windows/MEmu capture stack)')
            print('   On Linux run instead:')
            print('     python scripts/analyze_recording.py <recording_dir>')
            return
        module.main()
    else:
        print('Not implemented yet')


if __name__ == '__main__':
    main()
