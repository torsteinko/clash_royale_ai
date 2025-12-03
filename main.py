"""Clash Royale Bot - Main Entry Point"""

def main():
    print('='*80)
    print('CLASH ROYALE BOT v0.1')
    print('='*80)
    print()
    print('Options:')
    print('  1. Test detection')
    print('  2. Run rule-based bot')
    print('  3. Train RL agent')
    print('  4. Deploy bot')
    print()
    
    choice = input('Select option: ')
    
    if choice == '1':
        from tests.test_live_detection import main as test_main
        test_main()
    else:
        print('Not implemented yet')

if __name__ == '__main__':
    main()
