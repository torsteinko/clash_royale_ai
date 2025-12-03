"""Track card cycle"""

class DeckTracker:
    def __init__(self, deck):
        self.deck = deck
        self.played_cards = []
    
    def update(self, cards_in_hand):
        """Update based on cards in hand"""
        pass
    
    def get_next_card(self):
        """Predict next card in cycle"""
        return None
