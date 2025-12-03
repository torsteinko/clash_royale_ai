"""Elixir timing and prediction"""

class ElixirManager:
    def get_elixir_rate(self, match_time):
        """Calculate elixir generation rate"""
        if match_time is None:
            return 1.0
        
        if match_time > 60:
            return 1.0  # Normal
        elif match_time > 0:
            return 2.0  # Double elixir
        else:
            return 3.0  # Triple (overtime)
