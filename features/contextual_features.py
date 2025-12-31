"""
Contextual Features Calculator
Calculates game context features: home/away, rest days, back-to-back
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class ContextualFeaturesCalculator:
    """Calculates contextual features for games"""

    @staticmethod
    def parse_home_away(matchup: str) -> bool:
        """
        Determine if game is home or away from matchup string

        Args:
            matchup: Matchup string (e.g., "LAL vs. GSW" or "LAL @ GSW")

        Returns:
            True if home game, False if away
        """
        if pd.isna(matchup):
            return None
        return 'vs.' in matchup or 'vs' in matchup

    @staticmethod
    def calculate_rest_days(current_game_date: str, prior_games: pd.DataFrame) -> Optional[int]:
        """
        Calculate days since last game

        Args:
            current_game_date: Date of current game
            prior_games: DataFrame of prior games (sorted DESC by date)

        Returns:
            Number of rest days (0 = back-to-back, None if no prior games)
        """
        if prior_games.empty:
            return None

        last_game_date = pd.to_datetime(prior_games.iloc[0]['game_date'])
        current_date = pd.to_datetime(current_game_date)

        rest_days = (current_date - last_game_date).days
        return rest_days

    def calculate_for_game(self, game: pd.Series, prior_games: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all contextual features for a game

        Args:
            game: Series representing current game
            prior_games: DataFrame of prior games

        Returns:
            Dictionary of contextual features
        """
        # Home/Away
        is_home = self.parse_home_away(game['matchup'])

        # Rest days
        rest_days = self.calculate_rest_days(game['game_date'], prior_games)

        # Back-to-back
        is_back_to_back = (rest_days == 1) if rest_days is not None else False

        return {
            'is_home': is_home,
            'rest_days': rest_days,
            'is_back_to_back': is_back_to_back
        }