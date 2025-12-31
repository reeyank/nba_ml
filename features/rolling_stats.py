"""
Rolling Statistics Calculator
Calculates rolling averages for player performance metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ROLLING_WINDOWS, MIN_GAMES_FOR_PREDICTION


class RollingStatsCalculator:
    """Calculates rolling averages from game history"""

    def __init__(self, windows: List[int] = None):
        """
        Args:
            windows: List of window sizes (e.g., [5, 10, 15])
        """
        self.windows = windows or ROLLING_WINDOWS

    def calculate_for_game(self, prior_games: pd.DataFrame, window: int) -> Dict[str, float]:
        """
        Calculate rolling averages for a specific window

        Args:
            prior_games: DataFrame of games BEFORE the current game (sorted DESC by date)
            window: Number of games to average over

        Returns:
            Dictionary of rolling average features
        """
        if len(prior_games) < MIN_GAMES_FOR_PREDICTION:
            return None  # Not enough data

        # Take the most recent N games
        recent = prior_games.head(window) if len(prior_games) >= window else prior_games

        # Calculate averages (only for features in our schema)
        features = {
            f'points_avg_{window}': recent['points'].mean(),
            f'rebounds_avg_{window}': recent['rebounds'].mean(),
            f'assists_avg_{window}': recent['assists'].mean(),
            f'minutes_avg_{window}': recent['minutes'].mean(),
        }

        # Only add FG% for window=5 (schema limitation)
        if window == 5:
            features[f'field_goal_pct_avg_{window}'] = recent['field_goal_pct'].mean()
            features[f'three_point_pct_avg_{window}'] = recent['three_point_pct'].mean()
            features[f'free_throw_pct_avg_{window}'] = recent['free_throw_pct'].mean()
        elif window == 10:
            features[f'field_goal_pct_avg_{window}'] = recent['field_goal_pct'].mean()

        return features

    def calculate_all_windows(self, prior_games: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate rolling averages for all configured windows

        Args:
            prior_games: DataFrame of prior games (sorted DESC by date)

        Returns:
            Dictionary with all rolling average features
        """
        if len(prior_games) < MIN_GAMES_FOR_PREDICTION:
            return None

        all_features = {}

        for window in self.windows:
            window_features = self.calculate_for_game(prior_games, window)
            if window_features:
                all_features.update(window_features)

        return all_features

    def calculate_trend(self, prior_games: pd.DataFrame, stat: str, window: int = 5) -> float:
        """
        Calculate trend (slope) for a statistic over recent games

        Positive slope = improving, Negative = declining

        Args:
            prior_games: DataFrame of prior games
            stat: Stat to calculate trend for (e.g., 'points')
            window: Number of games to consider

        Returns:
            Slope of linear regression
        """
        if len(prior_games) < window:
            return 0.0

        recent = prior_games.head(window)
        values = recent[stat].values

        if len(values) < 3:
            return 0.0

        # Fit linear regression
        x = np.arange(len(values))
        try:
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except:
            return 0.0

    def calculate_trend_features(self, prior_games: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend features for key stats"""
        return {
            'points_trend_5': self.calculate_trend(prior_games, 'points', 5),
            'rebounds_trend_5': self.calculate_trend(prior_games, 'rebounds', 5),
            'assists_trend_5': self.calculate_trend(prior_games, 'assists', 5),
            'minutes_trend_5': self.calculate_trend(prior_games, 'minutes', 5),
        }


def test_rolling_stats():
    """Test the rolling stats calculator"""
    # Create sample game data
    games = pd.DataFrame({
        'game_date': pd.date_range('2024-01-01', periods=10, freq='2D'),
        'points': [25, 28, 22, 30, 26, 24, 27, 29, 23, 31],
        'rebounds': [6, 7, 5, 8, 6, 5, 7, 8, 6, 9],
        'assists': [7, 8, 6, 9, 7, 6, 8, 9, 7, 10],
        'steals': [2, 1, 2, 3, 2, 1, 2, 2, 1, 3],
        'blocks': [1, 0, 1, 2, 1, 0, 1, 1, 0, 2],
        'turnovers': [3, 2, 3, 4, 3, 2, 3, 3, 2, 4],
        'minutes': [35, 36, 34, 37, 35, 34, 36, 37, 35, 38],
        'field_goal_pct': [0.50, 0.55, 0.48, 0.58, 0.52, 0.49, 0.54, 0.57, 0.51, 0.60],
        'three_point_pct': [0.40, 0.45, 0.38, 0.48, 0.42, 0.39, 0.44, 0.47, 0.41, 0.50],
        'free_throw_pct': [0.85, 0.90, 0.83, 0.92, 0.87, 0.84, 0.89, 0.91, 0.86, 0.93],
    }).sort_values('game_date', ascending=False)

    calculator = RollingStatsCalculator()

    print("Testing Rolling Stats Calculator\n")
    print("Sample games (most recent first):")
    print(games[['game_date', 'points', 'rebounds', 'assists']].head())

    print("\nCalculating rolling averages...")
    features = calculator.calculate_all_windows(games)

    if features:
        print("\nRolling Averages (5 games):")
        print(f"  Points: {features['points_avg_5']:.2f}")
        print(f"  Rebounds: {features['rebounds_avg_5']:.2f}")
        print(f"  Assists: {features['assists_avg_5']:.2f}")

        print("\nRolling Averages (10 games):")
        print(f"  Points: {features['points_avg_10']:.2f}")
        print(f"  Rebounds: {features['rebounds_avg_10']:.2f}")

    print("\nCalculating trends...")
    trends = calculator.calculate_trend_features(games)
    print(f"  Points trend (5 games): {trends['points_trend_5']:.3f}")
    print(f"  Minutes trend (5 games): {trends['minutes_trend_5']:.3f}")

    print("\n✅ Rolling stats calculator working!")


if __name__ == "__main__":
    test_rolling_stats()
