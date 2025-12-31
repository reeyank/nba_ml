"""
Feature Pipeline - Main orchestrator for feature engineering
Processes all games and builds the complete feature store
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_manager import DatabaseManager
from features.rolling_stats import RollingStatsCalculator
from features.contextual_features import ContextualFeaturesCalculator
from config import SEASONS, MIN_GAMES_FOR_PREDICTION


class FeaturePipeline:
    """Main feature engineering pipeline"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.rolling_calculator = RollingStatsCalculator()
        self.contextual_calculator = ContextualFeaturesCalculator()

    def get_opponent_team_id(self, game: pd.Series, player_team_id: int) -> int:
        """
        Extract opponent team ID from game

        For now, we'll need to parse from matchup string or use team_game_stats
        This is a simplified version - in production you'd have a proper mapping
        """
        # TODO: Implement proper opponent extraction
        # For now, return None - we'll enhance this later
        return None

    def get_team_features(self, team_id: int, season: str) -> Dict[str, float]:
        """Get team-level features (pace, ratings)"""
        team_stats = self.db.get_team_stats_for_game(team_id, season)

        if team_stats:
            return {
                'team_pace': team_stats.get('pace'),
                'team_offensive_rating': team_stats.get('offensive_rating'),
                'team_defensive_rating': team_stats.get('defensive_rating')
            }

        return {
            'team_pace': None,
            'team_offensive_rating': None,
            'team_defensive_rating': None
        }

    def get_opponent_features(self, opponent_team_id: int, season: str, game_date: str) -> Dict[str, float]:
        """Get opponent defensive features"""
        if opponent_team_id is None:
            return {
                'opponent_defensive_rating': None,
                'opponent_pace': None
            }

        # Get opponent's defensive rating
        def_rating = self.db.get_opponent_defensive_rating(opponent_team_id, season, game_date)

        # Get opponent's pace
        opp_team_stats = self.db.get_team_stats_for_game(opponent_team_id, season)
        opp_pace = opp_team_stats.get('pace') if opp_team_stats else None

        return {
            'opponent_defensive_rating': def_rating,
            'opponent_pace': opp_pace
        }

    def build_features_for_game(self, game: pd.Series, prior_games: pd.DataFrame, player_team_id: int = None) -> Dict:
        """
        Build complete feature vector for a single game

        Args:
            game: Series representing current game
            prior_games: DataFrame of games before this one (sorted DESC by date)
            player_team_id: Player's team ID (optional)

        Returns:
            Dictionary of all features, or None if insufficient data
        """
        # Check minimum games requirement
        if len(prior_games) < MIN_GAMES_FOR_PREDICTION:
            return None

        features = {}

        # 1. Rolling statistics
        rolling_features = self.rolling_calculator.calculate_all_windows(prior_games)
        if not rolling_features:
            return None
        features.update(rolling_features)

        # 2. Trend features
        trend_features = self.rolling_calculator.calculate_trend_features(prior_games)
        features.update(trend_features)

        # 3. Contextual features
        contextual_features = self.contextual_calculator.calculate_for_game(game, prior_games)
        features.update(contextual_features)

        # 4. Team features (if we have team ID)
        if player_team_id:
            team_features = self.get_team_features(player_team_id, game['season'])
            features.update(team_features)

        # 5. Opponent features
        opponent_team_id = self.get_opponent_team_id(game, player_team_id)
        features['opponent_team_id'] = opponent_team_id
        opponent_features = self.get_opponent_features(opponent_team_id, game['season'], game['game_date'])
        features.update(opponent_features)

        # 6. Metadata
        features['player_id'] = game['player_id']
        features['game_id'] = game['game_id']
        features['game_date'] = game['game_date']
        features['season'] = game['season']

        # 7. Target variables (what we're trying to predict)
        features['target_points'] = game['points']
        features['target_rebounds'] = game['rebounds']
        features['target_assists'] = game['assists']

        return features

    def process_player_games(self, player_id: int, games: pd.DataFrame) -> List[Dict]:
        """
        Process all games for a single player

        Args:
            player_id: Player ID
            games: DataFrame of all games for this player (sorted ASC by date)

        Returns:
            List of feature dictionaries
        """
        features_list = []

        # Sort by date ascending
        games = games.sort_values('game_date', ascending=True).reset_index(drop=True)

        # Process each game
        for idx in range(len(games)):
            current_game = games.iloc[idx]

            # Get prior games (all games before this one)
            prior_games = games.iloc[:idx] if idx > 0 else pd.DataFrame()

            # Need to reverse for rolling calculations (most recent first)
            if not prior_games.empty:
                prior_games = prior_games.sort_values('game_date', ascending=False)

            # Build features
            game_features = self.build_features_for_game(current_game, prior_games)

            if game_features:
                features_list.append(game_features)

        return features_list

    def build_all_features(self, seasons: List[str] = None, sample_size: int = None) -> int:
        """
        Build features for all games across all players

        Args:
            seasons: List of seasons to process (defaults to config)
            sample_size: If set, only process this many players (for testing)

        Returns:
            Number of feature records created
        """
        if seasons is None:
            seasons = SEASONS

        print("="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)

        # Get all game logs
        print(f"\n1. Loading game logs for seasons: {seasons}")
        all_games = self.db.get_all_player_game_logs(seasons)
        print(f"   ✓ Loaded {len(all_games):,} game records")

        # Get unique players
        unique_players = all_games['player_id'].unique()
        print(f"   ✓ Found {len(unique_players):,} unique players")

        if sample_size:
            unique_players = unique_players[:sample_size]
            print(f"   ℹ Processing sample of {sample_size} players for testing")

        # Process each player
        print(f"\n2. Processing players...")
        total_features = 0

        for player_id in tqdm(unique_players, desc="Players"):
            # Get this player's games
            player_games = all_games[all_games['player_id'] == player_id].copy()

            if len(player_games) < MIN_GAMES_FOR_PREDICTION:
                continue  # Skip players with too few games

            # Process games
            player_features = self.process_player_games(player_id, player_games)

            if player_features:
                # Convert to DataFrame
                features_df = pd.DataFrame(player_features)

                # Store in database
                success = self.db.insert_features(features_df)

                if success:
                    total_features += len(features_df)

        print(f"\n3. Feature Engineering Complete!")
        print(f"   ✓ Created {total_features:,} feature records")
        print(f"   ✓ Stored in feature_store table")

        return total_features


def main():
    """Run feature engineering pipeline"""
    pipeline = FeaturePipeline()

    # For initial testing, process a small sample
    print("Running feature engineering on SAMPLE (50 players)...\n")
    count = pipeline.build_all_features(sample_size=50)

    print(f"\n{'='*60}")
    print(f"SAMPLE COMPLETE: {count:,} features created")
    print(f"{'='*60}")

    # Show database stats
    db = DatabaseManager()
    stats = db.get_database_stats()
    print("\nDatabase Statistics:")
    print(f"  feature_store: {stats['feature_store']:,} rows")

    print("\nTo process ALL players, run:")
    print("  pipeline.build_all_features()")


if __name__ == "__main__":
    main()
