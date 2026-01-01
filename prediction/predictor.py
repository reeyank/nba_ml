"""
Game Predictor - Makes predictions using trained models (XGBoost, LightGBM, or LSTM)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_manager import DatabaseManager
from features.rolling_stats import RollingStatsCalculator
from features.contextual_features import ContextualFeaturesCalculator
from config import MODELS_DIR, LSTM_PARAMS


class GamePredictor:
    """Predicts player stats for upcoming games using trained models"""

    def __init__(self, model_version: str = None, db_manager: DatabaseManager = None):
        """
        Initialize predictor with trained models

        Args:
            model_version: Specific model version to load (e.g., 'v1.0.0_20251230_161033')
                          If None, loads the active model from database
            db_manager: Database manager instance
        """
        self.db = db_manager or DatabaseManager()
        self.rolling_calculator = RollingStatsCalculator()
        self.contextual_calculator = ContextualFeaturesCalculator()

        # Load models
        self.model_version = model_version or self._get_active_model_version()
        self.model_type = self._detect_model_type()
        self.models = self._load_models()
        self.feature_columns = self._load_feature_columns()

        # LSTM-specific attributes
        self.scalers = {}
        self.lstm_config = None
        if self.model_type == 'lstm':
            self._load_lstm_components()

        print(f"Loaded {self.model_type.upper()} models: {self.model_version}")

    def _get_active_model_version(self) -> str:
        """Get latest model version from directory (prefers XGBoost over LightGBM)"""
        # List all model directories
        model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
        if not model_dirs:
            raise ValueError("No models found in directory")

        # Separate XGBoost and LightGBM models
        xgb_models = [d for d in model_dirs if not d.name.startswith('lightgbm')]
        lgb_models = [d for d in model_dirs if d.name.startswith('lightgbm')]

        # Prefer XGBoost, fallback to LightGBM
        if xgb_models:
            latest_model = sorted(xgb_models)[-1].name
        elif lgb_models:
            latest_model = sorted(lgb_models)[-1].name
        else:
            latest_model = sorted(model_dirs)[-1].name

        return latest_model

    def _detect_model_type(self) -> str:
        """Detect if model is XGBoost or LightGBM"""
        model_dir = MODELS_DIR / self.model_version

        # Check for model type file
        model_type_file = model_dir / "model_type.txt"
        if model_type_file.exists():
            with open(model_type_file, 'r') as f:
                return f.read().strip()

        # Fallback: check version name
        if 'lightgbm' in self.model_version.lower():
            return 'lightgbm'

        return 'xgboost'

    def _load_models(self) -> Dict:
        """Load trained models (XGBoost, LightGBM, or LSTM)"""
        model_dir = MODELS_DIR / self.model_version
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        models = {}

        if self.model_type == 'lstm':
            # Load PyTorch LSTM models
            import torch
            from models.lstm_model import PlayerLSTM

            for target in ['points', 'rebounds', 'assists']:
                model_path = model_dir / f"{target}_model.pt"
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                model = PlayerLSTM(
                    input_size=checkpoint['input_size'],
                    hidden_size=checkpoint['hidden_size'],
                    num_layers=checkpoint['num_layers']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[target] = model
        else:
            # Load pickle models (XGBoost/LightGBM)
            for target in ['points', 'rebounds', 'assists']:
                model_path = model_dir / f"{target}_model.pkl"
                with open(model_path, 'rb') as f:
                    models[target] = pickle.load(f)

        return models

    def _load_lstm_components(self):
        """Load LSTM-specific components (scalers and config)"""
        model_dir = MODELS_DIR / self.model_version

        # Load scalers for each target
        for target in ['points', 'rebounds', 'assists']:
            scaler_path = model_dir / f"{target}_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers[target] = pickle.load(f)

        # Load LSTM config
        config_path = model_dir / "lstm_config.pkl"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                self.lstm_config = pickle.load(f)
        else:
            self.lstm_config = LSTM_PARAMS

    def _load_feature_columns(self) -> list:
        """Load feature column names"""
        feature_path = MODELS_DIR / self.model_version / "feature_columns.pkl"
        with open(feature_path, 'rb') as f:
            return pickle.load(f)

    def _get_player_id_by_name(self, player_name: str) -> Optional[int]:
        """Get player ID from name (simplified lookup)"""
        # Known player IDs for demo
        known_players = {
            'LeBron James': 2544,
            'Stephen Curry': 201939,
            'Giannis Antetokounmpo': 203507,
            'Kevin Durant': 201142,
            'Luka Doncic': 1629029,
            'Nikola Jokic': 203999,
            'Joel Embiid': 203954,
            'Jayson Tatum': 1628369,
            'Damian Lillard': 203081,
            'Anthony Davis': 203076
        }

        if player_name in known_players:
            return known_players[player_name]

        # Try to find in database (simplified - would need better name matching)
        # For now, return None if not in known players
        return None

    def build_prediction_features(
        self,
        player_id: int,
        game_date: str,
        opponent_team_id: int,
        is_home: bool,
        current_team_id: int = None,
        season: str = "2024-25"
    ) -> Optional[pd.DataFrame]:
        """
        Build feature vector for prediction

        Args:
            player_id: Player's ID
            game_date: Date of game to predict (YYYY-MM-DD)
            opponent_team_id: Opponent team ID
            is_home: True if home game
            current_team_id: Player's current team ID (optional)
            season: Current season

        Returns:
            DataFrame with single row of features, or None if insufficient data
        """
        # Get player's recent games (before the prediction date)
        recent_games = self.db.get_player_games_before_date(player_id, game_date, limit=15)

        if len(recent_games) < 3:
            print(f"⚠️  Insufficient game history for player {player_id}")
            return None

        # Calculate rolling averages
        rolling_features = self.rolling_calculator.calculate_all_windows(recent_games)
        if not rolling_features:
            return None

        # Calculate trend features
        trend_features = self.rolling_calculator.calculate_trend_features(recent_games)

        # Calculate rest days
        rest_days = self.contextual_calculator.calculate_rest_days(game_date, recent_games)
        is_back_to_back = (rest_days == 1) if rest_days is not None else False

        # Get team stats (use player's most recent team if not provided)
        if current_team_id is None:
            current_team_id = recent_games.iloc[0].get('team_id') if 'team_id' in recent_games.columns else None

        team_pace = None
        team_offensive_rating = None
        team_defensive_rating = None

        if current_team_id:
            team_stats = self.db.get_team_stats_for_game(current_team_id, game_date)
            if team_stats:
                team_pace = team_stats.get('pace')
                team_offensive_rating = team_stats.get('offensive_rating')
                team_defensive_rating = team_stats.get('defensive_rating')

        # Get opponent defensive rating
        opponent_defensive_rating = self.db.get_opponent_defensive_rating(
            opponent_team_id, season, game_date
        )

        # Get opponent pace
        opponent_pace = None
        opp_team_stats = self.db.get_team_stats_for_game(opponent_team_id, game_date)
        if opp_team_stats:
            opponent_pace = opp_team_stats.get('pace')

        # Combine all features
        features = {
            **rolling_features,
            **trend_features,
            'is_home': int(is_home),
            'rest_days': rest_days if rest_days is not None else 0,
            'is_back_to_back': int(is_back_to_back),
            'opponent_team_id': opponent_team_id,
            'opponent_defensive_rating': opponent_defensive_rating,
            'opponent_pace': opponent_pace,
            'team_pace': team_pace,
            'team_offensive_rating': team_offensive_rating,
            'team_defensive_rating': team_defensive_rating,
        }

        # Create DataFrame with feature columns in correct order
        feature_df = pd.DataFrame([features])

        # Ensure all feature columns exist (fill missing with 0)
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0

        # Select only the feature columns in the correct order
        feature_df = feature_df[self.feature_columns]

        # Fill any remaining NaN values
        feature_df = feature_df.fillna(0)

        return feature_df

    def build_lstm_sequence(
        self,
        player_id: int,
        game_date: str
    ) -> Optional[np.ndarray]:
        """
        Build sequence of recent games for LSTM prediction

        Args:
            player_id: Player's ID
            game_date: Date of game to predict (get games before this)

        Returns:
            Numpy array of shape (seq_len, n_features) or None if insufficient data
        """
        seq_len = self.lstm_config.get('sequence_length', 10)

        # Get player's recent games
        recent_games = self.db.get_player_games_before_date(player_id, game_date, limit=seq_len)

        if len(recent_games) < seq_len:
            print(f"Insufficient game history for LSTM ({len(recent_games)}/{seq_len} games)")
            return None

        # Sort by date ascending (oldest first)
        recent_games = recent_games.sort_values('game_date').reset_index(drop=True)

        # Parse is_home from matchup
        recent_games['is_home'] = recent_games['matchup'].apply(lambda x: 1 if 'vs.' in str(x) else 0)

        # Calculate rest days
        recent_games['game_date_dt'] = pd.to_datetime(recent_games['game_date'])
        recent_games['rest_days'] = recent_games['game_date_dt'].diff().dt.days.fillna(3)
        recent_games['rest_days'] = recent_games['rest_days'].clip(0, 10)

        # Fill NaN in percentage columns
        for col in ['field_goal_pct', 'three_point_pct', 'free_throw_pct']:
            if col in recent_games.columns:
                recent_games[col] = recent_games[col].fillna(0)

        # Extract features in correct order
        sequence = recent_games[self.feature_columns].values

        return sequence

    def predict_game(
        self,
        player_name: str = None,
        player_id: int = None,
        game_date: str = None,
        opponent_team_id: int = None,
        is_home: bool = True,
        current_team_id: int = None,
        season: str = "2024-25"
    ) -> Optional[Dict]:
        """
        Predict stats for a player's next game

        Args:
            player_name: Player's name (alternative to player_id)
            player_id: Player's ID
            game_date: Date of game (YYYY-MM-DD)
            opponent_team_id: Opponent's team ID
            is_home: True if home game
            current_team_id: Player's team ID (optional)
            season: Current season

        Returns:
            Dictionary with predictions or None if prediction fails
        """
        # Get player ID from name if needed
        if player_id is None:
            if player_name is None:
                raise ValueError("Must provide either player_name or player_id")
            player_id = self._get_player_id_by_name(player_name)
            if player_id is None:
                print(f"Player '{player_name}' not found")
                return None
        else:
            player_name = player_name or f"Player {player_id}"

        # Make predictions based on model type
        predictions = {
            'player_name': player_name,
            'player_id': player_id,
            'game_date': game_date,
            'is_home': is_home,
            'opponent_team_id': opponent_team_id,
            'model_type': self.model_type
        }

        if self.model_type == 'lstm':
            # LSTM prediction using sequences
            sequence = self.build_lstm_sequence(player_id, game_date)
            if sequence is None:
                return None

            import torch

            for target, model in self.models.items():
                # Scale the sequence using the target's scaler
                if target in self.scalers:
                    scaler = self.scalers[target]
                    seq_scaled = scaler.transform(sequence)
                else:
                    seq_scaled = sequence

                # Convert to tensor and add batch dimension
                seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    pred = model(seq_tensor).item()

                predictions[f'predicted_{target}'] = round(pred, 1)
        else:
            # Tree-based model prediction (XGBoost/LightGBM)
            features = self.build_prediction_features(
                player_id=player_id,
                game_date=game_date,
                opponent_team_id=opponent_team_id,
                is_home=is_home,
                current_team_id=current_team_id,
                season=season
            )

            if features is None:
                return None

            for target, model in self.models.items():
                # Handle different prediction APIs
                if self.model_type == 'lightgbm':
                    pred = model.predict(features, num_iteration=model.best_iteration)[0]
                else:  # xgboost
                    pred = model.predict(features)[0]

                predictions[f'predicted_{target}'] = round(pred, 1)

        return predictions

    def predict_todays_games(self, game_date: str = None):
        """
        Predict stats for all games on a specific date

        Args:
            game_date: Date to predict (YYYY-MM-DD), defaults to today

        Returns:
            List of predictions
        """
        # This would require fetching today's schedule from ScoreboardV2
        # For now, it's a placeholder for future implementation
        print("⚠️  Bulk game prediction not yet implemented")
        print("   Use predict_game() for individual predictions")
        return []


def demo_predictions():
    """Demonstrate making predictions"""
    print("="*60)
    print("NBA STAT PREDICTION DEMO")
    print("="*60)

    # Initialize predictor
    predictor = GamePredictor()

    # Example predictions
    players = [
        {
            'name': 'LeBron James',
            'date': '2025-01-20',
            'opponent': 1610612744,  # Warriors
            'home': False
        },
        {
            'name': 'Stephen Curry',
            'date': '2025-01-20',
            'opponent': 1610612747,  # Lakers
            'home': True
        },
        {
            'name': 'Giannis Antetokounmpo',
            'date': '2025-01-20',
            'opponent': 1610612738,  # Celtics
            'home': True
        }
    ]

    print("\nPredictions for upcoming games:\n")

    for player in players:
        pred = predictor.predict_game(
            player_name=player['name'],
            game_date=player['date'],
            opponent_team_id=player['opponent'],
            is_home=player['home']
        )

        if pred:
            home_away = "vs" if pred['is_home'] else "@"
            print(f"{pred['player_name']} {home_away} Team {pred['opponent_team_id']}:")
            print(f"  📊 {pred['predicted_points']} pts | "
                  f"{pred['predicted_rebounds']} reb | "
                  f"{pred['predicted_assists']} ast")
            print()
        else:
            print(f"❌ Could not predict for {player['name']}\n")


if __name__ == "__main__":
    demo_predictions()
