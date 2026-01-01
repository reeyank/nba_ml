"""
Model Evaluation - Compare Predicted vs Actual Stats
Shows model performance on games not in training data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_manager import DatabaseManager
from prediction.predictor import GamePredictor
from config import VALIDATION_TEST_SPLIT_DATE
import pandas as pd


def get_predicted_vs_actual(limit=100, min_date=None):
    """
    Get predicted vs actual comparison for recent test set games

    Args:
        limit: Number of comparisons to return
        min_date: Only include games after this date (defaults to test split date)
    """
    db = DatabaseManager()
    predictor = GamePredictor()

    if min_date is None:
        min_date = VALIDATION_TEST_SPLIT_DATE

    # Get recent games from test set
    with db.get_connection() as conn:
        query = """
            SELECT
                player_id,
                player_name,
                game_id,
                game_date,
                matchup,
                points as actual_points,
                rebounds as actual_rebounds,
                assists as actual_assists
            FROM player_game_logs
            WHERE game_date >= ?
            ORDER BY game_date DESC
            LIMIT ?
        """

        games_df = pd.read_sql(query, conn, params=[min_date, limit * 2])

    comparisons = []

    for _, game in games_df.iterrows():
        try:
            # Determine if home game
            is_home = 'vs.' in game['matchup']

            # Get opponent team ID (simplified - would need proper parsing)
            opponent_team_id = 1610612744  # Mock for now

            # Make prediction
            pred = predictor.predict_game(
                player_id=int(game['player_id']),
                game_date=game['game_date'],
                opponent_team_id=opponent_team_id,
                is_home=is_home
            )

            if pred:
                pred_pts = round(float(pred['predicted_points']), 1)
                pred_reb = round(float(pred['predicted_rebounds']), 1)
                pred_ast = round(float(pred['predicted_assists']), 1)
                actual_pts = int(game['actual_points'])
                actual_reb = int(game['actual_rebounds'])
                actual_ast = int(game['actual_assists'])

                comparisons.append({
                    'player_name': game['player_name'],
                    'game_date': game['game_date'],
                    'matchup': game['matchup'],
                    'predicted_points': pred_pts,
                    'actual_points': actual_pts,
                    'predicted_rebounds': pred_reb,
                    'actual_rebounds': actual_reb,
                    'predicted_assists': pred_ast,
                    'actual_assists': actual_ast,
                    'points_error': round(abs(pred_pts - actual_pts), 1),
                    'rebounds_error': round(abs(pred_reb - actual_reb), 1),
                    'assists_error': round(abs(pred_ast - actual_ast), 1)
                })

                if len(comparisons) >= limit:
                    break

        except Exception as e:
            print(f"Error predicting for {game['player_name']}: {e}")
            continue

    return comparisons


def get_model_accuracy_summary():
    """Get overall model accuracy metrics"""
    comparisons = get_predicted_vs_actual(limit=500)

    if not comparisons:
        return None

    df = pd.DataFrame(comparisons)

    summary = {
        'points': {
            'mae': df['points_error'].mean(),
            'median_error': df['points_error'].median(),
            'max_error': df['points_error'].max()
        },
        'rebounds': {
            'mae': df['rebounds_error'].mean(),
            'median_error': df['rebounds_error'].median(),
            'max_error': df['rebounds_error'].max()
        },
        'assists': {
            'mae': df['assists_error'].mean(),
            'median_error': df['assists_error'].median(),
            'max_error': df['assists_error'].max()
        },
        'sample_size': len(comparisons)
    }

    return summary
