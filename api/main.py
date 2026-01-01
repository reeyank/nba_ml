"""
FastAPI Backend for NBA Predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction.predictor import GamePredictor
from database_manager import DatabaseManager
from api.nba_schedule import get_todays_games, get_upcoming_games, get_player_roster, get_recent_games_from_api, get_game_box_score
from api.model_evaluation import get_predicted_vs_actual
from config import MODELS_DIR

app = FastAPI(
    title="NBA Stats Predictor API",
    description="Predict NBA player statistics using ML models",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize predictors for all 3 model types
predictors = {}
db = DatabaseManager()


def get_latest_model_version(model_type: str) -> Optional[str]:
    """Get the latest model version for a given type"""
    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]

    if model_type == 'xgboost':
        # XGBoost models don't have prefix
        xgb_dirs = [d for d in model_dirs if d.name.startswith('v1.0.0')]
        if xgb_dirs:
            return sorted(xgb_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0].name
    elif model_type == 'lightgbm':
        lgb_dirs = [d for d in model_dirs if d.name.startswith('lightgbm_')]
        if lgb_dirs:
            return sorted(lgb_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0].name
    elif model_type == 'lstm':
        lstm_dirs = [d for d in model_dirs if d.name.startswith('lstm_')]
        if lstm_dirs:
            return sorted(lstm_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0].name

    return None


def initialize_predictors():
    """Initialize predictors for all available model types"""
    global predictors

    for model_type in ['xgboost', 'lightgbm', 'lstm']:
        try:
            version = get_latest_model_version(model_type)
            if version:
                predictors[model_type] = GamePredictor(model_version=version)
                print(f"Loaded {model_type}: {version}")
        except Exception as e:
            print(f"Could not load {model_type} model: {e}")
            predictors[model_type] = None


# Initialize all predictors at startup
initialize_predictors()

# Keep a reference to first available predictor for backwards compatibility
predictor = predictors.get('xgboost') or predictors.get('lightgbm') or predictors.get('lstm')


# Pydantic models
class PlayerPrediction(BaseModel):
    player_id: int
    player_name: str
    team: str
    predicted_points: float
    predicted_rebounds: float
    predicted_assists: float


class GamePredictions(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_players: List[PlayerPrediction]
    away_players: List[PlayerPrediction]


class PredictionComparison(BaseModel):
    player_name: str
    game_date: str
    predicted_points: float
    actual_points: float
    predicted_rebounds: float
    actual_rebounds: float
    predicted_assists: float
    actual_assists: float
    points_error: float
    rebounds_error: float
    assists_error: float


# Routes
@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": predictor.model_type,
        "version": predictor.model_version
    }


@app.get("/api/games/today")
async def get_today_games():
    """Get today's NBA games"""
    try:
        games = get_todays_games()
        return {"games": games, "count": len(games)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/games/recent")
async def get_recent_games(days: int = 7):
    """Get recent completed NBA games from NBA API"""
    try:
        games = get_recent_games_from_api(days=days)
        return {"games": games, "count": len(games)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/games/upcoming")
async def get_upcoming(days: int = 7):
    """Get upcoming NBA games for next N days"""
    try:
        games = get_upcoming_games(days)
        return {"games": games, "count": len(games)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/game/{game_id}")
async def predict_game(game_id: str):
    """Get predictions for all players in a specific game"""
    try:
        # Get game details and rosters
        game_info = get_player_roster(game_id)

        if not game_info['players']:
            raise HTTPException(
                status_code=404,
                detail="No players found for this game"
            )

        predictions = []
        errors = []

        for player in game_info['players']:
            try:
                pred = predictor.predict_game(
                    player_id=player['player_id'],
                    game_date=game_info['game_date'],
                    opponent_team_id=player['opponent_team_id'],
                    is_home=player['is_home']
                )

                if pred:
                    predictions.append({
                        'player_id': player['player_id'],
                        'player_name': player['player_name'],  # Use name from roster, not prediction
                        'team': player['team'],
                        'predicted_points': round(float(pred['predicted_points']), 1),
                        'predicted_rebounds': round(float(pred['predicted_rebounds']), 1),
                        'predicted_assists': round(float(pred['predicted_assists']), 1)
                    })
                else:
                    errors.append(f"{player['player_name']}: No recent game history")

            except Exception as e:
                errors.append(f"{player['player_name']}: {str(e)}")
                continue

        if not predictions and errors:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available. Errors: {errors[:3]}"
            )

        return {
            "game_id": game_id,
            "game_date": game_info['game_date'],
            "home_team": game_info['home_team'],
            "away_team": game_info['away_team'],
            "predictions": predictions,
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/api/predictions/game/{game_id}/all-models")
async def predict_game_all_models(game_id: str):
    """Get predictions from all 3 models for all players in a specific game"""

    try:
        # Get game details and rosters
        game_info = get_player_roster(game_id)

        if not game_info['players']:
            raise HTTPException(
                status_code=404,
                detail="No players found for this game"
            )

        # Try to get actual stats from NBA API box score
        actual_stats = get_game_box_score(game_id)
        game_status = "COMPLETED" if actual_stats else "UPCOMING"

        predictions = []
        errors = []

        for player in game_info['players']:
            player_preds = {
                'player_id': player['player_id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'models': {}
            }

            # Get actual stats if available from box score
            if player['player_id'] in actual_stats:
                player_preds['actual'] = {
                    'points': actual_stats[player['player_id']]['points'],
                    'rebounds': actual_stats[player['player_id']]['rebounds'],
                    'assists': actual_stats[player['player_id']]['assists']
                }

            # Get predictions from each model
            for model_type, pred_instance in predictors.items():
                if pred_instance is None:
                    continue

                try:
                    pred = pred_instance.predict_game(
                        player_id=player['player_id'],
                        game_date=game_info['game_date'],
                        opponent_team_id=player['opponent_team_id'],
                        is_home=player['is_home']
                    )

                    if pred:
                        player_preds['models'][model_type] = {
                            'points': round(float(pred['predicted_points']), 1),
                            'rebounds': round(float(pred['predicted_rebounds']), 1),
                            'assists': round(float(pred['predicted_assists']), 1)
                        }
                except Exception as e:
                    # Skip silently for individual model failures
                    pass

            # Only add if we have at least one model prediction
            if player_preds['models']:
                predictions.append(player_preds)
            else:
                errors.append(f"{player['player_name']}: No predictions available")

        if not predictions and errors:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available. Errors: {errors[:3]}"
            )

        return {
            "game_id": game_id,
            "game_date": game_info['game_date'],
            "game_status": game_status,
            "home_team": game_info['home_team'],
            "away_team": game_info['away_team'],
            "available_models": [m for m, p in predictors.items() if p is not None],
            "predictions": predictions,
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/api/games/actual/{game_id}")
async def get_actual_stats(game_id: str):
    """Get actual player stats for a completed game"""
    import pandas as pd

    try:
        with db.get_connection() as conn:
            query = """
                SELECT
                    player_id,
                    player_name,
                    team_id,
                    points,
                    rebounds,
                    assists,
                    minutes,
                    field_goals_made,
                    field_goals_attempted,
                    three_pointers_made,
                    three_pointers_attempted,
                    game_date,
                    matchup
                FROM player_game_logs
                WHERE game_id = ?
                ORDER BY points DESC
            """
            result = pd.read_sql(query, conn, params=[game_id])

        if result.empty:
            raise HTTPException(
                status_code=404,
                detail="No stats found for this game. Game may not be completed yet."
            )

        players = result.to_dict('records')

        # Parse home/away from first player's matchup
        matchup = players[0]['matchup'] if players else ''
        is_home = 'vs.' in matchup
        parts = matchup.replace(' vs. ', ' @ ').split(' @ ')
        team1 = parts[0].strip() if parts else ''
        team2 = parts[1].strip() if len(parts) > 1 else ''

        return {
            "game_id": game_id,
            "game_date": players[0]['game_date'] if players else '',
            "home_team": team1 if is_home else team2,
            "away_team": team2 if is_home else team1,
            "players": players
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/player/{player_id}")
async def predict_player(
    player_id: int,
    game_date: str,
    opponent_team_id: int,
    is_home: bool
):
    """Get prediction for a specific player"""
    try:
        prediction = predictor.predict_game(
            player_id=player_id,
            game_date=game_date,
            opponent_team_id=opponent_team_id,
            is_home=is_home
        )

        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not available")

        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/players/search")
async def search_players(query: str):
    """Search for players by name"""
    try:
        # Search in database
        with db.get_connection() as conn:
            import pandas as pd
            result = pd.read_sql(
                """
                SELECT DISTINCT player_id, player_name
                FROM player_game_logs
                WHERE player_name LIKE ?
                ORDER BY player_name
                LIMIT 50
                """,
                conn,
                params=[f"%{query}%"]
            )

        players = result.to_dict('records')
        return {"players": players, "count": len(players)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluation/comparison")
async def get_model_comparison(
    limit: int = 100,
    min_date: Optional[str] = None
):
    """Get predicted vs actual comparison for recent games"""
    try:
        comparisons = get_predicted_vs_actual(limit=limit, min_date=min_date)
        return {
            "comparisons": comparisons,
            "count": len(comparisons)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/summary")
async def get_summary_stats():
    """Get summary statistics about the dataset and models"""
    try:
        stats = db.get_database_stats()

        # Get date ranges
        with db.get_connection() as conn:
            import pandas as pd
            date_info = pd.read_sql(
                "SELECT MIN(game_date) as min_date, MAX(game_date) as max_date FROM player_game_logs",
                conn
            ).iloc[0]

            player_count = pd.read_sql(
                "SELECT COUNT(DISTINCT player_id) as count FROM player_game_logs",
                conn
            ).iloc[0]['count']

        return {
            "total_games": stats['player_game_logs'],
            "total_players": player_count,
            "features_created": stats['feature_store'],
            "date_range": {
                "start": date_info['min_date'],
                "end": date_info['max_date']
            },
            "model": {
                "type": predictor.model_type,
                "version": predictor.model_version
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/players/top")
async def get_top_players(limit: int = 20):
    """Get top players with recent activity"""
    try:
        with db.get_connection() as conn:
            import pandas as pd

            # Get latest date in database
            latest_date = pd.read_sql(
                "SELECT MAX(game_date) as max_date FROM player_game_logs",
                conn
            ).iloc[0]['max_date']

            result = pd.read_sql(
                """
                SELECT
                    player_id,
                    player_name,
                    COUNT(*) as total_games,
                    MAX(game_date) as last_game,
                    AVG(points) as avg_points,
                    AVG(rebounds) as avg_rebounds,
                    AVG(assists) as avg_assists
                FROM player_game_logs
                WHERE game_date >= date(?, '-90 days')
                GROUP BY player_id
                HAVING total_games >= 10
                ORDER BY total_games DESC, avg_points DESC
                LIMIT ?
                """,
                conn,
                params=[latest_date, limit]
            )

        players = result.to_dict('records')
        return {"players": players, "count": len(players)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
