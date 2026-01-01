# NBA ML Predictor

Predict NBA player stats using XGBoost models.

## Setup from Scratch

If you have no data yet, run this once:

```bash
python3 setup_from_scratch.py
```

This will:
1. Create database tables
2. Collect NBA game data (2022-2025)
3. Build ML features
4. Train prediction models

Takes 5-10 minutes.

## Web App (NEW!)

Launch the full web interface:

```bash
python3 run_api.py
```

Then open: **http://localhost:8000**

Features:
- View upcoming NBA games
- Get predictions for ALL players
- Search players
- See model performance (predicted vs actual)

See [WEB_APP.md](WEB_APP.md) for deployment to Vercel.

## CLI Predictions

**Quick predictions:**
```bash
python3 simple_predict.py
```

**Interactive mode:**
```bash
python3 predict_next_game.py
```

## Use in Code

```python
from prediction.predictor import GamePredictor

predictor = GamePredictor()

prediction = predictor.predict_game(
    player_name="LeBron James",
    game_date="2025-01-20",
    opponent_team_id=1610612744,
    is_home=False
)

print(f"{prediction['predicted_points']:.1f} pts")
```

## Model Options

You can train with **XGBoost** (default) or **LightGBM**:

**Train XGBoost only:**
```bash
python3 models/xgboost_model.py
```

**Train LightGBM only:**
```bash
python3 models/lightgbm_model.py
```

**Train both and compare:**
```bash
python3 compare_models.py
```

The predictor auto-loads the latest XGBoost model by default.

**Use specific model:**
```python
# Use LightGBM
predictor = GamePredictor(model_version='lightgbm_v1.0.0_...')

# Use XGBoost
predictor = GamePredictor(model_version='v1.0.0_...')
```

## Files

**Setup:**
- `setup_from_scratch.py` - Complete setup pipeline
- `setup_database.py` - Create database tables
- `collect_data.py` - Fetch NBA data

**Prediction:**
- `simple_predict.py` - Quick predictions
- `predict_next_game.py` - Interactive mode

**Models:**
- `models/xgboost_model.py` - Train XGBoost
- `models/lightgbm_model.py` - Train LightGBM
- `compare_models.py` - Compare both

**Core:**
- `config.py` - Settings
- `database_manager.py` - Database
- `features/` - Feature engineering
- `prediction/` - Prediction logic

That's it!
