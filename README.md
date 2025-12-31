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

## Make Predictions

```bash
python3 simple_predict.py
```

Output:
```
LeBron James @ Warriors: 24.7 pts | 6.4 reb | 7.9 ast
Stephen Curry vs Lakers: 22.3 pts | 4.6 reb | 5.8 ast
```

## Interactive Mode

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

## Files

**Setup:**
- `setup_from_scratch.py` - Complete setup pipeline
- `setup_database.py` - Create database tables
- `collect_data.py` - Fetch NBA data

**Prediction:**
- `simple_predict.py` - Quick predictions
- `predict_next_game.py` - Interactive mode

**Core:**
- `config.py` - Settings
- `database_manager.py` - Database
- `features/` - Feature engineering
- `models/` - Model training
- `prediction/` - Prediction logic

That's it!
