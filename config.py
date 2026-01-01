"""
Simple configuration file for NBA ML project
"""
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_STORAGE_DIR = PROJECT_ROOT / "data_storage"
MODELS_DIR = DATA_STORAGE_DIR / "models"
DATABASE_PATH = DATA_STORAGE_DIR / "nba_predictions.db"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data collection settings
SEASONS = ['2022-23', '2023-24', '2024-25']

# Feature engineering settings
ROLLING_WINDOWS = [5, 10, 15]
MIN_GAMES_FOR_PREDICTION = 3

# Model training settings
TARGET_VARIABLES = ['points', 'rebounds', 'assists']

# Data splits (chronological)
TRAIN_VALIDATION_SPLIT_DATE = '2024-02-01'
VALIDATION_TEST_SPLIT_DATE = '2024-04-15'

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# LSTM hyperparameters
LSTM_PARAMS = {
    'sequence_length': 10,      # Number of prior games in sequence
    'hidden_size': 64,          # LSTM hidden layer size
    'num_layers': 2,            # Number of LSTM layers
    'dropout': 0.2,             # Dropout rate
    'learning_rate': 0.001,     # Adam learning rate
    'batch_size': 64,           # Training batch size
    'epochs': 100,              # Max training epochs
    'patience': 15              # Early stopping patience
}

# Target MAE thresholds
TARGET_MAE = {
    'points': 5.0,
    'rebounds': 2.0,
    'assists': 1.5
}
