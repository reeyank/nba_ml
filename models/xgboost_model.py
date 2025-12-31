"""
XGBoost Model Training
Trains separate XGBoost models for points, rebounds, and assists prediction
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_manager import DatabaseManager
from config import (
    XGBOOST_PARAMS,
    TRAIN_VALIDATION_SPLIT_DATE,
    VALIDATION_TEST_SPLIT_DATE,
    TARGET_VARIABLES,
    MODELS_DIR
)


class XGBoostTrainer:
    """Trains XGBoost models for player stat prediction"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.models = {}
        self.feature_columns = None

    def load_data(self, seasons: list = None) -> pd.DataFrame:
        """Load feature data from database"""
        print("Loading feature data...")
        features = self.db.get_all_features(seasons)
        print(f"✓ Loaded {len(features):,} feature records")
        return features

    def prepare_data_splits(self, features: pd.DataFrame):
        """
        Split data chronologically into train/val/test sets

        Train: 2022-23 + first 60% of 2023-24
        Val: Last 40% of 2023-24
        Test: 2024-25
        """
        print("\nSplitting data chronologically...")

        # Sort by date
        features = features.sort_values('game_date')

        # Split based on dates
        train = features[features['game_date'] < TRAIN_VALIDATION_SPLIT_DATE]
        val = features[
            (features['game_date'] >= TRAIN_VALIDATION_SPLIT_DATE) &
            (features['game_date'] < VALIDATION_TEST_SPLIT_DATE)
        ]
        test = features[features['game_date'] >= VALIDATION_TEST_SPLIT_DATE]

        print(f"✓ Train: {len(train):,} samples ({train['game_date'].min()} to {train['game_date'].max()})")
        print(f"✓ Val:   {len(val):,} samples ({val['game_date'].min()} to {val['game_date'].max()})")
        print(f"✓ Test:  {len(test):,} samples ({test['game_date'].min()} to {test['game_date'].max()})")

        return train, val, test

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns (exclude metadata and targets)"""
        exclude_cols = [
            'id', 'player_id', 'game_id', 'game_date', 'season', 'opponent_team_id',
            'target_points', 'target_rebounds', 'target_assists',
            'created_at'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def prepare_features_and_targets(self, df: pd.DataFrame, target: str):
        """Prepare X and y for training"""
        # Get feature columns
        if self.feature_columns is None:
            self.feature_columns = self.get_feature_columns(df)

        X = df[self.feature_columns].copy()
        y = df[f'target_{target}'].copy()

        # Fill NaN values with 0 (for missing team/opponent stats)
        X = X.fillna(0)

        return X, y

    def train_model(self, X_train, y_train, X_val, y_val, target: str):
        """Train XGBoost model for a specific target"""
        print(f"\n{'='*60}")
        print(f"Training XGBoost for {target.upper()}")
        print(f"{'='*60}")

        # Initialize model
        model = XGBRegressor(
            **XGBOOST_PARAMS,
            early_stopping_rounds=50,
            eval_metric='mae'
        )

        # Train with early stopping
        print(f"Training with {X_train.shape[1]} features...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        print(f"✓ Training complete")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score:.4f}")

        return model

    def evaluate_model(self, model, X, y, dataset_name: str, target: str):
        """Evaluate model and print metrics"""
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        print(f"\n{dataset_name} Set Metrics for {target}:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.3f}")

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def train_all_models(self):
        """Train models for all target variables"""
        print("="*60)
        print("XGBOOST MODEL TRAINING PIPELINE")
        print("="*60)

        # Load data
        features = self.load_data()

        # Split data
        train, val, test = self.prepare_data_splits(features)

        # Train each target
        all_metrics = {}

        for target in TARGET_VARIABLES:
            # Prepare data
            X_train, y_train = self.prepare_features_and_targets(train, target)
            X_val, y_val = self.prepare_features_and_targets(val, target)
            X_test, y_test = self.prepare_features_and_targets(test, target)

            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val, target)

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, "Train", target)
            val_metrics = self.evaluate_model(model, X_val, y_val, "Val", target)
            test_metrics = self.evaluate_model(model, X_test, y_test, "Test", target)

            # Store model and metrics
            self.models[target] = model
            all_metrics[target] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }

            # Show feature importance
            self.show_feature_importance(model, target, top_n=10)

        # Save models
        self.save_models(all_metrics)

        # Print final summary
        self.print_summary(all_metrics)

        return all_metrics

    def show_feature_importance(self, model, target: str, top_n: int = 10):
        """Display top N most important features"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\nTop {top_n} Features for {target}:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {self.feature_columns[idx]}: {importances[idx]:.4f}")

    def save_models(self, metrics: dict):
        """Save trained models and metadata"""
        print(f"\n{'='*60}")
        print("SAVING MODELS")
        print(f"{'='*60}")

        # Create model version
        version = f"v1.0.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = MODELS_DIR / version
        version_dir.mkdir(exist_ok=True)

        # Save each model
        for target, model in self.models.items():
            model_path = version_dir / f"{target}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {target} model to {model_path}")

        # Save feature columns
        feature_path = version_dir / "feature_columns.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"✓ Saved feature columns to {feature_path}")

        # Save minimal metadata to database (optional - we load models from directory)
        try:
            metadata = {
                'model_version': version,
                'model_type': 'xgboost',
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'model_path': str(version_dir),
                'is_active': 1
            }
            self.db.insert_model_metadata(metadata)
            self.db.set_active_model(version)
            print(f"✓ Saved metadata to database (version: {version})")
        except Exception as e:
            print(f"⚠️  Could not save metadata to database (non-critical): {e}")

        return version

    def print_summary(self, metrics: dict):
        """Print final summary of all models"""
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")

        print("\nTest Set Performance:")
        print(f"{'Target':<12} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 42)

        for target in TARGET_VARIABLES:
            test_metrics = metrics[target]['test']
            print(f"{target.capitalize():<12} "
                  f"{test_metrics['mae']:<10.3f} "
                  f"{test_metrics['rmse']:<10.3f} "
                  f"{test_metrics['r2']:<10.3f}")

        # Check if targets met
        from config import TARGET_MAE
        print("\nTarget Achievement:")
        for target in TARGET_VARIABLES:
            actual_mae = metrics[target]['test']['mae']
            target_mae = TARGET_MAE[target]
            status = "✅" if actual_mae <= target_mae else "⚠️"
            print(f"  {status} {target.capitalize()}: {actual_mae:.2f} (target: {target_mae:.2f})")


def main():
    """Run training pipeline"""
    trainer = XGBoostTrainer()
    metrics = trainer.train_all_models()


if __name__ == "__main__":
    main()
