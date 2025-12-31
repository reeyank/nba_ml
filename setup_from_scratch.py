#!/usr/bin/env python3
"""
Complete Setup Pipeline - Builds everything from scratch

This script:
1. Creates database tables
2. Collects NBA data from API
3. Builds ML features
4. Trains prediction models

Run this once to set up the entire system.
"""

import sys
import time
from pathlib import Path

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))


def run_setup():
    """Run complete setup pipeline"""

    print("\n" + "=" * 70)
    print(" NBA ML PREDICTOR - COMPLETE SETUP FROM SCRATCH")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Create database tables")
    print("  2. Collect NBA game data (2022-2025)")
    print("  3. Build ML features")
    print("  4. Train XGBoost models")
    print("\nEstimated time: 5-10 minutes")
    print("=" * 70)

    input("\nPress ENTER to continue or Ctrl+C to cancel...")

    # Step 1: Create database
    print("\n" + "=" * 70)
    print("STEP 1/4: Creating Database Tables")
    print("=" * 70)

    from setup_database import create_tables
    create_tables()

    time.sleep(1)

    # Step 2: Collect data
    print("\n" + "=" * 70)
    print("STEP 2/4: Collecting NBA Data")
    print("=" * 70)
    print("This may take a few minutes...")

    from collect_data import collect_game_logs, collect_opponent_defensive_stats

    total_games = collect_game_logs()

    if total_games == 0:
        print("\n✗ Data collection failed. Exiting.")
        return False

    collect_opponent_defensive_stats()

    time.sleep(1)

    # Step 3: Build features
    print("\n" + "=" * 70)
    print("STEP 3/4: Building ML Features")
    print("=" * 70)
    print("Processing game data into ML features...")

    from features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    feature_count = pipeline.build_all_features()

    if feature_count == 0:
        print("\n✗ Feature building failed. Exiting.")
        return False

    time.sleep(1)

    # Step 4: Train models
    print("\n" + "=" * 70)
    print("STEP 4/4: Training XGBoost Models")
    print("=" * 70)
    print("Training 3 models (points, rebounds, assists)...")
    print("This may take several minutes...")

    from models.xgboost_model import XGBoostTrainer

    trainer = XGBoostTrainer()
    metrics = trainer.train_all_models()

    # Final summary
    print("\n" + "=" * 70)
    print(" SETUP COMPLETE!")
    print("=" * 70)
    print("\nYour NBA ML prediction system is ready!")
    print("\nNext steps:")
    print("  1. Run predictions: python3 simple_predict.py")
    print("  2. Interactive mode: python3 predict_next_game.py")
    print("\n" + "=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
