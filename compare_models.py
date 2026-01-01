#!/usr/bin/env python3
"""
Model Comparison Script
Compares XGBoost, LightGBM, and LSTM performance
"""

import sys
import gc
import os
import multiprocessing
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Force multiprocessing to use 'spawn' to avoid semaphore leaks
# This must be done before importing torch or sklearn
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method('spawn', force=True)


def compare_models(include_lstm: bool = True):
    """Train and compare all models"""

    print("=" * 70, flush=True)
    if include_lstm:
        print(" MODEL COMPARISON: XGBoost vs LightGBM vs LSTM", flush=True)
    else:
        print(" MODEL COMPARISON: XGBoost vs LightGBM", flush=True)
    print("=" * 70, flush=True)
    print("\nThis will train all models and compare their performance.", flush=True)
    if include_lstm:
        print("Estimated time: 10-20 minutes total (LSTM takes longer)", flush=True)
    else:
        print("Estimated time: 5-10 minutes total", flush=True)
    print("=" * 70, flush=True)

    input("\nPress ENTER to continue or Ctrl+C to cancel...")

    all_metrics = {}

    # Train XGBoost
    print("\n" + "=" * 70, flush=True)
    print("TRAINING XGBOOST MODELS", flush=True)
    print("=" * 70, flush=True)

    from models.xgboost_model import XGBoostTrainer
    xgb_trainer = XGBoostTrainer()
    all_metrics['xgboost'] = xgb_trainer.train_all_models()
    del xgb_trainer
    gc.collect()

    # Train LightGBM
    print("\n" + "=" * 70, flush=True)
    print("TRAINING LIGHTGBM MODELS", flush=True)
    print("=" * 70, flush=True)

    from models.lightgbm_model import LightGBMTrainer
    lgb_trainer = LightGBMTrainer()
    all_metrics['lightgbm'] = lgb_trainer.train_all_models()
    del lgb_trainer
    gc.collect()

    # Train LSTM (optional)
    if include_lstm:
        print("\n" + "=" * 70, flush=True)
        print("TRAINING LSTM MODELS", flush=True)
        print("=" * 70, flush=True)

        # Force cleanup of any lingering multiprocessing resources
        gc.collect()

        # Run LSTM training in a subprocess to avoid semaphore conflicts
        # between LightGBM's loky workers and PyTorch
        import subprocess
        import json
        import tempfile

        print("Running LSTM in isolated subprocess...", flush=True)

        results_file = Path(__file__).parent / "lstm_results.json"
        project_dir = str(Path(__file__).parent)

        lstm_script = f'''
import sys
import json
from pathlib import Path
sys.path.insert(0, "{project_dir}")

from models.lstm_model import LSTMTrainer
trainer = LSTMTrainer()
metrics = trainer.train_all_models()

# Convert numpy values to Python floats for JSON serialization
result = {{}}
for target, target_metrics in metrics.items():
    result[target] = {{}}
    for split, split_metrics in target_metrics.items():
        result[target][split] = {{k: float(v) for k, v in split_metrics.items()}}

with open("{results_file}", "w") as f:
    json.dump(result, f)
print("LSTM training complete, results saved.")
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(lstm_script)
            temp_script = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                text=True,
                cwd=project_dir,
                timeout=1800  # 30 minute timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"LSTM training failed with exit code {result.returncode}")

            # Load results
            with open(results_file, 'r') as f:
                all_metrics['lstm'] = json.load(f)
            os.unlink(results_file)
        finally:
            if os.path.exists(temp_script):
                os.unlink(temp_script)

        gc.collect()

    # Compare results
    print("\n" + "=" * 70, flush=True)
    print(" COMPARISON RESULTS", flush=True)
    print("=" * 70, flush=True)

    model_names = list(all_metrics.keys())
    targets = ['points', 'rebounds', 'assists']

    # Print header
    header = f"{'Metric':<15}"
    for model in model_names:
        header += f" {model.upper():<12}"
    header += " Winner"
    print(f"\nTest Set MAE Comparison:", flush=True)
    print(header, flush=True)
    print("-" * (15 + 13 * len(model_names) + 10), flush=True)

    winners = {model: 0 for model in model_names}

    for target in targets:
        row = f"{target.capitalize():<15}"
        maes = {}

        for model in model_names:
            mae = all_metrics[model][target]['test']['mae']
            maes[model] = mae
            row += f" {mae:<12.3f}"

        # Find winner (lowest MAE)
        winner = min(maes, key=maes.get)
        winners[winner] += 1
        row += f" {winner.upper()}"

        print(row, flush=True)

    print("-" * (15 + 13 * len(model_names) + 10), flush=True)

    # Overall winner
    print("\n" + "=" * 70, flush=True)
    max_wins = max(winners.values())
    overall_winners = [m for m, w in winners.items() if w == max_wins]

    if len(overall_winners) == 1:
        print(f" OVERALL WINNER: {overall_winners[0].upper()}", flush=True)
        print(f" Won {max_wins}/{len(targets)} metrics", flush=True)
    else:
        print(f" RESULT: Tie between {', '.join([m.upper() for m in overall_winners])}", flush=True)
        print(f" Each won {max_wins}/{len(targets)} metrics", flush=True)
    print("=" * 70, flush=True)

    # Detailed comparison table
    print("\n" + "=" * 70, flush=True)
    print(" DETAILED METRICS (Test Set)", flush=True)
    print("=" * 70, flush=True)

    for target in targets:
        print(f"\n{target.upper()}:", flush=True)
        print(f"  {'Model':<12} {'MAE':<10} {'RMSE':<10} {'R2':<10}", flush=True)
        print(f"  {'-'*40}", flush=True)
        for model in model_names:
            metrics = all_metrics[model][target]['test']
            print(f"  {model.upper():<12} {metrics['mae']:<10.3f} {metrics['rmse']:<10.3f} {metrics['r2']:<10.3f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" MODEL AVAILABILITY", flush=True)
    print("=" * 70, flush=True)
    print("\nAll models are now available.", flush=True)
    print("By default, predictions will use XGBoost.", flush=True)
    print("\nTo use a specific model:", flush=True)
    print("  predictor = GamePredictor(model_version='v1.0.0_...')        # XGBoost", flush=True)
    print("  predictor = GamePredictor(model_version='lightgbm_v1.0.0_...') # LightGBM", flush=True)
    if include_lstm:
        print("  predictor = GamePredictor(model_version='lstm_v1.0.0_...')    # LSTM", flush=True)

    return all_metrics


def compare_tree_models_only():
    """Compare only XGBoost and LightGBM (faster)"""
    return compare_models(include_lstm=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare NBA prediction models')
    parser.add_argument('--no-lstm', action='store_true',
                        help='Skip LSTM training (faster comparison)')
    args = parser.parse_args()

    try:
        compare_models(include_lstm=not args.no_lstm)
    except KeyboardInterrupt:
        print("\n\nComparison cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nComparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
