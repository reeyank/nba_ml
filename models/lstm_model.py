"""
LSTM Model Training for NBA Player Stat Prediction
Trains sequence-based LSTM models for points, rebounds, and assists prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_manager import DatabaseManager
from config import (
    TRAIN_VALIDATION_SPLIT_DATE,
    VALIDATION_TEST_SPLIT_DATE,
    TARGET_VARIABLES,
    MODELS_DIR,
    LSTM_PARAMS
)


class PlayerSequenceDataset(Dataset):
    """PyTorch Dataset for player game sequences"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        # Keep as numpy, convert to tensor on-the-fly (more memory efficient)
        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])


class PlayerLSTM(nn.Module):
    """LSTM model for predicting player stats from game sequences"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last timestep output
        out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, 1)
        return out.squeeze(-1)


class LSTMTrainer:
    """Trains LSTM models for player stat prediction"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}", flush=True)

    def load_game_logs(self, seasons: list = None) -> pd.DataFrame:
        """Load raw game logs for sequence building"""
        print("Loading game log data...", flush=True)
        game_logs = self.db.get_all_player_game_logs(seasons)
        print(f"Loaded {len(game_logs):,} game records", flush=True)
        return game_logs

    def get_sequence_features(self) -> list:
        """Get features to use for LSTM sequences (per-game raw features)"""
        # Use raw per-game stats - let LSTM learn the patterns
        return [
            'minutes',
            'points',
            'rebounds',
            'assists',
            'steals',
            'blocks',
            'turnovers',
            'field_goal_pct',
            'three_point_pct',
            'free_throw_pct',
            'is_home',
            'rest_days'
        ]

    def prepare_game_logs(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        """Prepare game logs with necessary features"""
        df = game_logs.copy()

        # Sort by player and date
        df = df.sort_values(['player_id', 'game_date'])

        # Parse is_home from matchup
        df['is_home'] = df['matchup'].apply(lambda x: 1 if 'vs.' in str(x) else 0)

        # Calculate rest days
        df['game_date_dt'] = pd.to_datetime(df['game_date'])
        df['rest_days'] = df.groupby('player_id')['game_date_dt'].diff().dt.days.fillna(3)
        df['rest_days'] = df['rest_days'].clip(0, 10)  # Cap at 10 days

        # Fill NaN in percentage columns with 0
        pct_cols = ['field_goal_pct', 'three_point_pct', 'free_throw_pct']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def build_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int = 10,
        target: str = 'points'
    ) -> tuple:
        """
        Build sequences for each player

        For each game, create a sequence of the prior N games
        Target is the stat in the current game
        """
        feature_cols = self.get_sequence_features()
        self.feature_columns = feature_cols

        sequences = []
        targets = []
        game_dates = []
        player_ids = []

        # Group by player
        for player_id, player_df in tqdm(df.groupby('player_id'), desc=f"Building sequences for {target}"):
            player_df = player_df.sort_values('game_date').reset_index(drop=True)

            # Need at least seq_len + 1 games
            if len(player_df) <= seq_len:
                continue

            # Build sequences
            for i in range(seq_len, len(player_df)):
                # Sequence: games [i-seq_len : i]
                seq_df = player_df.iloc[i-seq_len:i]

                # Get feature values
                seq_features = seq_df[feature_cols].values

                # Target: stat in game i
                target_val = player_df.iloc[i][target]

                # Store game date for splitting
                game_date = player_df.iloc[i]['game_date']

                sequences.append(seq_features)
                targets.append(target_val)
                game_dates.append(game_date)
                player_ids.append(player_id)

        return np.array(sequences), np.array(targets), game_dates, player_ids

    def split_by_date(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        game_dates: list
    ) -> tuple:
        """Split data chronologically"""
        game_dates = np.array(game_dates)

        train_mask = game_dates < TRAIN_VALIDATION_SPLIT_DATE
        val_mask = (game_dates >= TRAIN_VALIDATION_SPLIT_DATE) & (game_dates < VALIDATION_TEST_SPLIT_DATE)
        test_mask = game_dates >= VALIDATION_TEST_SPLIT_DATE

        return (
            (sequences[train_mask], targets[train_mask]),
            (sequences[val_mask], targets[val_mask]),
            (sequences[test_mask], targets[test_mask])
        )

    def scale_sequences(
        self,
        train_seq: np.ndarray,
        val_seq: np.ndarray,
        test_seq: np.ndarray,
        target: str
    ) -> tuple:
        """Scale sequences using StandardScaler (fit on train only)"""
        print("Scaling sequences...", flush=True)
        n_train, seq_len, n_features = train_seq.shape
        n_val = val_seq.shape[0]
        n_test = test_seq.shape[0]

        scaler = StandardScaler()

        # Fit on training data (use partial_fit for memory efficiency)
        print(f"  Fitting scaler on {n_train:,} training sequences...", flush=True)
        train_flat = train_seq.reshape(-1, n_features)
        scaler.fit(train_flat)

        # Transform all sets
        print(f"  Transforming training set...", flush=True)
        train_scaled = scaler.transform(train_flat).reshape(n_train, seq_len, n_features)
        del train_flat  # Free memory

        print(f"  Transforming validation set...", flush=True)
        val_scaled = scaler.transform(val_seq.reshape(-1, n_features)).reshape(n_val, seq_len, n_features)

        print(f"  Transforming test set...", flush=True)
        test_scaled = scaler.transform(test_seq.reshape(-1, n_features)).reshape(n_test, seq_len, n_features)

        # Store scaler for this target
        self.scalers[target] = scaler
        print("  Done scaling.", flush=True)

        return train_scaled, val_scaled, test_scaled

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_size: int,
        target: str
    ) -> nn.Module:
        """Train LSTM model for a specific target"""
        print(f"\n{'='*60}", flush=True)
        print(f"Training LSTM for {target.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        # Initialize model
        model = PlayerLSTM(
            input_size=input_size,
            hidden_size=LSTM_PARAMS['hidden_size'],
            num_layers=LSTM_PARAMS['num_layers'],
            dropout=LSTM_PARAMS['dropout']
        ).to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_PARAMS['learning_rate'])
        loss_fn = nn.L1Loss()  # MAE for consistency with other models

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        print(f"Starting training for {LSTM_PARAMS['epochs']} epochs...", flush=True)

        # Keep data as numpy arrays - only convert to tensor at batch level
        # This avoids memory issues and semaphore conflicts
        train_X_np = train_loader.dataset.sequences  # Already numpy float32
        train_Y_np = train_loader.dataset.targets    # Already numpy float32
        val_X_np = val_loader.dataset.sequences
        val_Y_np = val_loader.dataset.targets

        batch_size = LSTM_PARAMS['batch_size']
        n_train = len(train_X_np)
        n_batches = (n_train + batch_size - 1) // batch_size

        for epoch in range(LSTM_PARAMS['epochs']):
            # Training
            model.train()
            train_losses = []

            # Shuffle training data indices
            perm = np.random.permutation(n_train)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_train)
                batch_indices = perm[start_idx:end_idx]

                # Convert batch to tensor on-the-fly
                batch_x = torch.FloatTensor(train_X_np[batch_indices]).to(self.device)
                batch_y = torch.FloatTensor(train_Y_np[batch_indices]).to(self.device)

                optimizer.zero_grad()
                preds = model(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []

            with torch.no_grad():
                n_val = len(val_X_np)
                for batch_start in range(0, n_val, batch_size):
                    batch_end = min(batch_start + batch_size, n_val)
                    batch_x = torch.FloatTensor(val_X_np[batch_start:batch_end]).to(self.device)
                    batch_y = torch.FloatTensor(val_Y_np[batch_start:batch_end]).to(self.device)

                    preds = model(batch_x)
                    loss = loss_fn(preds, batch_y)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{LSTM_PARAMS['epochs']} - "
                      f"Train MAE: {train_loss:.3f}, Val MAE: {val_loss:.3f}", flush=True)

            if patience_counter >= LSTM_PARAMS['patience']:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

        # Load best model
        model.load_state_dict(best_model_state)
        print(f"Best validation MAE: {best_val_loss:.3f}", flush=True)

        return model

    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        dataset_name: str,
        target: str
    ) -> dict:
        """Evaluate model and return metrics"""
        model.eval()
        all_preds = []
        all_targets = []

        # Use numpy arrays directly
        X_np = data_loader.dataset.sequences
        Y_np = data_loader.dataset.targets

        batch_size = LSTM_PARAMS['batch_size']
        n_samples = len(X_np)

        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_x = torch.FloatTensor(X_np[batch_start:batch_end]).to(self.device)

                preds = model(batch_x)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(Y_np[batch_start:batch_end])

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        print(f"\n{dataset_name} Set Metrics for {target}:", flush=True)
        print(f"  MAE:  {mae:.3f}", flush=True)
        print(f"  RMSE: {rmse:.3f}", flush=True)
        print(f"  R2:   {r2:.3f}", flush=True)

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def train_all_models(self) -> dict:
        """Train LSTM models for all target variables"""
        print("="*60, flush=True)
        print("LSTM MODEL TRAINING PIPELINE", flush=True)
        print("="*60, flush=True)

        # Load game logs
        game_logs = self.load_game_logs()

        # Prepare data
        print("\nPreparing game logs...", flush=True)
        prepared_df = self.prepare_game_logs(game_logs)

        all_metrics = {}

        for target in TARGET_VARIABLES:
            print(f"\n{'='*60}", flush=True)
            print(f"Processing {target.upper()}", flush=True)
            print(f"{'='*60}", flush=True)

            # Build sequences
            sequences, targets_arr, game_dates, _ = self.build_sequences(
                prepared_df,
                seq_len=LSTM_PARAMS['sequence_length'],
                target=target
            )

            print(f"Built {len(sequences):,} sequences", flush=True)

            # Split chronologically
            (train_seq, train_y), (val_seq, val_y), (test_seq, test_y) = self.split_by_date(
                sequences, targets_arr, game_dates
            )

            print(f"Train: {len(train_seq):,} | Val: {len(val_seq):,} | Test: {len(test_seq):,}", flush=True)

            # Convert to float32 to save memory
            train_seq = train_seq.astype(np.float32)
            val_seq = val_seq.astype(np.float32)
            test_seq = test_seq.astype(np.float32)

            # Scale sequences
            train_scaled, val_scaled, test_scaled = self.scale_sequences(
                train_seq, val_seq, test_seq, target
            )

            # Free memory from unscaled sequences
            del train_seq, val_seq, test_seq

            # Create data loaders
            print("Creating datasets...", flush=True)
            train_dataset = PlayerSequenceDataset(train_scaled, train_y)
            val_dataset = PlayerSequenceDataset(val_scaled, val_y)
            test_dataset = PlayerSequenceDataset(test_scaled, test_y)

            print("Creating data loaders...", flush=True)
            # num_workers=0 to avoid multiprocessing issues when called from another script
            train_loader = DataLoader(train_dataset, batch_size=LSTM_PARAMS['batch_size'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=LSTM_PARAMS['batch_size'], num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=LSTM_PARAMS['batch_size'], num_workers=0)
            print("Data loaders ready.", flush=True)

            # Train model
            model = self.train_model(
                train_loader, val_loader,
                input_size=train_scaled.shape[2],
                target=target
            )

            # Evaluate
            train_metrics = self.evaluate_model(model, train_loader, "Train", target)
            val_metrics = self.evaluate_model(model, val_loader, "Val", target)
            test_metrics = self.evaluate_model(model, test_loader, "Test", target)

            # Store model and metrics
            self.models[target] = model
            all_metrics[target] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }

            # Free memory after each target
            del train_scaled, val_scaled, test_scaled
            del train_dataset, val_dataset, test_dataset
            del train_loader, val_loader, test_loader
            import gc
            gc.collect()

        # Save models
        self.save_models(all_metrics)

        # Print summary
        self.print_summary(all_metrics)

        return all_metrics

    def save_models(self, metrics: dict):
        """Save trained LSTM models and scalers"""
        print(f"\n{'='*60}", flush=True)
        print("SAVING MODELS", flush=True)
        print(f"{'='*60}", flush=True)

        # Create model version
        version = f"lstm_v1.0.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = MODELS_DIR / version
        version_dir.mkdir(exist_ok=True)

        # Save each model
        for target, model in self.models.items():
            # Save PyTorch model
            model_path = version_dir / f"{target}_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': model.lstm.input_size,
                'hidden_size': model.lstm.hidden_size,
                'num_layers': model.lstm.num_layers
            }, model_path)
            print(f"Saved {target} model to {model_path}", flush=True)

            # Save scaler
            scaler_path = version_dir / f"{target}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[target], f)
            print(f"Saved {target} scaler to {scaler_path}", flush=True)

        # Save feature columns
        feature_path = version_dir / "feature_columns.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Saved feature columns to {feature_path}", flush=True)

        # Save model type indicator
        model_type_path = version_dir / "model_type.txt"
        with open(model_type_path, 'w') as f:
            f.write('lstm')

        # Save LSTM config
        config_path = version_dir / "lstm_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(LSTM_PARAMS, f)
        print(f"Saved LSTM config to {config_path}", flush=True)

        # Save metadata to database
        try:
            metadata = {
                'model_version': version,
                'model_type': 'lstm',
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'model_path': str(version_dir),
                'is_active': 0  # Don't auto-activate LSTM
            }
            self.db.insert_model_metadata(metadata)
            print(f"Saved metadata to database (version: {version})", flush=True)
        except Exception as e:
            print(f"Could not save metadata to database: {e}", flush=True)

        return version

    def print_summary(self, metrics: dict):
        """Print final summary of all models"""
        print(f"\n{'='*60}", flush=True)
        print("LSTM TRAINING SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)

        print("\nTest Set Performance:", flush=True)
        print(f"{'Target':<12} {'MAE':<10} {'RMSE':<10} {'R2':<10}", flush=True)
        print("-" * 42, flush=True)

        for target in TARGET_VARIABLES:
            test_metrics = metrics[target]['test']
            print(f"{target.capitalize():<12} "
                  f"{test_metrics['mae']:<10.3f} "
                  f"{test_metrics['rmse']:<10.3f} "
                  f"{test_metrics['r2']:<10.3f}", flush=True)

        # Check if targets met
        from config import TARGET_MAE
        print("\nTarget Achievement:", flush=True)
        for target in TARGET_VARIABLES:
            actual_mae = metrics[target]['test']['mae']
            target_mae = TARGET_MAE[target]
            status = "" if actual_mae <= target_mae else ""
            print(f"  {status} {target.capitalize()}: {actual_mae:.2f} (target: {target_mae:.2f})", flush=True)


def main():
    """Run LSTM training pipeline"""
    trainer = LSTMTrainer()
    metrics = trainer.train_all_models()


if __name__ == "__main__":
    main()
