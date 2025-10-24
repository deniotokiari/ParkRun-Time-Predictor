#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
import os
import pickle
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to reduce warnings and improve performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(False)
tf.config.optimizer.set_jit(True)

OUTPUT_FILE = "output/park_run_results.csv"
MODEL_SAVE_PATH = "models/parkrun_model.keras"
SCALER_SAVE_PATH = "models/scalers.pkl"
DISTANCE_KM = 5.0

class ParkRunPredictor:
    def __init__(self, data_file: str = OUTPUT_FILE):
        self.data_file = data_file
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        print(f"Loading data from {self.data_file}...")
        df = pd.read_csv(self.data_file)
        print(f"Loaded {len(df)} records")
        
        required_columns = ['event_id', 'position', 'time', 'month', 'participants']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def add_relative_position(self, df: pd.DataFrame) -> pd.DataFrame:
        df['relative_position'] = (df['position'] - 1) / (df['participants'] - 1)
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df[['relative_position', 'month', 'participants']].values
        y = df['time'].values.reshape(-1, 1)
        
        if np.any(np.isnan(X)):
            nan_count = np.isnan(X).sum()
            raise ValueError(f"Feature matrix contains {nan_count} NaN values")
        
        if np.any(np.isnan(y)):
            nan_count = np.isnan(y).sum()
            raise ValueError(f"Target vector contains {nan_count} NaN values")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, save_model: bool = True) -> dict:
        print("Training model...")
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        self.model = models.Sequential([
            layers.Input(shape=(3,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='swish'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse'],
            run_eagerly=False  # Use graph execution for better performance
        )
        
        # Configure TensorFlow to reduce retracing
        tf.config.run_functions_eagerly(False)
        tf.config.optimizer.set_jit(True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=0
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        if save_model:
            os.makedirs('models', exist_ok=True)
            checkpoint = ModelCheckpoint(
                MODEL_SAVE_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            callbacks.append(checkpoint)
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=300,
            batch_size=64,
            verbose=1,
            callbacks=callbacks,
            shuffle=True
        )
        
        y_pred = self.model.predict(X_test)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        y_test_original = self.scaler_y.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        
        print(f"Model trained successfully!")
        print(f"Test MAE: {mae:.1f} seconds")
        print(f"Test MSE: {mse:.1f} seconds²")
        
        if save_model:
            with open(SCALER_SAVE_PATH, 'wb') as f:
                pickle.dump((self.scaler_X, self.scaler_y), f)
            # Save data hash for future freshness checks
            self._save_data_hash()
        
        print(f"\nTraining Summary:")
        print(f"  Final Learning Rate: {self.model.optimizer.learning_rate.numpy():.2e}")
        print(f"  Training Epochs: {len(history.history['loss'])}")
        print(f"  Best Validation Loss: {min(history.history['val_loss']):.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2_score(y_test_original, y_pred_original),
            'history': history.history
        }
    
    def load_model(self) -> bool:
        model_paths = [MODEL_SAVE_PATH, "models/parkrun_model.h5"]
        model_path = None
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path or not os.path.exists(SCALER_SAVE_PATH):
            return False
        
        try:
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }
            
            self.model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            with open(SCALER_SAVE_PATH, 'rb') as f:
                self.scaler_X, self.scaler_y = pickle.load(f)
            print(f"Loaded pre-trained model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will retrain model instead...")
            self._cleanup_corrupted_model()
            return False
    
    def _cleanup_corrupted_model(self):
        try:
            model_files = [MODEL_SAVE_PATH, "models/parkrun_model.h5"]
            for model_file in model_files:
                if os.path.exists(model_file):
                    os.remove(model_file)
                    print(f"Removed corrupted model file: {model_file}")
            
            if os.path.exists(SCALER_SAVE_PATH):
                os.remove(SCALER_SAVE_PATH)
                print("Removed corrupted scaler file")
        except Exception as e:
            print(f"Could not clean up files: {e}")
    
    def predict(self, position: int, month: int = None) -> dict:
        if position < 1:
            raise ValueError("Position must be >= 1")
        
        if month is None:
            from datetime import datetime, timedelta
            today = datetime.now()
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_saturday = today + timedelta(days_ahead)
            month = next_saturday.month
        
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")
        
        median_participants = self.df['participants'].median()
        
        if pd.isna(median_participants) or median_participants <= 1:
            raise ValueError(f"Invalid participants data: {median_participants}")
        
        # Prevent division by zero
        if median_participants <= 1:
            relative_position = 0.0 if position == 1 else 1.0
        else:
            relative_position = (position - 1) / (median_participants - 1)
        
        # Clamp relative position to valid range [0, 1]
        relative_position = max(0.0, min(1.0, relative_position))
        
        if pd.isna(relative_position):
            raise ValueError(f"Invalid relative position: {relative_position}")
        
        # Ensure consistent input types and shapes
        input_data = np.array([[float(relative_position), float(month), float(median_participants)]], dtype=np.float32)
        
        if np.any(np.isnan(input_data)):
            raise ValueError(f"Input contains NaN values: {input_data}")
        
        input_scaled = self.scaler_X.transform(input_data)
        
        # Use consistent prediction with explicit shape and caching
        pred_scaled = self._predict_cached(input_scaled)
        pred_seconds = float(self.scaler_y.inverse_transform(pred_scaled)[0][0])
        
        pace_min_per_km = (pred_seconds / 60) / DISTANCE_KM
        
        return {
            'time_seconds': pred_seconds,
            'time_minutes': pred_seconds / 60,
            'pace_min_per_km': pace_min_per_km,
            'position': position,
            'month': month,
            'participants': median_participants
        }
    
    def _predict_cached(self, input_scaled):
        """Cached prediction function to avoid retracing."""
        # Use model.predict with consistent parameters to avoid retracing
        return self.model.predict(input_scaled, verbose=0, batch_size=1)
    
    def check_data_freshness(self) -> bool:
        if not os.path.exists(MODEL_SAVE_PATH):
            print("No model found, will train new model")
            return True
        
        # Check if data hash file exists
        hash_file = MODEL_SAVE_PATH.replace('.keras', '_data_hash.txt')
        if not os.path.exists(hash_file):
            print("No data hash found, will retrain model")
            return True
        
        # Calculate current data hash
        current_hash = self._calculate_data_hash()
        
        # Read stored hash
        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if current_hash != stored_hash:
                print("Data content changed, retraining model")
                return True
            else:
                print("Data content unchanged, using existing model")
                return False
        except:
            print("Error reading data hash, will retrain model")
            return True
    
    def _calculate_data_hash(self) -> str:
        """Calculate hash of the actual data content, not file modification time."""
        import hashlib
        
        if not os.path.exists(self.data_file):
            return ""
        
        # Read the CSV file and calculate hash of content
        with open(self.data_file, 'rb') as f:
            content = f.read()
        
        # Calculate MD5 hash of the file content
        return hashlib.md5(content).hexdigest()
    
    def _save_data_hash(self) -> None:
        """Save the current data hash for future comparison."""
        hash_file = MODEL_SAVE_PATH.replace('.keras', '_data_hash.txt')
        current_hash = self._calculate_data_hash()
        
        with open(hash_file, 'w') as f:
            f.write(current_hash)
    
    def run_full_pipeline(self, retrain: bool = False) -> None:
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} records")
        except Exception as e:
            raise FileNotFoundError(f"Could not load data file {self.data_file}: {e}")
        
        if len(self.df) == 0:
            raise ValueError("Data file is empty")
        
        initial_count = len(self.df)
        self.df = self.df[self.df['time'] != '']
        self.df = self.df[self.df['time'].notna()]
        self.df = self.df[self.df['position'] > 0]
        self.df = self.df[self.df['participants'] > 0]
        
        print(f"Cleaned data: removed {initial_count - len(self.df)} invalid records")
        print(f"Remaining records: {len(self.df)}")
        
        if len(self.df) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        # Prevent division by zero in relative position calculation
        self.df = self.df[self.df['participants'] > 1]  # Remove single-participant events
        if len(self.df) == 0:
            raise ValueError("No valid data with multiple participants")
        
        self.df['relative_position'] = (self.df['position'] - 1) / (self.df['participants'] - 1)
        
        data_updated = self.check_data_freshness()
        
        if retrain or data_updated:
            if data_updated:
                print("🔄 Data file has been updated since last training. Retraining model...")
            else:
                print("🔄 Force retraining requested...")
            
            X, y = self.prepare_features(self.df)
            self.train_model(X, y)
        else:
            if self.load_model() and self.model is not None:
                print("✅ Using existing model (data is up to date)")
            else:
                print("⚠️ Could not load existing model, training new one...")
                X, y = self.prepare_features(self.df)
                self.train_model(X, y)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Predict parkrun finish time based on position and month',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python park_run_speed_predict.py 5
  python park_run_speed_predict.py 10 --month 6
  python park_run_speed_predict.py 1 --retrain
        """
    )
    
    parser.add_argument('position', type=int,
                       help='Position in race (must be >= 1)')
    parser.add_argument('--month', type=int, default=None,
                       help='Month (1-12). If not provided, uses nearest Saturday')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retrain the model')
    parser.add_argument('--data-file', default=OUTPUT_FILE,
                       help=f'Path to CSV data file (default: {OUTPUT_FILE})')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show data statistics and exit')
    
    args = parser.parse_args()
    
    if args.position < 1:
        parser.error("Position must be >= 1")
    if args.month is not None and (args.month < 1 or args.month > 12):
        parser.error("Month must be between 1 and 12")
    
    return args

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_pace(pace_min_per_km: float) -> str:
    minutes = int(pace_min_per_km)
    seconds = int((pace_min_per_km - minutes) * 60)
    return f"{minutes}:{seconds:02d} min/km"

def show_data_statistics(data_file: str):
    try:
        df = pd.read_csv(data_file)
        print(f"\n📊 DATA STATISTICS")
        print("="*50)
        print(f"Total records: {len(df):,}")
        print(f"Unique events: {df['event_id'].nunique():,}")
        print(f"Event range: {df['event_id'].min()} to {df['event_id'].max()}")
        print(f"Month distribution:")
        month_counts = df['month'].value_counts().sort_index()
        for month, count in month_counts.items():
            print(f"  Month {month}: {count:,} records")
        print(f"Participants statistics:")
        print(f"  Min: {df['participants'].min():.0f}")
        print(f"  Max: {df['participants'].max():.0f}")
        print(f"  Median: {df['participants'].median():.0f}")
        print(f"Time range: {df['time'].min()} to {df['time'].max()} seconds")
        print(f"Position range: {df['position'].min()} to {df['position'].max()}")
        
        if os.path.exists(MODEL_SAVE_PATH):
            data_mtime = os.path.getmtime(data_file)
            model_mtime = os.path.getmtime(MODEL_SAVE_PATH)
            if data_mtime > model_mtime:
                print(f"\n⚠️  Data file is newer than model. Consider retraining with --retrain")
            else:
                print(f"\n✅ Model is up to date with data")
        else:
            print(f"\n🆕 No model found. Will train on first run.")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error reading data file: {e}")

def main():
    args = parse_arguments()
    
    try:
        if args.show_stats:
            show_data_statistics(args.data_file)
            return
        
        predictor = ParkRunPredictor(args.data_file)
        predictor.run_full_pipeline(retrain=args.retrain)
        result = predictor.predict(args.position, args.month)
        
        print("\n" + "="*50)
        print("🏃‍♂️ PARKRUN TIME PREDICTION")
        print("="*50)
        print(f"Input Parameters:")
        print(f"  Position: {result['position']}")
        print(f"  Month: {result['month']}")
        print(f"  Participants (median): {result['participants']:.0f}")
        print()
        print(f"Predicted Results:")
        print(f"  Finish Time: {format_time(result['time_seconds'])} ({result['time_minutes']:.1f} minutes)")
        print(f"  Pace: {format_pace(result['pace_min_per_km'])}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()