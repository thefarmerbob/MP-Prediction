# TensorFlow 2.x LSTM Time Series Prediction with Optuna Optimization (Quick Version)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.integration import TFKerasPruningCallback
import joblib
import os
from util import preprocess, bucket_avg

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def get_rnn_data(N_rows, bucket_size):
    """Get and preprocess data for LSTM training"""
    parse_dates = [['Date', 'Time']]
    filename = "household_power_consumption.txt"
    df = preprocess(N_rows, parse_dates, filename)
    df = pd.DataFrame(bucket_avg(df["Global_active_power"], bucket_size))
    df.dropna(inplace=True)
    return df.Global_active_power.values

def create_sequences(data, window_size, prediction_steps=1):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - window_size - prediction_steps + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size:i + window_size + prediction_steps])
    return np.array(X), np.array(y)

def build_lstm_model(trial, window_size, num_features=1):
    """Build LSTM model with Optuna hyperparameters"""
    
    # Hyperparameters to optimize
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)  # Reduced range
    num_units_1 = trial.suggest_int('num_units_1', 64, 128, step=32)  # Reduced range
    num_units_2 = trial.suggest_int('num_units_2', 32, 96, step=32) if num_lstm_layers > 1 else 0
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    dense_units = trial.suggest_int('dense_units', 32, 64, step=16)  # Reduced range
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Build model architecture
    model = keras.Sequential()
    
    # First LSTM layer
    if num_lstm_layers == 1:
        model.add(layers.LSTM(num_units_1, input_shape=(window_size, num_features)))
    else:
        model.add(layers.LSTM(num_units_1, return_sequences=True, input_shape=(window_size, num_features)))
    
    model.add(layers.Dropout(dropout_rate))
    
    # Additional LSTM layer
    if num_lstm_layers >= 2:
        model.add(layers.LSTM(num_units_2))
        model.add(layers.Dropout(dropout_rate))
    
    # Dense layers
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(1))
    
    # Compile model
    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)  # Use legacy optimizer for M1 Mac
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters
    window_size = trial.suggest_int('window_size', 50, 120, step=10)
    batch_size = trial.suggest_int('batch_size', 32, 64, step=16)
    
    # Load and preprocess data
    data = get_rnn_data(12000, "15T")  # Slightly reduced data size for speed
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(data_scaled, window_size, 1)
    
    if len(X) < 200:  # Ensure we have enough data
        return float('inf')
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # Build model
    model = build_lstm_model(trial, window_size, num_features=1)
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Optuna pruning callback
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
    
    # Train model
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,  # Reduced epochs for speed
            batch_size=batch_size,
            callbacks=[early_stopping, pruning_callback],
            verbose=0
        )
        
        # Return validation loss (what we want to minimize)
        return min(history.history['val_loss'])
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

def plot_optimization_history(study, save_path='quick_optuna_optimization_history.png'):
    """Plot optimization history"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Optimization history
    plt.subplot(1, 2, 1)
    trials = study.trials
    values = [trial.value for trial in trials if trial.value is not None and trial.value != float('inf')]
    plt.plot(values)
    plt.title('Optimization History')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value (Validation Loss)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Parameter importance
    plt.subplot(1, 2, 2)
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        importances = list(importance.values())
        
        plt.barh(params, importances)
        plt.title('Parameter Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
    except:
        plt.text(0.5, 0.5, 'Parameter importance\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_best_model(best_params, forecast_steps=200):
    """Train final model with best parameters and make predictions"""
    
    print("Training final model with best parameters...")
    print("Best parameters:", best_params)
    
    # Load data
    data = get_rnn_data(18000, "15T")
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Use best window size
    window_size = best_params['window_size']
    X, y = create_sequences(data_scaled, window_size, 1)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create a mock trial with best parameters for model building
    class MockTrial:
        def __init__(self, params):
            self.params = params
        
        def suggest_int(self, name, low, high, step=1):
            return self.params.get(name, low)
        
        def suggest_float(self, name, low, high, log=False):
            return self.params.get(name, low)
    
    mock_trial = MockTrial(best_params)
    model = build_lstm_model(mock_trial, window_size, num_features=1)
    
    print(model.summary())
    
    # Train final model
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
    
    # Generate validation predictions
    val_predictions_scaled = model.predict(X_val, verbose=0).flatten()
    val_predictions_original = scaler.inverse_transform(val_predictions_scaled.reshape(-1, 1)).flatten()
    
    # Make forecasts
    print(f"Making {forecast_steps} step forecasts...")
    last_sequence = data_scaled[-window_size:]
    forecasts = []
    
    for _ in range(forecast_steps):
        current_sequence = last_sequence.reshape(1, window_size, 1)
        pred = model.predict(current_sequence, verbose=0)
        forecasts.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred[0, 0])
    
    # Transform back to original scale
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_original = scaler.inverse_transform(forecasts).flatten()
    data_original = scaler.inverse_transform(data_scaled.reshape(-1, 1)).flatten()
    
    # Plot results
    plot_predictions(data_original, train_size, val_size, val_predictions_original, 
                    forecasts_original, forecast_steps, 'quick_optimized_lstm_predictions.png')
    
    return model, history, forecasts_original

def plot_predictions(observed, train_size, val_size, val_predictions, predictions, 
                    forecast_steps=200, filename='lstm_predict_result.png'):
    """Plot training, validation, and forecast results"""
    plt.figure(figsize=(15, 8))
    
    # Plot observed data
    plt.plot(range(len(observed)), observed, label="Observed", color="k", alpha=0.7)
    
    # Plot training/validation split
    plt.axvline(train_size, linestyle="--", color="blue", alpha=0.5, label="Train/Val Split")
    plt.axvline(train_size + val_size, linestyle="--", color="green", alpha=0.5, label="Val/Test Split")
    
    # Plot validation predictions
    if len(val_predictions) > 0:
        val_start = train_size
        val_x = range(val_start, val_start + len(val_predictions))
        plt.plot(val_x, val_predictions, label="Validation Predictions", color="g", linewidth=2, alpha=0.8)
    
    # Plot forecasts
    if len(predictions) > 0:
        pred_start = train_size + val_size
        pred_x = range(pred_start, pred_start + len(predictions))
        plt.plot(pred_x, predictions, label="Forecasts", color="r", linewidth=2)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Global Active Power (kW)")
    plt.title("Optimized LSTM Time Series Prediction - Household Power Consumption")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    print("Starting Quick Optuna hyperparameter optimization for LSTM...")
    print("Dataset: Household Power Consumption")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # Run optimization
    print("Running optimization trials (20 trials, 30 min timeout)...")
    study.optimize(objective, n_trials=20, timeout=1800)  # 30 minutes timeout
    
    print("Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (validation loss): {study.best_value:.6f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study
    joblib.dump(study, 'quick_optuna_study.pkl')
    print("Study saved to 'quick_optuna_study.pkl'")
    
    # Plot optimization history
    plot_optimization_history(study)
    print("Optimization history saved to 'quick_optuna_optimization_history.png'")
    
    # Train final model with best parameters
    best_model, history, forecasts = train_best_model(study.best_params)
    
    # Save best model
    best_model.save('quick_best_lstm_model.h5')
    print("Best model saved to 'quick_best_lstm_model.h5'")
    
    print("\nOptimization and final training completed!")
    print("Files created:")
    print("- quick_optuna_study.pkl: Saved optimization study")
    print("- quick_optuna_optimization_history.png: Optimization visualization")
    print("- quick_optimized_lstm_predictions.png: Final model predictions")
    print("- quick_best_lstm_model.h5: Best trained model")
    
    print(f"\nFinal model performance:")
    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"Forecast range: {forecasts.min():.3f} to {forecasts.max():.3f} kW")

if __name__ == '__main__':
    main() 