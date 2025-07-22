# TensorFlow 2.x LSTM Time Series Prediction
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from util import preprocess, bucket_avg

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_rnn_data(N_rows, bucket_size):
    """Get and preprocess data for LSTM training"""
    parse_dates = [['Date', 'Time']]
    filename = "household_power_consumption.txt"
    df = preprocess(N_rows, parse_dates, filename)
    df = pd.DataFrame(bucket_avg(df["Global_active_power"], bucket_size))
    df.dropna(inplace=True)
    return df.Global_active_power.values

def create_sequences_with_step(data, window_size, prediction_steps=1, step=1):
    """Create sequences for LSTM training with custom step size"""
    X, y = [], []
    for i in range(0, len(data) - window_size - prediction_steps + 1, step):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size:i + window_size + prediction_steps])
    return np.array(X), np.array(y)

def build_lstm_model(window_size, num_features=1):
    """Build optimized LSTM model with best parameters from Optuna Trial 1"""
    
    # Best parameters from Trial 1
    learning_rate = 3.4146857120481125e-05
    beta_1 = 0.8656795616453691
    beta_2 = 0.9483057587011294
    epsilon = 3.197170431201194e-07
    dropout_rate = 0.22126514822331714
    recurrent_dropout = 0.29766043798512826
    l1_rate = 1.8855347677212793e-06
    l2_rate = 0.0008154837696146691
    
    # Create L1L2 regularizer
    kernel_regularizer = keras.regularizers.L1L2(l1=l1_rate, l2=l2_rate)
    
    model = keras.Sequential([
        # Single LSTM layer with 96 units (optimized architecture)
        layers.LSTM(96, 
                   activation='relu',
                   recurrent_activation='sigmoid',
                   use_bias=True,
                   kernel_initializer='random_normal',
                   recurrent_initializer='orthogonal',
                   unit_forget_bias=True,
                   dropout=dropout_rate,
                   recurrent_dropout=recurrent_dropout,
                   kernel_regularizer=kernel_regularizer,
                   input_shape=(window_size, num_features)),
        
        # Batch normalization
        layers.BatchNormalization(),
        
        # Three dense layers with ELU activation (optimized configuration)
        layers.Dense(48, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(8, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1)
    ])
    
    # Compile model with optimized Adam parameters
    optimizer = keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def plot_predictions(observed, train_size, val_size, val_predictions, predictions, forecast_steps=200):
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
    plt.ylabel("Global Active Power")
    plt.title("Optimized LSTM Time Series Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_predict_result.png', dpi=300)
    plt.close()

def main():
    # Configuration - Optimal parameters from Trial 1
    window_size = 90  # Optimized window size
    forecast_steps = 200
    batch_size = 56  # Optimized batch size
    epochs = 50  # Optimized epochs
    sequence_step = 2  # Data augmentation step
    
    # Data split ratios (optimized)
    train_ratio = 0.7479599614961582
    val_ratio = 0.16048404045018172
    
    print("🚀 Loading and preprocessing data with OPTIMAL parameters...")
    print(f"📊 Configuration: window_size={window_size}, batch_size={batch_size}, epochs={epochs}")
    
    # Load data
    data = get_rnn_data(18000, "15T")
    print(f"Data shape: {data.shape}")
    
    # Normalize data with Standard Scaler (optimized)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Create sequences with optimized step size
    X, y = create_sequences_with_step(data_scaled, window_size, 1, sequence_step)
    print(f"Sequence shape: X={X.shape}, y={y.shape}")
    
    # Split data with optimized ratios
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    
    # Ensure we don't exceed data length
    if train_size + val_size >= len(X):
        val_size = len(X) - train_size - 1
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build optimized model
    print("🏗️ Building optimized LSTM model...")
    model = build_lstm_model(window_size, num_features=1)
    print(model.summary())
    
    # Optimized callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,  # Optimized patience
        min_delta=1.4426565023804986e-06,  # Optimized min_delta
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau (optimized)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.17558850066697557,  # Optimized factor
        patience=2,  # Optimized patience
        min_lr=5.614059649118678e-06,  # Optimized min_lr
        verbose=1
    )
    
    # Train model with optimized parameters
    print("🎯 Training optimized model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=True
    )
    
    # Evaluate model
    print("📈 Evaluating model...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Generate validation predictions for plotting
    print("🔮 Generating validation predictions...")
    val_predictions_scaled = model.predict(X_val, verbose=0).flatten()
    val_predictions_original = scaler.inverse_transform(val_predictions_scaled.reshape(-1, 1)).flatten()
    
    # Make predictions for forecasting
    print(f"📊 Making {forecast_steps} step forecasts...")
    
    # Use the last window_size points for forecasting
    last_sequence = data_scaled[-window_size:]
    forecasts = []
    
    for _ in range(forecast_steps):
        # Reshape for prediction
        current_sequence = last_sequence.reshape(1, window_size, 1)
        
        # Predict next value
        pred = model.predict(current_sequence, verbose=0)
        forecasts.append(pred[0, 0])
        
        # Update sequence (remove first element, add prediction)
        last_sequence = np.append(last_sequence[1:], pred[0, 0])
    
    # Transform predictions back to original scale
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_original = scaler.inverse_transform(forecasts).flatten()
    
    # Transform original data back for plotting
    data_original = scaler.inverse_transform(data_scaled.reshape(-1, 1)).flatten()
    
    print(f"Forecast range: {forecasts_original.min():.3f} to {forecasts_original.max():.3f}")
    
    # Plot results
    plot_predictions(data_original, train_size, val_size, val_predictions_original, forecasts_original, forecast_steps)
    
    # Print training history
    print("\n📊 Training History:")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    
    # Print optimization summary
    print("\n🏆 OPTIMIZATION SUMMARY:")
    print("="*50)
    print("✅ Architecture: 1 LSTM layer (96 units) + 3 dense layers (48→128→8)")
    print("✅ Regularization: L1/L2 + Batch Norm + Dropout")
    print("✅ Optimizer: Adam with optimized parameters")
    print("✅ Data: Standard scaler + sequence step=2")
    print("✅ Training: 50 epochs, batch size 56")
    print("✅ Expected validation loss: ~0.176 (from Optuna Trial 1)")
    print("="*50)
    
    print(f"\n🎉 Forecasting completed! Results saved to 'lstm_predict_result.png'")

if __name__ == '__main__':
    main()
