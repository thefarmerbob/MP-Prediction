# TensorFlow 2.x LSTM Time Series Prediction with Optuna Optimization
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
    """Build comprehensive LSTM model with ALL possible Optuna hyperparameters"""
    
    # =============================================================================
    # ARCHITECTURE PARAMETERS
    # =============================================================================
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 4)
    num_units_1 = trial.suggest_int('num_units_1', 32, 512, step=32)
    num_units_2 = trial.suggest_int('num_units_2', 32, 512, step=32) if num_lstm_layers > 1 else 0
    num_units_3 = trial.suggest_int('num_units_3', 32, 256, step=32) if num_lstm_layers > 2 else 0
    num_units_4 = trial.suggest_int('num_units_4', 32, 128, step=32) if num_lstm_layers > 3 else 0
    
    # Dense layers configuration
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_units_1 = trial.suggest_int('dense_units_1', 16, 256, step=16)
    dense_units_2 = trial.suggest_int('dense_units_2', 8, 128, step=8) if num_dense_layers > 1 else 0
    dense_units_3 = trial.suggest_int('dense_units_3', 4, 64, step=4) if num_dense_layers > 2 else 0
    
    # =============================================================================
    # LSTM-SPECIFIC PARAMETERS
    # =============================================================================
    lstm_activation = trial.suggest_categorical('lstm_activation', ['tanh', 'relu', 'sigmoid'])
    lstm_recurrent_activation = trial.suggest_categorical('lstm_recurrent_activation', ['sigmoid', 'hard_sigmoid'])
    use_bias = trial.suggest_categorical('use_bias', [True, False])
    unit_forget_bias = trial.suggest_categorical('unit_forget_bias', [True, False])
    
    # Initializers
    kernel_initializer = trial.suggest_categorical('kernel_initializer', 
                                                 ['glorot_uniform', 'he_uniform', 'lecun_uniform', 'random_normal'])
    recurrent_initializer = trial.suggest_categorical('recurrent_initializer', 
                                                    ['orthogonal', 'glorot_uniform', 'he_uniform'])
    
    # =============================================================================
    # REGULARIZATION PARAMETERS
    # =============================================================================
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.5)
    
    # L1/L2 Regularization
    use_kernel_regularizer = trial.suggest_categorical('use_kernel_regularizer', [True, False])
    l1_rate = trial.suggest_float('l1_rate', 1e-6, 1e-3, log=True) if use_kernel_regularizer else 0
    l2_rate = trial.suggest_float('l2_rate', 1e-6, 1e-3, log=True) if use_kernel_regularizer else 0
    
    use_recurrent_regularizer = trial.suggest_categorical('use_recurrent_regularizer', [True, False])
    recurrent_l1_rate = trial.suggest_float('recurrent_l1_rate', 1e-6, 1e-3, log=True) if use_recurrent_regularizer else 0
    recurrent_l2_rate = trial.suggest_float('recurrent_l2_rate', 1e-6, 1e-3, log=True) if use_recurrent_regularizer else 0
    
    # Batch Normalization
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Dense layer activation
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh', 'elu', 'selu'])
    
    # =============================================================================
    # TRAINING PARAMETERS
    # =============================================================================
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'rmsprop', 'sgd'])  # Removed adamw - not in legacy
    
    # Optimizer-specific parameters
    if optimizer_type == 'adam':
        beta_1 = trial.suggest_float('beta_1', 0.8, 0.99)
        beta_2 = trial.suggest_float('beta_2', 0.9, 0.999)
        epsilon = trial.suggest_float('epsilon', 1e-8, 1e-6, log=True)
    
    if optimizer_type == 'sgd':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
        nesterov = trial.suggest_categorical('nesterov', [True, False])
    
    if optimizer_type == 'rmsprop':
        rho = trial.suggest_float('rho', 0.8, 0.99)
        momentum = trial.suggest_float('rmsprop_momentum', 0.0, 0.9)
    
    # Learning rate scheduling
    use_lr_schedule = trial.suggest_categorical('use_lr_schedule', [True, False])
    if use_lr_schedule:
        lr_schedule_type = trial.suggest_categorical('lr_schedule_type', ['exponential', 'step', 'cosine'])
        if lr_schedule_type == 'exponential':
            decay_rate = trial.suggest_float('decay_rate', 0.9, 0.99)
        elif lr_schedule_type == 'step':
            step_size = trial.suggest_int('step_size', 5, 20)
            gamma = trial.suggest_float('gamma', 0.1, 0.9)
    
    # =============================================================================
    # BUILD MODEL
    # =============================================================================
    
    # Create regularizers
    kernel_regularizer = None
    if use_kernel_regularizer:
        kernel_regularizer = keras.regularizers.L1L2(l1=l1_rate, l2=l2_rate)
    
    recurrent_regularizer = None
    if use_recurrent_regularizer:
        recurrent_regularizer = keras.regularizers.L1L2(l1=recurrent_l1_rate, l2=recurrent_l2_rate)
    
    model = keras.Sequential()
    
    # LSTM Layers
    lstm_kwargs = {
        'activation': lstm_activation,
        'recurrent_activation': lstm_recurrent_activation,
        'use_bias': use_bias,
        'kernel_initializer': kernel_initializer,
        'recurrent_initializer': recurrent_initializer,
        'unit_forget_bias': unit_forget_bias,
        'dropout': dropout_rate,
        'recurrent_dropout': recurrent_dropout,
        'kernel_regularizer': kernel_regularizer,
        'recurrent_regularizer': recurrent_regularizer
    }
    
    # First LSTM layer
    if num_lstm_layers == 1:
        model.add(layers.LSTM(num_units_1, input_shape=(window_size, num_features), **lstm_kwargs))
    else:
        model.add(layers.LSTM(num_units_1, return_sequences=True, input_shape=(window_size, num_features), **lstm_kwargs))
    
    if use_batch_norm:
        model.add(layers.BatchNormalization())
    
    # Additional LSTM layers
    if num_lstm_layers >= 2:
        if num_lstm_layers == 2:
            model.add(layers.LSTM(num_units_2, **lstm_kwargs))
        else:
            model.add(layers.LSTM(num_units_2, return_sequences=True, **lstm_kwargs))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
    
    if num_lstm_layers >= 3:
        if num_lstm_layers == 3:
            model.add(layers.LSTM(num_units_3, **lstm_kwargs))
        else:
            model.add(layers.LSTM(num_units_3, return_sequences=True, **lstm_kwargs))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
    
    if num_lstm_layers >= 4:
        model.add(layers.LSTM(num_units_4, **lstm_kwargs))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
    
    # Dense layers
    model.add(layers.Dense(dense_units_1, activation=dense_activation))
    if use_batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    if num_dense_layers >= 2:
        model.add(layers.Dense(dense_units_2, activation=dense_activation))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    if num_dense_layers >= 3:
        model.add(layers.Dense(dense_units_3, activation=dense_activation))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1))
    
    # =============================================================================
    # COMPILE MODEL
    # =============================================================================
    
    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = keras.optimizers.legacy.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer_type == 'sgd':
        optimizer = keras.optimizers.legacy.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    elif optimizer_type == 'rmsprop':
        optimizer = keras.optimizers.legacy.RMSprop(
            learning_rate=learning_rate, rho=rho, momentum=momentum)
    
    # Learning rate schedule
    if use_lr_schedule:
        if lr_schedule_type == 'exponential':
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=100, decay_rate=decay_rate)
            optimizer.learning_rate = lr_schedule
        # Note: Step and Cosine schedules would be implemented as callbacks
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def objective(trial):
    """Comprehensive objective function for Optuna optimization with ALL parameters"""
    
    # =============================================================================
    # DATA PREPROCESSING PARAMETERS
    # =============================================================================
    window_size = trial.suggest_int('window_size', 20, 200, step=10)
    batch_size = trial.suggest_int('batch_size', 8, 256, step=8)
    epochs = trial.suggest_int('epochs', 5, 100, step=5)
    
    # Data normalization method
    scaler_type = trial.suggest_categorical('scaler_type', ['minmax', 'standard', 'robust'])
    
    # Data split ratios
    train_ratio = trial.suggest_float('train_ratio', 0.6, 0.8)
    val_ratio = trial.suggest_float('val_ratio', 0.15, 0.25)
    
    # Sequence overlap (for data augmentation)
    sequence_step = trial.suggest_int('sequence_step', 1, min(5, window_size//4))
    
    # =============================================================================
    # CALLBACK PARAMETERS
    # =============================================================================
    # Early stopping
    early_stopping_patience = trial.suggest_int('early_stopping_patience', 3, 15)
    early_stopping_monitor = trial.suggest_categorical('early_stopping_monitor', ['val_loss', 'val_mae'])
    min_delta = trial.suggest_float('min_delta', 1e-6, 1e-3, log=True)
    
    # Reduce learning rate on plateau
    use_reduce_lr = trial.suggest_categorical('use_reduce_lr', [True, False])
    if use_reduce_lr:
        reduce_lr_factor = trial.suggest_float('reduce_lr_factor', 0.1, 0.8)
        reduce_lr_patience = trial.suggest_int('reduce_lr_patience', 2, 10)
        reduce_lr_min_lr = trial.suggest_float('reduce_lr_min_lr', 1e-7, 1e-4, log=True)
    
    # =============================================================================
    # LOAD AND PREPROCESS DATA
    # =============================================================================
    data = get_rnn_data(18000, "15T")
    
    # Apply chosen normalization
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Create sequences with custom step size
    def create_sequences_with_step(data, window_size, prediction_steps=1, step=1):
        X, y = [], []
        for i in range(0, len(data) - window_size - prediction_steps + 1, step):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size:i + window_size + prediction_steps])
        return np.array(X), np.array(y)
    
    X, y = create_sequences_with_step(data_scaled, window_size, 1, sequence_step)
    
    if len(X) < 100:  # Ensure we have enough data
        return float('inf')
    
    # Dynamic data splitting
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    
    # Ensure we don't exceed data length
    if train_size + val_size >= len(X):
        val_size = len(X) - train_size - 1
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # =============================================================================
    # BUILD MODEL
    # =============================================================================
    model = build_lstm_model(trial, window_size, num_features=1)
    
    # =============================================================================
    # SETUP CALLBACKS
    # =============================================================================
    callbacks = []
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=0
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    if use_reduce_lr:
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=reduce_lr_min_lr,
            verbose=0
        )
        callbacks.append(reduce_lr)
    
    # Optuna pruning callback
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
    callbacks.append(pruning_callback)
    
    # =============================================================================
    # TRAIN MODEL
    # =============================================================================
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )
        
        # Return best validation loss
        best_val_loss = min(history.history['val_loss'])
        
        # Add penalty for models that are too complex (regularization)
        num_params = model.count_params()
        complexity_penalty = (num_params / 1000000) * 0.001  # Small penalty for very large models
        
        return best_val_loss + complexity_penalty
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

def plot_optimization_history(study, save_path='optuna_optimization_history.png'):
    """Plot optimization history"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Optimization history
    plt.subplot(1, 2, 1)
    trials = study.trials
    values = [trial.value for trial in trials if trial.value is not None]
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
        
        def suggest_categorical(self, name, choices):
            return self.params.get(name, choices[0])
    
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
        epochs=best_params.get('epochs', 30),  # Use optimized epochs or default to 50
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
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
                    forecasts_original, forecast_steps, 'optimized_lstm_predictions.png')
    
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
    plt.ylabel("Global Active Power")
    plt.title("Optimized LSTM Time Series Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def print_parameter_summary():
    """Print comprehensive list of all parameters being optimized"""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE LSTM HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print("🎯 Total Parameters Being Optimized: 38+")
    print("\n🏗️  ARCHITECTURE PARAMETERS:")
    print("   • num_lstm_layers: 1-4 layers")
    print("   • num_units_1/2/3/4: 32-512 units per layer")
    print("   • num_dense_layers: 1-3 dense layers")
    print("   • dense_units_1/2/3: 4-256 units per dense layer")
    
    print("\n🧠 LSTM-SPECIFIC PARAMETERS:")
    print("   • lstm_activation: tanh, relu, sigmoid")
    print("   • lstm_recurrent_activation: sigmoid, hard_sigmoid")
    print("   • use_bias: True/False")
    print("   • unit_forget_bias: True/False")
    print("   • kernel_initializer: glorot_uniform, he_uniform, lecun_uniform, random_normal")
    print("   • recurrent_initializer: orthogonal, glorot_uniform, he_uniform")
    
    print("\n🎓 TRAINING PARAMETERS:")
    print("   • learning_rate: 1e-5 to 1e-1 (log scale)")
    print("   • optimizer_type: adam, rmsprop, sgd")
    print("   • beta_1/beta_2: Adam optimizer parameters")
    print("   • momentum: SGD/RMSprop momentum")
    print("   • epsilon: Adam epsilon")
    print("   • use_lr_schedule: exponential/step/cosine decay")
    
    print("\n🛡️  REGULARIZATION PARAMETERS:")
    print("   • dropout_rate: 0.0-0.7")
    print("   • recurrent_dropout: 0.0-0.5")
    print("   • use_kernel_regularizer: L1/L2 for weights")
    print("   • use_recurrent_regularizer: L1/L2 for recurrent weights")
    print("   • l1_rate/l2_rate: 1e-6 to 1e-3 (log scale)")
    print("   • use_batch_norm: True/False")
    print("   • dense_activation: relu, tanh, elu, selu")
    
    print("\n📊 DATA & PREPROCESSING:")
    print("   • window_size: 20-200 timesteps")
    print("   • batch_size: 8-256")
    print("   • epochs: 5-100")
    print("   • scaler_type: minmax, standard, robust")
    print("   • train_ratio: 0.6-0.8")
    print("   • val_ratio: 0.15-0.25")
    print("   • sequence_step: 1-5 (data augmentation)")
    
    print("\n⚙️  CALLBACK PARAMETERS:")
    print("   • early_stopping_patience: 3-15 epochs")
    print("   • early_stopping_monitor: val_loss, val_mae")
    print("   • min_delta: 1e-6 to 1e-3 (log scale)")
    print("   • use_reduce_lr: True/False")
    print("   • reduce_lr_factor: 0.1-0.8")
    print("   • reduce_lr_patience: 2-10 epochs")
    print("="*80)

def main():
    print("🚀 Starting COMPREHENSIVE Optuna hyperparameter optimization for LSTM...")
    print_parameter_summary()
    
    # Create study with more sophisticated pruning
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
        sampler=optuna.samplers.TPESampler(n_startup_trials=20)
    )
    
    # Run optimization with more trials for comprehensive search
    print("Running optimization trials...")
    print("⚠️  This may take several hours due to the comprehensive parameter space!")
    study.optimize(objective, n_trials=200, timeout=14400)  # 4 hour timeout for comprehensive search
    
    print("Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (validation loss): {study.best_value:.4f}")
    print("\n" + "="*80)
    print("🏆 BEST PARAMETERS FOUND:")
    print("="*80)
    
    # Group parameters by category for better readability
    architecture_params = {}
    lstm_params = {}
    training_params = {}
    regularization_params = {}
    data_params = {}
    callback_params = {}
    
    for key, value in study.best_params.items():
        if any(x in key for x in ['num_lstm_layers', 'num_units', 'num_dense_layers', 'dense_units']):
            architecture_params[key] = value
        elif any(x in key for x in ['lstm_activation', 'recurrent_activation', 'use_bias', 'unit_forget_bias', 'initializer']):
            lstm_params[key] = value
        elif any(x in key for x in ['learning_rate', 'optimizer', 'beta', 'momentum', 'epsilon', 'rho', 'nesterov', 'lr_schedule', 'decay', 'gamma', 'step_size']):
            training_params[key] = value
        elif any(x in key for x in ['dropout', 'regularizer', 'l1_rate', 'l2_rate', 'batch_norm', 'dense_activation']):
            regularization_params[key] = value
        elif any(x in key for x in ['window_size', 'batch_size', 'epochs', 'scaler_type', 'train_ratio', 'val_ratio', 'sequence_step']):
            data_params[key] = value
        elif any(x in key for x in ['early_stopping', 'min_delta', 'reduce_lr']):
            callback_params[key] = value
    
    def print_category(name, params_dict):
        if params_dict:
            print(f"\n📋 {name}:")
            for key, value in params_dict.items():
                print(f"   {key}: {value}")
    
    print_category("ARCHITECTURE", architecture_params)
    print_category("LSTM-SPECIFIC", lstm_params)
    print_category("TRAINING", training_params)
    print_category("REGULARIZATION", regularization_params)
    print_category("DATA & PREPROCESSING", data_params)
    print_category("CALLBACKS", callback_params)
    
    print("\n" + "="*80)
    
    # Save study
    joblib.dump(study, 'optuna_study.pkl')
    print("Study saved to 'optuna_study.pkl'")
    
    # Plot optimization history
    plot_optimization_history(study)
    print("Optimization history saved to 'optuna_optimization_history.png'")
    
    # Train final model with best parameters
    best_model, history, forecasts = train_best_model(study.best_params)
    
    # Save best model
    best_model.save('best_lstm_model.h5')
    print("Best model saved to 'best_lstm_model.h5'")
    
    print("\nOptimization and final training completed!")
    print("Files created:")
    print("- optuna_study.pkl: Saved optimization study")
    print("- optuna_optimization_history.png: Optimization visualization")
    print("- optimized_lstm_predictions.png: Final model predictions")
    print("- best_lstm_model.h5: Best trained model")

if __name__ == '__main__':
    main() 