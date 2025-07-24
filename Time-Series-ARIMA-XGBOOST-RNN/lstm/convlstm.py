import sys
print(sys.executable)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
import os
from datetime import datetime

# New imports for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Input, Reshape, Conv2DTranspose, Dropout, ConvLSTM2D, BatchNormalization
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import json

# Import wandb for experiment tracking
import wandb

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Get files for training
nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
print(f"Processing {len(nc_files)} files with proper temporal splitting")

def extract_timestamps_from_filenames(nc_files):
    """Extract timestamps from NetCDF filenames."""
    timestamps = []
    for file in nc_files:
        filename = Path(file).name
        date_part = filename.split('.')[2][1:9]  # Extract s20180816 -> 20180816
        timestamp = datetime.strptime(date_part, '%Y%m%d')
        timestamps.append(timestamp)
    return timestamps

def downsample_data(data_array, target_height=32, target_width=64):
    """Downsample to smaller resolution to save memory."""
    if data_array.ndim == 2:
        return resize(data_array, (target_height, target_width), preserve_range=True)
    elif data_array.ndim == 3:
        return np.array([resize(img, (target_height, target_width), preserve_range=True) for img in data_array])
    else:
        raise ValueError(f"Unexpected number of dimensions: {data_array.ndim}")

def process_and_plot_to_array(nc_file):
    """Convert a netCDF file to a simple normalized array."""
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    
    # Normalize the data to 0-1 range
    data_min = np.nanmin(data_array_2d)
    data_max = np.nanmax(data_array_2d)
    if data_max > data_min:
        normalized_data = (data_array_2d - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data_array_2d)
    
    # Replace NaN values with 0
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    
    return normalized_data

def extract_data_and_clusters(nc_files):
    """Extract and preprocess data for training."""
    data = []
    
    print("Processing files and creating normalized images...")
    for i, nc_file in enumerate(nc_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(nc_files)}")
        
        normalized_image = process_and_plot_to_array(nc_file)
        downsampled_image = downsample_data(normalized_image)
        data.append(downsampled_image)
    
    data = np.array(data)
    print(f"Final data shape: {data.shape}")
    
    return data

def temporal_train_test_split_proper(data, seq_length, test_ratio=0.2):
    """
    Properly split time series data to avoid leakage.
    
    Key insight: We need to leave a gap of seq_length between train and test
    to ensure no training targets appear in test sequences.
    """
    total_files = len(data)
    
    # Calculate split points with gap
    test_files_needed = int(total_files * test_ratio)
    gap_needed = seq_length  # Need gap equal to sequence length
    
    # Ensure we have enough data
    min_train_files = seq_length + 1  # Need at least seq_length + 1 for training
    min_files_needed = min_train_files + gap_needed + test_files_needed + seq_length
    
    if total_files < min_files_needed:
        print(f"WARNING: Only {total_files} files available, need {min_files_needed} for proper split")
        print("Reducing test ratio to fit available data...")
        test_files_needed = max(1, (total_files - min_train_files - gap_needed - seq_length) // 2)
    
    # Calculate split indices
    train_end = total_files - test_files_needed - gap_needed - seq_length
    test_start = train_end + gap_needed
    
    print(f"\n=== PROPER TEMPORAL SPLIT ===")
    print(f"Total files: {total_files}")
    print(f"Training files: 0 to {train_end-1} ({train_end} files)")
    print(f"Gap (no data used): {train_end} to {test_start-1} ({gap_needed} files)")
    print(f"Test files: {test_start} to {total_files-1} ({total_files - test_start} files)")
    
    # Split the data
    train_data = data[:train_end]
    test_data = data[test_start:]
    
    return train_data, test_data, train_end, test_start

def create_sequences_from_data(data, seq_length):
    """Create sequences from data array."""
    sequences = []
    target_images = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length] 
        target_image = data[i + seq_length]
        sequences.append(seq)
        target_images.append(target_image)
    return np.array(sequences), np.array(target_images)

def create_convlstm_model(seq_length, img_height, img_width):
    """Create ConvLSTM model that preserves spatial structure throughout temporal processing."""
    
    # Model parameters - adapted for ConvLSTM
    convlstm1_filters = 32
    convlstm2_filters = 64
    conv_filters = 32
    dropout_rate = 0.3
    learning_rate = 0.007
    
    model = Sequential([
        # Input layer: (batch, seq_length, height, width, channels)
        Input(shape=(seq_length, img_height, img_width, 1)),
        
        # ConvLSTM layers - these maintain spatial structure while processing temporal sequences
        ConvLSTM2D(
            filters=convlstm1_filters,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,  # Return sequences for stacking multiple ConvLSTM layers
            activation='relu',
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        ),
        BatchNormalization(),
        
        ConvLSTM2D(
            filters=convlstm2_filters,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False,  # Return only the final output
            activation='relu',
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        ),
        BatchNormalization(),
        
        # Additional convolutional layers for refinement
        Conv2D(conv_filters, (3, 3), activation='relu', padding='same'),
        Dropout(dropout_rate),
        
        # Output layer - single channel for microplastic concentration
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    
    # Compile with optimized parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def verify_no_leakage(X_train, X_test, y_train, y_test, seq_length):
    """Verify there's no data leakage between train and test sets."""
    print("\n=== LEAKAGE VERIFICATION ===")
    
    # Check if any training target appears in test sequences
    train_targets_flat = y_train.reshape(len(y_train), -1)
    test_sequences_flat = X_test.reshape(len(X_test), seq_length, -1)
    
    leakage_count = 0
    for i, train_target in enumerate(train_targets_flat):
        for j, test_sequence in enumerate(test_sequences_flat):
            for k, test_frame in enumerate(test_sequence):
                if np.allclose(train_target, test_frame, rtol=1e-10):
                    leakage_count += 1
                    if leakage_count <= 3:  # Only print first few
                        print(f"  LEAKAGE: train_target[{i}] == test_sequence[{j}][{k}]")
    
    if leakage_count == 0:
        print("✓ NO LEAKAGE DETECTED - Train and test sets are properly separated!")
        return True
    else:
        print(f"✗ LEAKAGE DETECTED - Found {leakage_count} instances of training data in test sequences!")
        return False

def train_model_fixed(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=100):
    """Train the model with validation data."""
    print("\nStarting model training...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Log training metrics to wandb
    for epoch in range(epochs):
        wandb.log({
            "loss": history.history['loss'][epoch],
            "val_loss": history.history['val_loss'][epoch],
            "mae": history.history['mae'][epoch],
            "val_mae": history.history['val_mae'][epoch],
            "epoch": epoch
        })
    
    return history

def evaluate_model(model, X_test, y_test, timestamps, test_start_idx):
    """Evaluate the model and create visualizations."""
    print("\nEvaluating model...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    
    # Calculate SSIM for each prediction
    ssim_scores = []
    for i in range(len(predictions)):
        score = ssim(y_test[i].squeeze(), predictions[i].squeeze(),
                    win_size=3, data_range=1.0, channel_axis=None)
        ssim_scores.append(score)
    
    mean_ssim = np.mean(ssim_scores)
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    
    # Create visualization
    n_predictions = min(len(predictions), 5)
    plt.figure(figsize=(20, 12))  # Made taller to accommodate difference plots
    
    # Debug: Check if predictions are different from true values
    print(f"\nDebug: Checking prediction differences...")
    for i in range(min(3, len(predictions))):
        true_img = y_test[i].squeeze()
        pred_img = predictions[i].squeeze()
        diff = np.abs(true_img - pred_img)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Image {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    for i in range(n_predictions):
        # True image
        plt.subplot(3, n_predictions, i + 1)
        true_img = y_test[i].squeeze()
        plt.imshow(true_img, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'True Image {i+1}\nDate: {timestamps[test_start_idx + 4 + i].strftime("%Y-%m-%d")}')
        plt.axis('off')
        
        # Predicted image
        plt.subplot(3, n_predictions, n_predictions + i + 1)
        pred_img = predictions[i].squeeze()
        plt.imshow(pred_img, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Predicted Image {i+1}\nSSIM: {ssim_scores[i]:.3f}')
        plt.axis('off')
        
        # Difference image
        plt.subplot(3, n_predictions, 2*n_predictions + i + 1)
        diff_img = np.abs(true_img - pred_img)
        plt.imshow(diff_img, cmap='hot', vmin=0, vmax=0.2)
        plt.title(f'Difference {i+1}\nMax: {np.max(diff_img):.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('convlstm_model_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mse, mae, mean_ssim, predictions

#_______________________________
# Main execution
#_______________________________

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="convlstm-microplastic-prediction")
    
    print("=" * 60)
    print("CONVLSTM WITH PROPER DATA SPLITTING")
    print("=" * 60)
    
    # Extract timestamps and ensure temporal ordering
    timestamps = extract_timestamps_from_filenames(nc_files)
    print(f"Data spans from {timestamps[0]} to {timestamps[-1]}")
    
    # Extract data
    print("\nExtracting data...")
    data = extract_data_and_clusters(nc_files)
    
    # Set sequence length (from Optuna optimization)
    seq_length = 4
    print(f"\nUsing sequence length: {seq_length}")
    
    # Proper temporal split
    train_data, test_data, train_end_idx, test_start_idx = temporal_train_test_split_proper(
        data, seq_length, test_ratio=0.2
    )
    
    # Create sequences separately for train and test
    X_train, y_train = create_sequences_from_data(train_data, seq_length)
    X_test, y_test = create_sequences_from_data(test_data, seq_length)
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    
    print(f"\nFinal data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Verify no leakage
    no_leakage = verify_no_leakage(X_train, X_test, y_train, y_test, seq_length)
    
    if not no_leakage:
        print("ERROR: Data leakage detected! Check the splitting logic.")
        exit(1)
    
    # Create validation split from training data (temporal split)
    val_split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train[:val_split_idx]
    y_train_final = y_train[:val_split_idx]
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    
    print(f"\nValidation split:")
    print(f"Final training: {X_train_final.shape}")
    print(f"Validation: {X_val.shape}")
    
    # Create and train model
    img_height, img_width = data[0].shape[:2]
    model = create_convlstm_model(seq_length, img_height, img_width)
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    history = train_model_fixed(model, X_train_final, y_train_final, X_val, y_val)
    
    # Evaluate model
    mse, mae, mean_ssim, predictions = evaluate_model(model, X_test, y_test, timestamps, test_start_idx)
    
    # Save training history plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('convlstm_model_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'model_type': 'ConvLSTM with proper splitting',
        'no_data_leakage': no_leakage,
        'sequence_length': seq_length,
        'total_files': len(nc_files),
        'train_files': train_end_idx,
        'test_files': len(test_data),
        'gap_files': test_start_idx - train_end_idx,
        'final_metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'ssim': float(mean_ssim)
        },
        'training_history': {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_mae': float(history.history['mae'][-1]),
            'final_val_mae': float(history.history['val_mae'][-1])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('convlstm_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"✓ No data leakage: {no_leakage}")
    print(f"✓ Test MSE: {mse:.6f}")
    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training files: {train_end_idx}")
    print(f"✓ Gap files: {test_start_idx - train_end_idx}")
    print(f"✓ Test files: {len(test_data)}")
    print("\nFiles generated:")
    print("- convlstm_model_predictions.png")
    print("- convlstm_model_training_history.png") 
    print("- convlstm_model_results.json")
    print("=" * 60)
    
    # Finish wandb run
    wandb.finish()