import sys
print(sys.executable)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import json
import wandb

# Set device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
print(f"Using device: {device}")

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

def downsample_data(data_array, target_height=128, target_width=128):
    """Downsample to higher resolution optimized for Japan region."""
    if data_array.ndim == 2:
        return resize(data_array, (target_height, target_width), preserve_range=True)
    elif data_array.ndim == 3:
        return np.array([resize(img, (target_height, target_width), preserve_range=True) for img in data_array])
    else:
        raise ValueError(f"Unexpected number of dimensions: {data_array.ndim}")

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def process_and_plot_to_array(nc_file):
    """Convert a netCDF file to a Japan region cropped normalized array."""
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    
    # Try to get the actual lat/lon coordinates from the dataset
    lats = None
    lons = None
    
    # Check for different possible coordinate variable names
    possible_lat_names = ['lat', 'latitude', 'y', 'lat_1', 'lat_2']
    possible_lon_names = ['lon', 'longitude', 'x', 'lon_1', 'lon_2']
    
    for lat_name in possible_lat_names:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            break
    
    for lon_name in possible_lon_names:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            break
    
    # Apply Japan region cropping if coordinates are available
    if lats is not None and lons is not None:
        # Japan region coordinates (broader)
        japan_sw_lat, japan_sw_lon = 25.35753, 118.85766
        japan_ne_lat, japan_ne_lon = 36.98134, 145.47117
        
        # Convert to grid indices
        japan_sw_lat_idx, japan_sw_lon_idx = lat_lon_to_indices(japan_sw_lat, japan_sw_lon, lats, lons)
        japan_ne_lat_idx, japan_ne_lon_idx = lat_lon_to_indices(japan_ne_lat, japan_ne_lon, lats, lons)
        
        # Ensure proper ordering
        japan_lat_start = min(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lat_end = max(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lon_start = min(japan_sw_lon_idx, japan_ne_lon_idx)
        japan_lon_end = max(japan_sw_lon_idx, japan_ne_lon_idx)
        
        # Crop the data to Japan region
        data_array_2d = data_array_2d[japan_lat_start:japan_lat_end, japan_lon_start:japan_lon_end]
        
        # Only print once at the beginning
        if nc_file == nc_files[0]:
            print(f"Japan region cropped to: lat[{japan_lat_start}:{japan_lat_end}], lon[{japan_lon_start}:{japan_lon_end}]")
            print(f"Japan region shape: {data_array_2d.shape}")
    else:
        print("Warning: No coordinate information found, using full dataset")
    
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
    """Extract and preprocess Japan region data for training."""
    data = []
    
    # Limit processing to first 500 files for testing
    max_files = 500
    nc_files_limited = nc_files[:max_files]
    
    print(f"Processing {len(nc_files_limited)} files (limited from {len(nc_files)} total files)")
    print("Processing files and creating normalized Japan region images...")
    
    for i, nc_file in enumerate(nc_files_limited):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(nc_files_limited)} - Japan region cropping")
        
        normalized_image = process_and_plot_to_array(nc_file)
        downsampled_image = downsample_data(normalized_image)
        data.append(downsampled_image)
    
    data = np.array(data)
    print(f"Final Japan region data shape: {data.shape}")
    print(f"Geographic region: Japan (25.35753°N-36.98134°N, 118.85766°E-145.47117°E)")
    
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
    min_train_files = seq_length + 1  
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

class EfficientCNNLSTMModel(nn.Module):
    """Efficient CNN-LSTM model that won't get stuck during training."""
    
    def __init__(self, seq_length, img_height, img_width):
        super(EfficientCNNLSTMModel, self).__init__()
        
        # Simplified parameters for efficiency
        self.conv1_filters = 32
        self.conv2_filters = 64
        self.lstm_units = 128
        self.dropout_rate = 0.1
        
        self.seq_length = seq_length
        self.img_height = img_height
        self.img_width = img_width
        
        # Simplified CNN layers
        self.conv1 = nn.Conv2d(1, self.conv1_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.conv1_filters, self.conv2_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Calculate the size after CNN layers (2 pooling layers: 256->128->64, 128->64->32)
        conv_output_height = img_height // 4  # 128 -> 32
        conv_output_width = img_width // 4    # 256 -> 64
        conv_output_size = self.conv2_filters * conv_output_height * conv_output_width
        
        # Single LSTM layer for efficiency
        self.lstm = nn.LSTM(conv_output_size, self.lstm_units, batch_first=True)
        
        # Simple decoder
        self.decoder = nn.Linear(self.lstm_units, img_height * img_width)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, height, width)
        batch_size, seq_length, height, width = x.shape
        x = x.unsqueeze(-1)  # Add channel dimension
        
        # Process each sequence step through CNN
        cnn_outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :, :, :].permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
            
            # Simplified CNN
            conv1_out = F.relu(self.conv1(x_t))
            conv1_out = self.pool(conv1_out)
            conv1_out = self.dropout(conv1_out)
            
            conv2_out = F.relu(self.conv2(conv1_out))
            conv2_out = self.pool(conv2_out)
            conv2_out = self.dropout(conv2_out)
            
            # Flatten for LSTM
            conv2_out_flat = conv2_out.view(batch_size, -1)
            cnn_outputs.append(conv2_out_flat)
        
        # Stack CNN outputs for LSTM
        cnn_outputs = torch.stack(cnn_outputs, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_outputs)
        
        # Take the last output from LSTM
        last_output = lstm_out[:, -1, :]
        
        # Generate output image
        output = self.decoder(last_output)
        output = output.view(batch_size, height, width)
        
        # IMPROVED: Use tanh for better contrast then scale to 0-1
        output = torch.tanh(output)  # Range [-1, 1] for better contrast
        output = (output + 1) / 2   # Scale to [0, 1] while preserving contrast
        
        return output

def improved_contrast_loss(pred, target, low_threshold=0.2, high_threshold=0.6, 
                          low_weight=2.0, high_weight=2.0, contrast_weight=1.0):
    """
    Improved loss function that balances high/low concentrations and encourages contrast.
    """
    # Basic MAE
    mae = torch.mean(torch.abs(pred - target))
    
    # Create masks for different concentration levels
    low_concentration_mask = (target < low_threshold).float()
    high_concentration_mask = (target > high_threshold).float()
    mid_concentration_mask = 1.0 - low_concentration_mask - high_concentration_mask
    
    # Weighted loss that pays attention to ALL concentration levels
    weighted_mae = torch.mean(
        torch.abs(pred - target) * (
            1.0 +  # Base weight
            low_weight * low_concentration_mask +      # Extra attention to low concentrations
            high_weight * high_concentration_mask +    # Extra attention to high concentrations
            0.5 * mid_concentration_mask               # Some attention to mid concentrations
        )
    )
    
    # Contrast preservation loss - encourages maintaining the range of values
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    contrast_loss = torch.abs(pred_std - target_std)
    
    # Edge preservation loss - maintains sharp transitions
    pred_grad_x = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
    target_grad_x = torch.abs(target[:, :-1, :] - target[:, 1:, :])
    pred_grad_y = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1] - target[:, :, 1:])
    
    edge_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x)) + \
                torch.mean(torch.abs(pred_grad_y - target_grad_y))
    
    # Combine all losses
    total_loss = weighted_mae + contrast_weight * contrast_loss + 0.1 * edge_loss
    
    return total_loss

def train_efficient_model_pytorch(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train the model efficiently with progress monitoring."""
    print(f"\nStarting efficient model training for {epochs} epochs...")
    
    model.to(device)
    
    # Simple optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx % 5 == 0:  # Print progress every 5 batches
                print(f"  Training batch {batch_idx+1}/{len(train_loader)}")
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # USE IMPROVED LOSS FUNCTION
            loss = improved_contrast_loss(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        print(f"  Starting validation...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = improved_contrast_loss(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"  Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    print(f"\nTraining completed successfully!")
    
    return history

def evaluate_model(model, test_loader, timestamps, test_start_idx):
    """Evaluate the model and create visualizations with proper geographic context."""
    print("\nEvaluating model...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    # Calculate SSIM for each prediction
    ssim_scores = []
    for i in range(len(predictions)):
        pred_img = predictions[i]
        target_img = targets[i]
        
        # Normalize for SSIM calculation
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
        
        ssim_score = ssim(target_img, pred_img, data_range=1.0)
        ssim_scores.append(ssim_score)
    
    mean_ssim = np.mean(ssim_scores)
    
    print(f"Test MAE: {mae:.6f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    
    # Create visualization with proper aspect ratio and flipped orientation
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # Wider figure for 2:1 aspect ratio
    fig.suptitle('Enhanced Japan Region CNN-LSTM Predictions vs Actual\n(Covering Japan, Korea, and Eastern China)', fontsize=16)
    
    # Show first 4 test samples
    for i in range(min(4, len(predictions))):
        # Original - flip vertically for correct geographic orientation
        target_flipped = np.flipud(targets[i])
        axes[0, i].imshow(target_flipped, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Actual (Japan Region)\n{timestamps[test_start_idx + i].strftime("%Y-%m-%d")}')
        axes[0, i].axis('off')
        
        # Predicted - flip vertically for correct geographic orientation
        pred_flipped = np.flipud(predictions[i])
        axes[1, i].imshow(pred_flipped, cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Predicted (Japan Region)\n{timestamps[test_start_idx + i].strftime("%Y-%m-%d")}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('japan_lstm_predictions_enhanced.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mae, mean_ssim, predictions

#_______________________________
# Main execution
#_______________________________

if __name__ == "__main__":
    print("=" * 60)
    print("JAPAN REGION CNN-LSTM WITH PYTORCH AND GEOGRAPHIC CROPPING")
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
    
    print(f"\nFinal data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Create validation split from training data (temporal split)
    val_split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train[:val_split_idx]
    y_train_final = y_train[:val_split_idx]
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    
    print(f"\nValidation split:")
    print(f"Final training: {X_train_final.shape}")
    print(f"Validation: {X_val.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_final)
    y_train_tensor = torch.FloatTensor(y_train_final)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    batch_size = 8  # Smaller batch size for efficiency
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize wandb
    wandb.init(
        project="mp-prediction-pytorch",
        config={
            "model_type": "Efficient Japan Region CNN-LSTM",
            "framework": "PyTorch",
            "sequence_length": 4,
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 10,
            "image_resolution": "256x128",
            "aspect_ratio": "2:1_width_height",
            "loss_function": "simple_weighted_mae",
            "high_concentration_threshold": 0.4,
            "high_concentration_weight": 5.0,
            "cnn_layers": 2,
            "lstm_layers": 1,
            "conv_filters": [32, 64],
            "lstm_units": 128,
            "dropout_rate": 0.1,
            "simplified_architecture": True,
            "geographic_fixes": {
                "vertical_flip_correction": True,
                "proper_aspect_ratio": "2:1",
                "geographic_coverage": "Japan_Korea_EasternChina"
            }
        }
    )
    
    # Create model
    img_height, img_width = data[0].shape[:2]
    model = EfficientCNNLSTMModel(seq_length, img_height, img_width)
    
    print("\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_efficient_model_pytorch(model, train_loader, val_loader, epochs=10)
    
    # Evaluate model
    mae, mean_ssim, predictions = evaluate_model(model, test_loader, timestamps, test_start_idx)
    
    # Log final test metrics to wandb
    wandb.log({
        "test_mae": mae,
        "test_ssim": mean_ssim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Save simple training history plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training MAE')
    plt.plot(history['val_loss'], label='Validation MAE')
    plt.title('Efficient Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], 'o-', label='Train')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], 's-', label='Validation')
    plt.title('Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('japan_lstm_training_efficient.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save simplified results
    results = {
        'model_type': 'Efficient Japan Region CNN-LSTM with Geographic Corrections',
        'framework': 'PyTorch',
        'device': str(device),
        'sequence_length': seq_length,
        'image_resolution': f"{img_height}x{img_width}",
        'training_samples': len(X_train_final),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'test_mae': mae,
        'test_ssim': mean_ssim,
        'model_parameters': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv1_filters': model.conv1_filters,
            'conv2_filters': model.conv2_filters,
            'lstm_units': model.lstm_units,
            'dropout_rate': model.dropout_rate,
            'simplified_architecture': True
        },
        'training_config': {
            'epochs': 10,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        },
        'geographic_region': {
            'name': 'Japan',
            'southwest': [25.35753, 118.85766],
            'northeast': [36.98134, 145.47117],
            'description': 'Efficient Japan region model with corrected orientation and 2:1 aspect ratio',
            'coverage': 'Japan, Korea, and Eastern China maritime regions',
            'fixes_applied': {
                'vertical_flip': True,
                'aspect_ratio': '2:1 (width:height)',
                'simplified_model': True
            }
        }
    }
    
    with open('japan_lstm_results_efficient.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EFFICIENT JAPAN REGION CNN-LSTM TRAINING COMPLETED")
    print("=" * 60)
    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training files: {train_end_idx}")
    print(f"✓ Test files: {len(test_data)}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Image resolution: 256x128 (2:1 aspect ratio)")
    print(f"✓ Geographic coverage: Japan, Korea, Eastern China")
    print(f"✓ Orientation: Corrected (vertical flip applied)")
    print("\nFiles generated:")
    print("- japan_lstm_predictions_enhanced.png")
    print("- japan_lstm_training_efficient.png") 
    print("- japan_lstm_results_efficient.json")
    print("=" * 60)
    
    # Finish wandb run
    wandb.finish() 