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

def downsample_data(data_array, target_height=64, target_width=64):
    """Downsample to smaller resolution optimized for Tsushima region."""
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
    """Convert a netCDF file to a Tsushima region cropped normalized array."""
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
    
    # Apply Tsushima region cropping if coordinates are available
    if lats is not None and lons is not None:
        # Tsushima region coordinates
        tsushima_sw_lat, tsushima_sw_lon = 34.02837, 129.11613
        tsushima_ne_lat, tsushima_ne_lon = 34.76456, 129.55801
        
        # Convert to grid indices
        tsushima_sw_lat_idx, tsushima_sw_lon_idx = lat_lon_to_indices(tsushima_sw_lat, tsushima_sw_lon, lats, lons)
        tsushima_ne_lat_idx, tsushima_ne_lon_idx = lat_lon_to_indices(tsushima_ne_lat, tsushima_ne_lon, lats, lons)
        
        # Ensure proper ordering
        tsushima_lat_start = min(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
        tsushima_lat_end = max(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
        tsushima_lon_start = min(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
        tsushima_lon_end = max(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
        
        # Crop the data to Tsushima region
        data_array_2d = data_array_2d[tsushima_lat_start:tsushima_lat_end, tsushima_lon_start:tsushima_lon_end]
        
        print(f"Tsushima region cropped to: lat[{tsushima_lat_start}:{tsushima_lat_end}], lon[{tsushima_lon_start}:{tsushima_lon_end}]")
        print(f"Tsushima region shape: {data_array_2d.shape}")
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
    """Extract and preprocess Tsushima region data for training."""
    data = []
    
    print("Processing files and creating normalized Tsushima region images...")
    for i, nc_file in enumerate(nc_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(nc_files)} - Tsushima region cropping")
        
        normalized_image = process_and_plot_to_array(nc_file)
        downsampled_image = downsample_data(normalized_image)
        data.append(downsampled_image)
    
    data = np.array(data)
    print(f"Final Tsushima region data shape: {data.shape}")
    print(f"Geographic region: Tsushima Strait (34.02837°N-34.76456°N, 129.11613°E-129.55801°E)")
    
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

class CNNLSTMModel(nn.Module):
    """CNN-LSTM model in PyTorch."""
    
    def __init__(self, seq_length, img_height, img_width):
        super(CNNLSTMModel, self).__init__()
        
        # Best parameters from Optuna optimization
        self.conv1_filters = 32
        self.conv2_filters = 64
        self.lstm1_units = 128
        self.lstm2_units = 64
        self.dropout_rate = 0.3
        
        self.seq_length = seq_length
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN layers (applied to each frame in sequence)
        self.conv1 = nn.Conv2d(1, self.conv1_filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(self.dropout_rate)
        
        self.conv2 = nn.Conv2d(self.conv1_filters, self.conv2_filters, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(self.dropout_rate)
        
        # Calculate feature dimensions after conv layers
        self.conv_height = img_height // 4  # Two max pooling layers
        self.conv_width = img_width // 4
        self.conv_features = self.conv_height * self.conv_width * self.conv2_filters
        
        # LSTM layers
        self.lstm1 = nn.LSTM(self.conv_features, self.lstm1_units, batch_first=True, dropout=self.dropout_rate)
        self.lstm2 = nn.LSTM(self.lstm1_units, self.lstm2_units, batch_first=True)
        
        # Dense layer
        self.dense = nn.Linear(self.lstm2_units, self.conv_features)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        # Upsampling layers (transposed convolutions)
        self.conv_transpose1 = nn.ConvTranspose2d(self.conv2_filters, self.conv1_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(self.conv1_filters, self.conv1_filters//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(self.conv1_filters//2, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, height, width, channels)
        batch_size, seq_length, height, width = x.shape[:4]
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(seq_length):
            # Get frame at time t: (batch_size, height, width) -> (batch_size, 1, height, width)
            frame = x[:, t].unsqueeze(1)
            
            # CNN forward pass
            out = F.relu(self.conv1(frame))
            out = self.pool1(out)
            out = self.dropout1(out)
            
            out = F.relu(self.conv2(out))
            out = self.pool2(out)
            out = self.dropout2(out)
            
            # Flatten: (batch_size, conv2_filters, conv_height, conv_width) -> (batch_size, conv_features)
            out = out.view(batch_size, -1)
            cnn_features.append(out)
        
        # Stack features: (batch_size, seq_length, conv_features)
        lstm_input = torch.stack(cnn_features, dim=1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm1(lstm_input)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Take last output from LSTM
        lstm_final = lstm_out[:, -1, :]  # (batch_size, lstm2_units)
        
        # Dense layer
        dense_out = F.relu(self.dense(lstm_final))
        dense_out = self.dropout3(dense_out)
        
        # Reshape for upsampling: (batch_size, conv_features) -> (batch_size, conv2_filters, conv_height, conv_width)
        reshaped = dense_out.view(batch_size, self.conv2_filters, self.conv_height, self.conv_width)
        
        # Upsampling
        out = F.relu(self.conv_transpose1(reshaped))
        out = F.relu(self.conv_transpose2(out))
        out = torch.sigmoid(self.conv_transpose3(out))
        
        # Remove channel dimension: (batch_size, 1, height, width) -> (batch_size, height, width)
        return out.squeeze(1)
'''
def verify_no_leakage(X_train, X_test, y_train, y_test, seq_length):
    """Verify there's no data leakage between train and test sets."""
    print("\n=== LEAKAGE VERIFICATION ===")
    
    # Convert to numpy if tensors
    if torch.is_tensor(X_train):
        X_train = X_train.numpy()
    if torch.is_tensor(X_test):
        X_test = X_test.numpy()
    if torch.is_tensor(y_train):
        y_train = y_train.numpy()
    if torch.is_tensor(y_test):
        y_test = y_test.numpy()
    
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
'''
def aggressive_weighted_loss(pred, target, high_threshold=0.4, high_weight=20.0, gradient_weight=0.1):
    """
    Aggressive weighted loss to force sharp, detailed predictions.
    
    Args:
        pred: Predicted values
        target: True values
        high_threshold: Lower threshold to catch more important pixels
        high_weight: Much higher weight for high-concentration pixels
        gradient_weight: Weight for gradient penalty to discourage smoothness
    """
    # 1. Weighted MAE with aggressive weighting
    weight_mask = torch.ones_like(target)
    
    # Very high concentration pixels get extreme weight
    very_high_mask = target > 0.7
    high_mask = (target > high_threshold) & (target <= 0.7)
    
    weight_mask[high_mask] = high_weight
    weight_mask[very_high_mask] = high_weight * 2  # Even more weight for very bright pixels
    
    # Calculate weighted MAE
    mae = torch.abs(pred - target)
    weighted_mae = (mae * weight_mask).mean()
    
    # 2. Gradient penalty to discourage overly smooth predictions
    # Calculate spatial gradients
    if len(pred.shape) == 3:  # batch_size, height, width
        # Horizontal gradients
        pred_grad_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        target_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        
        # Vertical gradients
        pred_grad_y = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        target_grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :])
        
        # Penalize when prediction gradients are much smaller than target gradients
        grad_loss_x = torch.relu(target_grad_x - pred_grad_x).mean()
        grad_loss_y = torch.relu(target_grad_y - pred_grad_y).mean()
        
        gradient_penalty = gradient_weight * (grad_loss_x + grad_loss_y)
    else:
        gradient_penalty = 0
    
    return weighted_mae + gradient_penalty

def train_model_pytorch(model, train_loader, val_loader, epochs=5, learning_rate=0.01):
    """Train the PyTorch model using weighted MAE loss for better high-concentration prediction."""
    print("\nStarting model training with weighted loss for high concentrations...")
    
    # Using custom weighted loss instead of standard L1Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = aggressive_weighted_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = aggressive_weighted_loss(outputs, batch_y)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Log to wandb
        wandb.log({
            "mae": avg_train_loss,
            "val_mae": avg_val_loss,
            "epoch": epoch
        })
        
        print(f'Epoch [{epoch+1}/{epochs}], Train MAE: {avg_train_loss:.6f}, Val MAE: {avg_val_loss:.6f}')
        
        # Debug: Show dropout effect and aggressive loss stats
        if epoch == 0:
            print(f"  Note: Using AGGRESSIVE weighted loss:")
            print(f"    - High pixels (>0.4) get 20x weight")
            print(f"    - Very high pixels (>0.7) get 40x weight") 
            print(f"    - Added gradient penalty to discourage smoothness")
            print(f"  This should force sharp, detailed predictions!")
            print(f"  Val loss < Train loss is normal with dropout - indicates good regularization!")
    
    return history

def evaluate_model(model, test_loader, timestamps, test_start_idx):
    """Evaluate the model and create visualizations using MAE only."""
    print("\nEvaluating model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    y_test = np.concatenate(all_targets, axis=0)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    
    # Calculate SSIM for each prediction
    ssim_scores = []
    for i in range(len(predictions)):
        score = ssim(y_test[i], predictions[i],
                    win_size=3, data_range=1.0, channel_axis=None)
        ssim_scores.append(score)
    
    mean_ssim = np.mean(ssim_scores)
    
    print(f"Test MAE: {mae:.6f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    
    # Analyze high-concentration prediction performance with aggressive thresholds
    high_threshold = 0.4
    very_high_threshold = 0.7
    
    high_mask = y_test.flatten() > high_threshold
    very_high_mask = y_test.flatten() > very_high_threshold
    low_mask = y_test.flatten() <= high_threshold
    
    if np.sum(high_mask) > 0:
        high_mae = mean_absolute_error(y_test.flatten()[high_mask], predictions.flatten()[high_mask])
        low_mae = mean_absolute_error(y_test.flatten()[low_mask], predictions.flatten()[low_mask])
        
        high_pixel_percentage = np.sum(high_mask) / len(high_mask) * 100
        very_high_pixel_percentage = np.sum(very_high_mask) / len(very_high_mask) * 100
        
        print(f"High-concentration pixels (>{high_threshold}): {high_pixel_percentage:.2f}% of total")
        print(f"Very high-concentration pixels (>{very_high_threshold}): {very_high_pixel_percentage:.2f}% of total")
        print(f"MAE on high-concentration pixels: {high_mae:.6f}")
        print(f"MAE on low-concentration pixels: {low_mae:.6f}")
        print(f"High/Low MAE ratio: {high_mae/low_mae:.2f} (lower is better)")
        
        if np.sum(very_high_mask) > 0:
            very_high_mae = mean_absolute_error(y_test.flatten()[very_high_mask], predictions.flatten()[very_high_mask])
            print(f"MAE on very high-concentration pixels: {very_high_mae:.6f}")
    
    # Create visualization
    n_predictions = min(len(predictions), 5)
    plt.figure(figsize=(20, 12))  # Made taller to accommodate difference plots
    
    # Debug: Check if predictions are different from true values
    print(f"\nDebug: Checking prediction differences...")
    for i in range(min(3, len(predictions))):
        true_img = y_test[i]
        pred_img = predictions[i]
        diff = np.abs(true_img - pred_img)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Image {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    for i in range(n_predictions):
        # True image
        plt.subplot(3, n_predictions, i + 1)
        true_img = y_test[i]
        plt.imshow(true_img, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'True Image {i+1}\nDate: {timestamps[test_start_idx + 4 + i].strftime("%Y-%m-%d")}')
        plt.axis('off')
        
        # Predicted image
        plt.subplot(3, n_predictions, n_predictions + i + 1)
        pred_img = predictions[i]
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
    plt.savefig('tsushima_lstm_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mae, mean_ssim, predictions

#_______________________________
# Main execution
#_______________________________

if __name__ == "__main__":
    print("=" * 60)
    print("TEMPORAL LSTM WITH PYTORCH AND PROPER DATA SPLITTING")
    print("=" * 60)
    
    # Initialize wandb
    wandb.init(
        project="mp-prediction-pytorch",
        config={
            "model_type": "CNN-LSTM",
            "framework": "PyTorch",
            "sequence_length": 4,
            "learning_rate": 0.01,
            "batch_size": 25,
            "epochs": 12,
            "image_resolution": "128x256",
            "loss_function": "aggressive_weighted_mae",
            "high_concentration_threshold": 0.4,
            "high_concentration_weight": 20.0,
            "very_high_concentration_threshold": 0.7,
            "very_high_concentration_weight": 30.0,
            "gradient_penalty_weight": 0.1
        }
    )
    
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
    '''
    # Verify no leakage
    no_leakage = verify_no_leakage(X_train, X_test, y_train, y_test, seq_length)
    
    if not no_leakage:
        print("ERROR: Data leakage detected! Check the splitting logic.")
        exit(1)
    '''
    
    # Create validation split from training data (temporal split)
    val_split_idx = int(len(X_train) * 0.6)
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
    batch_size = 25  # Reduced due to higher resolution
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    img_height, img_width = data[0].shape[:2]
    model = CNNLSTMModel(seq_length, img_height, img_width)
    
    print("\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_model_pytorch(model, train_loader, val_loader, epochs=8)
    
    # Evaluate model
    mae, mean_ssim, predictions = evaluate_model(model, test_loader, timestamps, test_start_idx)
    
    # Log final test metrics to wandb
    wandb.log({
        "test_mae": mae,
        "test_ssim": mean_ssim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Save training history plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training MAE')
    plt.plot(history['val_loss'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training MAE')
    plt.plot(history['val_loss'], label='Validation MAE')
    plt.title('Model MAE (Duplicate)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tsushima_lstm_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'model_type': 'Tsushima Region CNN-LSTM with PyTorch and Geographic Cropping',
        'framework': 'PyTorch',
        'device': str(device),
        'sequence_length': seq_length,
        'total_files': len(nc_files),
        'train_files': train_end_idx,
        'test_files': len(test_data),
        'gap_files': test_start_idx - train_end_idx,
        'model_parameters': {
            'total': int(total_params),
            'trainable': int(trainable_params)
        },
        'final_metrics': {
            'mae': float(mae),
            'ssim': float(mean_ssim)
        },
        'training_history': {
            'final_train_mae': float(history['train_loss'][-1]),
            'final_val_mae': float(history['val_loss'][-1])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('tsushima_lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PYTORCH RESULTS SUMMARY")
    print("=" * 60)
    print(f"✓ Framework: PyTorch")
    print(f"✓ Device: {device}")

    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training files: {train_end_idx}")
    print(f"✓ Gap files: {test_start_idx - train_end_idx}")
    print(f"✓ Test files: {len(test_data)}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Geographic region: Tsushima Strait (34.02837°N-34.76456°N, 129.11613°E-129.55801°E)")
    print("\nFiles generated:")
    print("- tsushima_lstm_predictions.png")
    print("- tsushima_lstm_training_history.png") 
    print("- tsushima_lstm_results.json")
    print("=" * 60)
    
    # Finish wandb run
    wandb.finish()