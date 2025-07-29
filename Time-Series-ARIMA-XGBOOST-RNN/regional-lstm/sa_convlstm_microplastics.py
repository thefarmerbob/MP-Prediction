import sys
print(sys.executable)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
import os
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import json
import wandb

# Import the SA-ConvLSTM model
from sa_convlstm import SA_ConvLSTM_Model

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get files for training
nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
print(f"Processing {len(nc_files)} files with proper temporal splitting")

class Args:
    """Configuration class for SA-ConvLSTM model"""
    def __init__(self):
        self.batch_size = 2  # Reduced further for debugging
        self.gpu_num = 1
        self.img_size = 64  # Reduced from 128 for faster debugging
        self.num_layers = 1  # Reduced from 2 for debugging
        self.frame_num = 3  # Reduced from 4 for debugging
        self.input_dim = 1  # single channel for microplastics concentration
        self.hidden_dim = 32  # Reduced from 64 for debugging
        self.learning_rate = 0.001
        self.epochs = 10  # Reduced for debugging
        self.patch_size = 4  # Reduced patch size for smaller images

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
    
    # Limit processing to first 50 files for debugging
    max_files = 50
    nc_files_limited = nc_files[:max_files]
    
    print(f"Processing {len(nc_files_limited)} files (limited from {len(nc_files)} total files)")
    print("Processing files and creating normalized Japan region images...")
    
    for i, nc_file in enumerate(nc_files_limited):
        if i % 10 == 0:
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
    """Create sequences from data array for SA-ConvLSTM."""
    sequences = []
    target_images = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length] 
        target_image = data[i + seq_length]
        sequences.append(seq)
        target_images.append(target_image)
    
    sequences = np.array(sequences)
    target_images = np.array(target_images)
    
    # Add channel dimension: (batch, seq_len, height, width) -> (batch, seq_len, 1, height, width)
    sequences = np.expand_dims(sequences, axis=2)
    target_images = np.expand_dims(target_images, axis=1)  # (batch, 1, height, width)
    
    return sequences, target_images

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
    pred_grad_x = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_grad_x = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    pred_grad_y = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    target_grad_y = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    
    edge_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x)) + \
                torch.mean(torch.abs(pred_grad_y - target_grad_y))
    
    # Combine all losses
    total_loss = weighted_mae + contrast_weight * contrast_loss + 0.1 * edge_loss
    
    return total_loss

def train_sa_convlstm(model, train_loader, val_loader, args):
    """Train the SA-ConvLSTM model."""
    print(f"\nStarting SA-ConvLSTM training for {args.epochs} epochs...")
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx % 5 == 0:
                print(f"  Training batch {batch_idx+1}/{len(train_loader)}")
            
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_x)  # SA-ConvLSTM output: (batch, seq_len, 1, height, width)
            
            # Take the last prediction from the sequence
            outputs = outputs[:, -1, :, :, :]  # (batch, 1, height, width)
            
            loss = improved_contrast_loss(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Log batch-level metrics to wandb
            if batch_idx % 10 == 0:  # Log every 10th batch
                wandb.log({
                    'batch_train_loss': loss.item(),
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # Validation phase
        print(f"  Starting validation...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                
                outputs = model(batch_x)
                outputs = outputs[:, -1, :, :, :]  # Take last prediction
                
                loss = improved_contrast_loss(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate additional metrics for logging
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        param_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"  Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Log epoch-level metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm,
            'param_norm': param_norm,
            'loss_difference': abs(avg_train_loss - avg_val_loss)
        })
    
    print(f"\nSA-ConvLSTM training completed successfully!")
    
    return history

def evaluate_sa_convlstm(model, test_loader, timestamps, test_start_idx):
    """Evaluate the SA-ConvLSTM model and create visualizations."""
    print("\nEvaluating SA-ConvLSTM model...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            outputs = model(batch_x)
            outputs = outputs[:, -1, :, :, :]  # Take last prediction
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Remove channel dimension for visualization
    predictions = predictions.squeeze(1)  # (batch, height, width)
    targets = targets.squeeze(1)
    
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
    
    # Calculate additional metrics for comprehensive evaluation
    mse = np.mean((targets.flatten() - predictions.flatten()) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate per-sample metrics for distribution analysis
    sample_maes = [np.mean(np.abs(targets[i] - predictions[i])) for i in range(len(predictions))]
    
    # Log detailed evaluation metrics to wandb
    wandb.log({
        "test_mae_detailed": mae,
        "test_mse": mse,
        "test_rmse": rmse,
        "test_ssim_detailed": mean_ssim,
        "min_sample_mae": np.min(sample_maes),
        "max_sample_mae": np.max(sample_maes),
        "std_sample_mae": np.std(sample_maes),
        "predictions_mean": np.mean(predictions),
        "predictions_std": np.std(predictions),
        "targets_mean": np.mean(targets),
        "targets_std": np.std(targets)
    })
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('SA-ConvLSTM Japan Region Predictions vs Actual\n(Self-Attention Convolutional LSTM)', fontsize=16)
    
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
        axes[1, i].set_title(f'SA-ConvLSTM Predicted\n{timestamps[test_start_idx + i].strftime("%Y-%m-%d")}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sa_convlstm_predictions.png', dpi=150, bbox_inches='tight')
    
    # Log the prediction visualization to wandb
    wandb.log({"predictions_visualization": wandb.Image('sa_convlstm_predictions.png')})
    
    # Create individual comparison images for wandb
    comparison_images = []
    for i in range(min(4, len(predictions))):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target
        target_flipped = np.flipud(targets[i])
        im1 = ax1.imshow(target_flipped, cmap='viridis', aspect='auto')
        ax1.set_title(f'Actual\n{timestamps[test_start_idx + i].strftime("%Y-%m-%d")}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Prediction
        pred_flipped = np.flipud(predictions[i])
        im2 = ax2.imshow(pred_flipped, cmap='viridis', aspect='auto')
        ax2.set_title(f'SA-ConvLSTM Predicted\n{timestamps[test_start_idx + i].strftime("%Y-%m-%d")}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Difference
        diff = np.abs(target_flipped - pred_flipped)
        im3 = ax3.imshow(diff, cmap='Reds', aspect='auto')
        ax3.set_title(f'Absolute Difference\nMAE: {np.mean(diff):.4f}')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        comparison_path = f'comparison_{i}.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        comparison_images.append(wandb.Image(comparison_path))
        plt.close()
    
    # Log individual comparisons to wandb
    wandb.log({"prediction_comparisons": comparison_images})
    
    plt.close()
    
    return mae, mean_ssim, predictions

#_______________________________
# Main execution
#_______________________________

if __name__ == "__main__":
    print("=" * 60)
    print("SA-CONVLSTM WITH MICROPLASTICS DATA - JAPAN REGION")
    print("=" * 60)
    
    # Create args object
    args = Args()
    
    # Extract timestamps and ensure temporal ordering
    timestamps = extract_timestamps_from_filenames(nc_files)
    print(f"Data spans from {timestamps[0]} to {timestamps[-1]}")
    
    # Extract data
    print("\nExtracting data...")
    data = extract_data_and_clusters(nc_files)
    
    # Update args with actual image size
    args.img_size = data[0].shape[0]  # Assuming square images
    print(f"\nUsing image size: {args.img_size}x{args.img_size}")
    print(f"Using sequence length: {args.frame_num}")
    
    # Proper temporal split
    train_data, test_data, train_end_idx, test_start_idx = temporal_train_test_split_proper(
        data, args.frame_num, test_ratio=0.2
    )
    
    # Create sequences separately for train and test
    X_train, y_train = create_sequences_from_data(train_data, args.frame_num)
    X_test, y_test = create_sequences_from_data(test_data, args.frame_num)
    
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
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize wandb
    wandb.init(
        project="sa-convlstm-microplastics",
        name=f"sa-convlstm-japan-{args.img_size}x{args.img_size}-h{args.hidden_dim}-l{args.num_layers}",
        config={
            "model_type": "SA-ConvLSTM Japan Region",
            "framework": "PyTorch",
            "sequence_length": args.frame_num,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "image_size": args.img_size,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "patch_size": args.patch_size,
            "device": str(device),
            "self_attention": True,
            "memory_module": True,
            "data_files": len(nc_files),
            "geographic_region": "Japan",
            "loss_function": "improved_contrast_loss"
        }
    )
    
    # Create model
    model = SA_ConvLSTM_Model(args)
    
    print("\nSA-ConvLSTM Model architecture:")
    print(f"- Input dimensions: {args.input_dim}")
    print(f"- Hidden dimensions: {args.hidden_dim}")
    print(f"- Number of layers: {args.num_layers}")
    print(f"- Sequence length: {args.frame_num}")
    print(f"- Image size: {args.img_size}x{args.img_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_sa_convlstm(model, train_loader, val_loader, args)
    
    # Evaluate model
    mae, mean_ssim, predictions = evaluate_sa_convlstm(model, test_loader, timestamps, test_start_idx)
    
    # Log final test metrics to wandb
    wandb.log({
        "test_mae": mae,
        "test_ssim": mean_ssim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Create a summary table for wandb
    summary_data = [
        ["Metric", "Value"],
        ["Test MAE", f"{mae:.6f}"],
        ["Test SSIM", f"{mean_ssim:.4f}"],
        ["Final Train Loss", f"{history['train_loss'][-1]:.6f}"],
        ["Final Val Loss", f"{history['val_loss'][-1]:.6f}"],
        ["Total Parameters", f"{total_params:,}"],
        ["Training Samples", f"{len(X_train_final):,}"],
        ["Test Samples", f"{len(X_test):,}"],
        ["Image Size", f"{args.img_size}x{args.img_size}"],
        ["Hidden Dim", f"{args.hidden_dim}"],
        ["Sequence Length", f"{args.frame_num}"],
        ["Device", str(device)]
    ]
    
    wandb.log({"model_summary": wandb.Table(data=summary_data, columns=["Metric", "Value"])})
    
    # Set wandb summary metrics for easy comparison across runs
    wandb.summary["final_test_mae"] = mae
    wandb.summary["final_test_ssim"] = mean_ssim
    wandb.summary["final_train_loss"] = history['train_loss'][-1]
    wandb.summary["final_val_loss"] = history['val_loss'][-1]
    wandb.summary["total_parameters"] = total_params
    wandb.summary["epochs_completed"] = args.epochs
    
    # Save training history plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('SA-ConvLSTM Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], 'o-', label='Train')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], 's-', label='Validation')
    plt.title('Loss Progress Detail')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sa_convlstm_training.png', dpi=150, bbox_inches='tight')
    
    # Log training history to wandb
    wandb.log({"training_history": wandb.Image('sa_convlstm_training.png')})
    
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'sa_convlstm_japan_microplastics.pth')
    
    # Save results
    results = {
        'model_type': 'SA-ConvLSTM Japan Region with Self-Attention',
        'framework': 'PyTorch',
        'device': str(device),
        'sequence_length': args.frame_num,
        'image_size': args.img_size,
        'training_samples': len(X_train_final),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'test_mae': mae,
        'test_ssim': mean_ssim,
        'model_parameters': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'self_attention': True,
            'memory_module': True
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        },
        'geographic_region': {
            'name': 'Japan',
            'southwest': [25.35753, 118.85766],
            'northeast': [36.98134, 145.47117],
            'description': 'SA-ConvLSTM model with self-attention for Japan region microplastics prediction'
        }
    }
    
    with open('sa_convlstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SA-CONVLSTM TRAINING COMPLETED")
    print("=" * 60)
    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training files: {train_end_idx}")
    print(f"✓ Test files: {len(test_data)}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Image resolution: {args.img_size}x{args.img_size}")
    print(f"✓ Self-attention enabled: Yes")
    print(f"✓ Memory module enabled: Yes")
    print("\nFiles generated:")
    print("- sa_convlstm_predictions.png")
    print("- sa_convlstm_training.png") 
    print("- sa_convlstm_results.json")
    print("- sa_convlstm_japan_microplastics.pth")
    print("=" * 60)
    
    # Finish wandb run
    wandb.finish() 