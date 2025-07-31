import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from skimage.transform import resize
import pandas as pd
from datetime import datetime

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the existing model
from sa_convlstm import SA_ConvLSTM_Model

def create_attention_hooks(model):
    """Create hooks to capture attention weights from the model"""
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(module, 'attention_layer'):
            # This is a ConvLSTM cell with attention
            attention_weights.append({
                'layer_name': str(module),
                'input_shape': input[0].shape if input else None,
                'output_shape': output[0].shape if isinstance(output, tuple) else output.shape
            })
    
    # Register hooks on attention layers
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower():
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    return hooks, attention_weights

def analyze_gradient_based_attention(model, input_sequence, target_region_pixels):
    """
    Use gradient-based analysis to understand spatial importance.
    This is a simpler alternative to extracting attention weights directly.
    """
    model.eval()
    
    # Enable gradients for input
    input_sequence.requires_grad_(True)
    
    # Forward pass
    output = model(input_sequence)
    
    # Extract the target region from the output
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    
    # Sum output values in the target region
    target_output = output[:, -1, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]].sum()
    
    # Backward pass to get gradients
    target_output.backward()
    
    # Get gradients with respect to input
    input_gradients = input_sequence.grad.detach().cpu().numpy()
    
    # Average gradients across time steps and batch
    avg_gradients = np.mean(np.abs(input_gradients), axis=(0, 1, 2))  # (H, W)
    
    return avg_gradients

def integrated_gradients_analysis(model, input_sequence, target_region_pixels, steps=20):
    """
    Compute integrated gradients to understand which input regions are most important
    for predicting the target region.
    """
    model.eval()
    
    # Create baseline (zeros)
    baseline = torch.zeros_like(input_sequence)
    
    # Create interpolated inputs
    alphas = torch.linspace(0, 1, steps).to(device)
    integrated_gradients = torch.zeros_like(input_sequence[0, 0, 0])  # (H, W)
    
    for alpha in alphas:
        # Interpolate between baseline and input
        interpolated_input = baseline + alpha * (input_sequence - baseline)
        interpolated_input.requires_grad_(True)
        
        # Forward pass
        output = model(interpolated_input)
        
        # Extract target region
        lat_range = target_region_pixels['lat_range']
        lon_range = target_region_pixels['lon_range']
        target_output = output[:, -1, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]].sum()
        
        # Backward pass
        if interpolated_input.grad is not None:
            interpolated_input.grad.zero_()
        target_output.backward(retain_graph=True)
        
        # Accumulate gradients
        gradients = interpolated_input.grad[0, -1, 0].detach()  # Last timestep, first channel
        integrated_gradients += gradients
    
    # Average and multiply by (input - baseline)
    integrated_gradients = integrated_gradients / steps
    integrated_gradients = integrated_gradients * (input_sequence[0, -1, 0] - baseline[0, -1, 0]).detach()
    
    return integrated_gradients.cpu().numpy()

def occlusion_sensitivity_analysis(model, input_sequence, target_region_pixels, patch_size=8):
    """
    Perform occlusion sensitivity analysis to understand spatial importance.
    """
    model.eval()
    
    with torch.no_grad():
        # Get baseline prediction
        baseline_output = model(input_sequence)
        lat_range = target_region_pixels['lat_range']
        lon_range = target_region_pixels['lon_range']
        baseline_target = baseline_output[:, -1, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]].sum()
        
        # Get input dimensions
        _, seq_len, channels, height, width = input_sequence.shape
        
        # Initialize sensitivity map
        sensitivity_map = np.zeros((height, width))
        
        # Occlude patches and measure impact
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                # Create occluded input
                occluded_input = input_sequence.clone()
                
                # Occlude the patch across all timesteps
                end_i = min(i + patch_size, height)
                end_j = min(j + patch_size, width)
                occluded_input[:, :, :, i:end_i, j:end_j] = 0
                
                # Get prediction with occlusion
                occluded_output = model(occluded_input)
                occluded_target = occluded_output[:, -1, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]].sum()
                
                # Calculate sensitivity (difference from baseline)
                sensitivity = (baseline_target - occluded_target).abs().item()
                
                # Assign sensitivity to all pixels in the patch
                sensitivity_map[i:end_i, j:end_j] = sensitivity
        
        return sensitivity_map

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def extract_geographic_coords_from_data():
    """Extract latitude and longitude coordinates from the first NetCDF file"""
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    if not nc_files:
        raise FileNotFoundError("No NetCDF files found")
    
    ds = xr.open_dataset(nc_files[0])
    
    # Check for different possible coordinate variable names
    possible_lat_names = ['lat', 'latitude', 'y', 'lat_1', 'lat_2']
    possible_lon_names = ['lon', 'longitude', 'x', 'lon_1', 'lon_2']
    
    lats, lons = None, None
    
    for lat_name in possible_lat_names:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            break
    
    for lon_name in possible_lon_names:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            break
    
    if lats is None or lons is None:
        raise ValueError("Could not find latitude/longitude coordinates in the data")
    
    return lats, lons

def map_region_to_grid(target_sw_lat, target_sw_lon, target_ne_lat, target_ne_lon, 
                      full_lats, full_lons, img_size=64):
    """Map the target geographic region to the model grid coordinates."""
    
    # Japan region bounds used in the model
    japan_sw_lat, japan_sw_lon = 25.35753, 118.85766
    japan_ne_lat, japan_ne_lon = 36.98134, 145.47117
    
    # Convert Japan bounds to data indices
    japan_sw_lat_idx, japan_sw_lon_idx = lat_lon_to_indices(japan_sw_lat, japan_sw_lon, full_lats, full_lons)
    japan_ne_lat_idx, japan_ne_lon_idx = lat_lon_to_indices(japan_ne_lat, japan_ne_lon, full_lats, full_lons)
    
    # Ensure proper ordering
    japan_lat_start = min(japan_sw_lat_idx, japan_ne_lat_idx)
    japan_lat_end = max(japan_sw_lat_idx, japan_ne_lat_idx)
    japan_lon_start = min(japan_sw_lon_idx, japan_ne_lon_idx)
    japan_lon_end = max(japan_sw_lon_idx, japan_ne_lon_idx)
    
    # Get cropped coordinate arrays for Japan region
    japan_lats = full_lats[japan_lat_start:japan_lat_end]
    japan_lons = full_lons[japan_lon_start:japan_lon_end]
    
    # Convert target region to Japan-cropped indices
    target_sw_lat_idx, target_sw_lon_idx = lat_lon_to_indices(target_sw_lat, target_sw_lon, japan_lats, japan_lons)
    target_ne_lat_idx, target_ne_lon_idx = lat_lon_to_indices(target_ne_lat, target_ne_lon, japan_lats, japan_lons)
    
    # Ensure proper ordering
    target_lat_start = min(target_sw_lat_idx, target_ne_lat_idx)
    target_lat_end = max(target_sw_lat_idx, target_ne_lat_idx)
    target_lon_start = min(target_sw_lon_idx, target_ne_lon_idx)
    target_lon_end = max(target_sw_lon_idx, target_ne_lon_idx)
    
    # Scale to model grid size (64x64)
    japan_shape = (japan_lat_end - japan_lat_start, japan_lon_end - japan_lon_start)
    
    lat_scale = img_size / japan_shape[0]
    lon_scale = img_size / japan_shape[1]
    
    # Convert target region to model grid coordinates
    target_lat_start_grid = int(target_lat_start * lat_scale)
    target_lat_end_grid = int(target_lat_end * lat_scale)
    target_lon_start_grid = int(target_lon_start * lon_scale)
    target_lon_end_grid = int(target_lon_end * lon_scale)
    
    return {
        'lat_range': (target_lat_start_grid, target_lat_end_grid),
        'lon_range': (target_lon_start_grid, target_lon_end_grid),
        'original_coords': {
            'sw_lat': target_sw_lat,
            'sw_lon': target_sw_lon,
            'ne_lat': target_ne_lat,
            'ne_lon': target_ne_lon
        }
    }

def visualize_spatial_importance(gradient_map, integrated_gradients_map, occlusion_map, 
                                target_region_pixels, img_size=64):
    """Visualize the spatial importance maps"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Spatial Influence Analysis for Target Region\n'
                 f'SW({target_region_pixels["original_coords"]["sw_lat"]:.4f}, '
                 f'{target_region_pixels["original_coords"]["sw_lon"]:.4f}) '
                 f'NE({target_region_pixels["original_coords"]["ne_lat"]:.4f}, '
                 f'{target_region_pixels["original_coords"]["ne_lon"]:.4f})', fontsize=14)
    
    # Plot maps
    maps = [
        ('Gradient-based Importance', gradient_map),
        ('Integrated Gradients', integrated_gradients_map),
        ('Occlusion Sensitivity', occlusion_map)
    ]
    
    for i, (name, importance_map) in enumerate(maps):
        # Original map
        im1 = axes[0, i].imshow(np.flipud(importance_map), cmap='hot', aspect='auto')
        axes[0, i].set_title(f'{name}\n(All Influential Areas)')
        axes[0, i].axis('off')
        
        # Add target region overlay
        lat_range = target_region_pixels['lat_range']
        lon_range = target_region_pixels['lon_range']
        
        rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                           lon_range[1] - lon_range[0], 
                           lat_range[1] - lat_range[0],
                           linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[0, i].add_patch(rect)
        
        # Thresholded map (top 20%)
        threshold = np.percentile(importance_map, 80)
        thresholded = importance_map.copy()
        thresholded[thresholded < threshold] = 0
        
        im2 = axes[1, i].imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
        axes[1, i].set_title(f'{name}\n(Top 20% Most Influential)')
        axes[1, i].axis('off')
        
        # Add target region overlay
        rect2 = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                            lon_range[1] - lon_range[0], 
                            lat_range[1] - lat_range[0],
                            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[1, i].add_patch(rect2)
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('spatial_importance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("SPATIAL IMPORTANCE ANALYSIS RESULTS")
    print("="*60)
    print(f"Target Region Coordinates:")
    print(f"  SW: ({target_region_pixels['original_coords']['sw_lat']:.5f}, "
          f"{target_region_pixels['original_coords']['sw_lon']:.5f})")
    print(f"  NE: ({target_region_pixels['original_coords']['ne_lat']:.5f}, "
          f"{target_region_pixels['original_coords']['ne_lon']:.5f})")
    print(f"  Grid coordinates: lat[{target_region_pixels['lat_range'][0]}:{target_region_pixels['lat_range'][1]}], "
          f"lon[{target_region_pixels['lon_range'][0]}:{target_region_pixels['lon_range'][1]}]")
    
    for name, importance_map in maps:
        print(f"\n{name}:")
        print(f"  Mean importance: {np.mean(importance_map):.6f}")
        print(f"  Max importance: {np.max(importance_map):.6f}")
        print(f"  Std/Mean ratio: {np.std(importance_map)/np.mean(importance_map):.4f}")
        
        # Find top 5 most important locations
        flat_indices = np.argsort(importance_map.flatten())[-5:]
        print(f"  Top 5 most important locations (grid coordinates):")
        for j, idx in enumerate(reversed(flat_indices)):
            lat_idx, lon_idx = np.unravel_index(idx, importance_map.shape)
            importance = importance_map[lat_idx, lon_idx]
            print(f"    {j+1}. ({lat_idx}, {lon_idx}) - Importance: {importance:.6f}")

def main():
    """Main function to analyze spatial importance for the target region"""
    
    # Target region coordinates
    target_sw_lat, target_sw_lon = 34.02837, 129.11613
    target_ne_lat, target_ne_lon = 34.76456, 129.55801
    
    print("="*60)
    print("SPATIAL IMPORTANCE ANALYSIS FOR MICROPLASTIC PREDICTION")
    print("="*60)
    print(f"Target Region:")
    print(f"  Southwest: ({target_sw_lat}, {target_sw_lon})")
    print(f"  Northeast: ({target_ne_lat}, {target_ne_lon})")
    
    # Load geographic coordinates
    print("\nExtracting geographic coordinates...")
    try:
        full_lats, full_lons = extract_geographic_coords_from_data()
        print(f"Coordinate grid shape: {full_lats.shape} x {full_lons.shape}")
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return
    
    # Set up model configuration
    class Args:
        def __init__(self):
            self.batch_size = 1  # Use batch size 1 for analysis
            self.gpu_num = 1
            self.img_size = 64
            self.num_layers = 1
            self.frame_num = 3
            self.input_dim = 1
            self.hidden_dim = 32
            self.patch_size = 4
    
    args = Args()
    
    # Map target region to grid
    print("\nMapping target region to model grid...")
    target_region_pixels = map_region_to_grid(
        target_sw_lat, target_sw_lon, target_ne_lat, target_ne_lon,
        full_lats, full_lons, args.img_size
    )
    
    print(f"Target region mapped to grid: {target_region_pixels}")
    
    # Load trained model
    model_path = 'sa_convlstm_japan_microplastics.pth'
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        print("Please run the training script first to generate the model weights.")
        return
    
    print("\nLoading trained model...")
    model = SA_ConvLSTM_Model(args)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load sample data
    print("\nLoading sample data...")
    from sa_convlstm_microplastics import extract_data_and_clusters, create_sequences_from_data
    
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    data = extract_data_and_clusters(nc_files[-5:])  # Use last 5 files
    
    X_sample, _ = create_sequences_from_data(data, args.frame_num)
    
    # Take one sample for analysis
    sample_data = torch.FloatTensor(X_sample[0:1]).to(device)  # Shape: (1, seq_len, 1, H, W)
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Perform different types of analysis
    print("\nPerforming gradient-based analysis...")
    gradient_importance = analyze_gradient_based_attention(model, sample_data, target_region_pixels)
    
    print("Performing integrated gradients analysis...")
    integrated_gradients_importance = integrated_gradients_analysis(model, sample_data, target_region_pixels)
    
    print("Performing occlusion sensitivity analysis...")
    occlusion_importance = occlusion_sensitivity_analysis(model, sample_data, target_region_pixels)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_spatial_importance(
        gradient_importance, 
        integrated_gradients_importance, 
        occlusion_importance,
        target_region_pixels
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- spatial_importance_analysis.png")
    print("\nThe analysis shows which spatial areas most influence")
    print("microplastic concentration predictions in your target region.")

if __name__ == "__main__":
    main() 