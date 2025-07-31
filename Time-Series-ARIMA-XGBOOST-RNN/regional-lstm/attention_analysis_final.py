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

def analyze_gradient_importance(model, input_sequence, target_region_pixels):
    """
    Use gradient-based analysis to understand spatial importance.
    """
    model.eval()
    model.zero_grad()
    
    # Create a copy that requires gradients
    input_copy = input_sequence.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(input_copy)
    
    # Extract the target region from the output
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    
    # Sum output values in the target region
    target_output = output[:, -1, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]].sum()
    
    # Backward pass to get gradients
    target_output.backward()
    
    # Get gradients with respect to input - use the last timestep
    input_gradients = input_copy.grad[0, -1, 0].detach().cpu().numpy()  # (H, W)
    
    # Take absolute value for importance
    importance_map = np.abs(input_gradients)
    
    return importance_map

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
        
        print(f"Running occlusion analysis with {patch_size}x{patch_size} patches...")
        total_patches = (height // patch_size) * (width // patch_size)
        patch_count = 0
        
        # Occlude patches and measure impact
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch_count += 1
                if patch_count % 10 == 0:
                    print(f"  Processing patch {patch_count}/{total_patches}")
                
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

def saliency_map_analysis(model, input_sequence, target_region_pixels):
    """
    Generate saliency map using vanilla gradients.
    """
    model.eval()
    model.zero_grad()
    
    # Prepare input with gradients
    input_copy = input_sequence.clone().detach()
    input_copy.requires_grad_(True)
    
    # Forward pass
    output = model(input_copy)
    
    # Get target region output
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    target_output = output[:, -1, 0, lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]].mean()
    
    # Compute gradients
    target_output.backward()
    
    # Get saliency map from gradients - average across time and channels
    saliency = input_copy.grad.abs().mean(dim=(0, 1, 2)).cpu().numpy()  # (H, W)
    
    return saliency

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

def visualize_spatial_importance(gradient_map, occlusion_map, saliency_map,
                                target_region_pixels, img_size=64):
    """Visualize the spatial importance maps"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Spatial Influence Analysis: Which Areas Affect Target Region\n'
                 f'Target Region: SW({target_region_pixels["original_coords"]["sw_lat"]:.4f}, '
                 f'{target_region_pixels["original_coords"]["sw_lon"]:.4f}) '
                 f'NE({target_region_pixels["original_coords"]["ne_lat"]:.4f}, '
                 f'{target_region_pixels["original_coords"]["ne_lon"]:.4f})', fontsize=14)
    
    # Plot maps
    maps = [
        ('Gradient-based Importance', gradient_map),
        ('Occlusion Sensitivity', occlusion_map),
        ('Saliency Map', saliency_map)
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
                           linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--', 
                           label='Target Region')
        axes[0, i].add_patch(rect)
        
        # Thresholded map (top 20%)
        threshold = np.percentile(importance_map, 80)
        thresholded = importance_map.copy()
        thresholded[thresholded < threshold] = 0
        
        im2 = axes[1, i].imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
        axes[1, i].set_title(f'{name}\n(Top 20% Most Influential Areas)')
        axes[1, i].axis('off')
        
        # Add target region overlay
        rect2 = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                            lon_range[1] - lon_range[0], 
                            lat_range[1] - lat_range[0],
                            linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[1, i].add_patch(rect2)
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('spatial_attention_analysis.png', dpi=150, bbox_inches='tight')
    
    # Create a combined importance map
    combined_importance = (gradient_map + occlusion_map + saliency_map) / 3
    
    # Create a detailed analysis figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Combined Spatial Influence Analysis', fontsize=14)
    
    # Combined map
    im1 = axes2[0].imshow(np.flipud(combined_importance), cmap='hot', aspect='auto')
    axes2[0].set_title('Combined Influence Map\n(Average of All Methods)')
    axes2[0].axis('off')
    
    # Add target region
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    axes2[0].add_patch(rect)
    plt.colorbar(im1, ax=axes2[0], fraction=0.046, pad=0.04)
    
    # Most influential areas only
    top_threshold = np.percentile(combined_importance, 90)
    top_influential = combined_importance.copy()
    top_influential[top_influential < top_threshold] = 0
    
    im2 = axes2[1].imshow(np.flipud(top_influential), cmap='hot', aspect='auto')
    axes2[1].set_title('Top 10% Most Influential Areas\n(Combined Analysis)')
    axes2[1].axis('off')
    
    rect2 = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                        lon_range[1] - lon_range[0], 
                        lat_range[1] - lat_range[0],
                        linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    axes2[1].add_patch(rect2)
    plt.colorbar(im2, ax=axes2[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('combined_influence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("SPATIAL INFLUENCE ANALYSIS RESULTS")
    print("="*70)
    print(f"Target Region Coordinates:")
    print(f"  Southwest: ({target_region_pixels['original_coords']['sw_lat']:.5f}, "
          f"{target_region_pixels['original_coords']['sw_lon']:.5f})")
    print(f"  Northeast: ({target_region_pixels['original_coords']['ne_lat']:.5f}, "
          f"{target_region_pixels['original_coords']['ne_lon']:.5f})")
    print(f"  Grid coordinates: lat[{target_region_pixels['lat_range'][0]}:{target_region_pixels['lat_range'][1]}], "
          f"lon[{target_region_pixels['lon_range'][0]}:{target_region_pixels['lon_range'][1]}]")
    
    all_maps = [
        ('Gradient-based Importance', gradient_map),
        ('Occlusion Sensitivity', occlusion_map),
        ('Saliency Map', saliency_map),
        ('Combined Influence', combined_importance)
    ]
    
    for name, importance_map in all_maps:
        print(f"\n{name}:")
        print(f"  Mean influence: {np.mean(importance_map):.6f}")
        print(f"  Max influence: {np.max(importance_map):.6f}")
        print(f"  Std/Mean ratio: {np.std(importance_map)/np.mean(importance_map):.4f}")
        
        # Find top 5 most important locations
        flat_indices = np.argsort(importance_map.flatten())[-5:]
        print(f"  Top 5 most influential locations (grid coordinates):")
        for j, idx in enumerate(reversed(flat_indices)):
            lat_idx, lon_idx = np.unravel_index(idx, importance_map.shape)
            influence = importance_map[lat_idx, lon_idx]
            print(f"    {j+1}. Grid({lat_idx}, {lon_idx}) - Influence: {influence:.6f}")
    
    return combined_importance

def main():
    """Main function to analyze spatial importance for the target region"""
    
    # Target region coordinates
    target_sw_lat, target_sw_lon = 34.02837, 129.11613
    target_ne_lat, target_ne_lon = 34.76456, 129.55801
    
    print("="*70)
    print("SPATIAL ATTENTION ANALYSIS: WHICH AREAS AFFECT TARGET REGION")
    print("="*70)
    print(f"Target Region (appears to be near Tsushima area):")
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
            self.batch_size = 1
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
    
    print("\nLoading trained SA-ConvLSTM model...")
    model = SA_ConvLSTM_Model(args)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load sample data
    print("\nLoading recent microplastic data for analysis...")
    from sa_convlstm_microplastics import extract_data_and_clusters, create_sequences_from_data
    
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    data = extract_data_and_clusters(nc_files[-8:])  # Use last 8 files for robust analysis
    
    X_sample, _ = create_sequences_from_data(data, args.frame_num)
    
    # Take a representative sample for analysis
    sample_data = torch.FloatTensor(X_sample[2:3]).to(device)  # Middle sample
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Analyzing how spatial areas influence target region predictions...")
    
    # Perform different types of analysis
    print("\n1. Performing gradient-based analysis...")
    gradient_importance = analyze_gradient_importance(model, sample_data, target_region_pixels)
    
    print("2. Performing saliency map analysis...")
    saliency_importance = saliency_map_analysis(model, sample_data, target_region_pixels)
    
    print("3. Performing occlusion sensitivity analysis...")
    occlusion_importance = occlusion_sensitivity_analysis(model, sample_data, target_region_pixels, patch_size=4)
    
    # Visualize results
    print("\n4. Generating comprehensive visualization...")
    combined_importance = visualize_spatial_importance(
        gradient_importance, 
        occlusion_importance,
        saliency_importance,
        target_region_pixels
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("✓ spatial_attention_analysis.png")
    print("✓ combined_influence_analysis.png")
    print("\nInterpretation:")
    print("- Brighter/hotter colors indicate areas that more strongly influence")
    print("  microplastic concentration in your target region")
    print("- The cyan dashed box shows your target region")
    print("- Use the 'Top 20%' maps to focus on the most influential areas")
    print("- The combined analysis provides the most reliable results")

if __name__ == "__main__":
    main() 