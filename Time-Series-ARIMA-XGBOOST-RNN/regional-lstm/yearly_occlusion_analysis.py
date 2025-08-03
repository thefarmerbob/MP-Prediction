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
from datetime import datetime, timedelta
import cv2
from PIL import Image
import imageio

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the existing model and functions
from sa_convlstm import SA_ConvLSTM_Model
from sa_convlstm_microplastics import extract_data_and_clusters, create_sequences_from_data
from attention_analysis_final import (
    occlusion_sensitivity_analysis, 
    map_region_to_grid, 
    extract_geographic_coords_from_data
)

def create_yearly_forecast_data(nc_files, num_forecasts=365):
    """
    Create data for yearly forecasts by sampling from different time periods.
    """
    print(f"Creating {num_forecasts} forecast starting points from {len(nc_files)} files...")
    
    # Load all data
    all_data = extract_data_and_clusters(nc_files)
    print(f"Total data points: {len(all_data)}")
    
    # Create sequences for the entire dataset
    frame_num = 3  # Use 3 frames for sequence
    X_all, y_all = create_sequences_from_data(all_data, frame_num)
    
    print(f"Total sequences available: {len(X_all)}")
    
    # Sample evenly across the year
    if len(X_all) < num_forecasts:
        print(f"Warning: Only {len(X_all)} sequences available, using all of them")
        num_forecasts = len(X_all)
    
    # Create evenly spaced indices
    indices = np.linspace(0, len(X_all)-1, num_forecasts, dtype=int)
    
    forecast_data = []
    for i, idx in enumerate(indices):
        forecast_data.append({
            'sequence': X_all[idx],
            'target': y_all[idx] if idx < len(y_all) else None,
            'day': i + 1,
            'original_idx': idx
        })
    
    print(f"Created {len(forecast_data)} forecast starting points")
    return forecast_data

def run_occlusion_for_forecast(model, forecast_data_point, target_region_pixels, patch_size=4):
    """
    Run occlusion analysis for a single forecast.
    """
    # Prepare input sequence
    input_sequence = torch.FloatTensor(forecast_data_point['sequence'][np.newaxis, ...]).to(device)
    
    # Run occlusion sensitivity analysis
    sensitivity_map = occlusion_sensitivity_analysis(
        model, input_sequence, target_region_pixels, patch_size=patch_size
    )
    
    return sensitivity_map

def create_occlusion_image(sensitivity_map, target_region_pixels, day, img_size=64, threshold_percentile=80):
    """
    Create an image showing the top 20% most influential areas (threshold_percentile=80).
    """
    # Apply threshold to show top 20% most influential areas
    threshold = np.percentile(sensitivity_map, threshold_percentile)
    thresholded = sensitivity_map.copy()
    thresholded[thresholded < threshold] = 0
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot the thresholded sensitivity map
    im = ax.imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
    ax.set_title(f'Day {day}: Top 20% Most Influential Areas\n(Occlusion Sensitivity)', fontsize=14)
    ax.axis('off')
    
    # Add target region overlay
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Save the image
    output_dir = Path('yearly_occlusion_frames')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f'occlusion_day_{day:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def create_gif_from_images(image_dir, output_filename='yearly_occlusion_analysis.gif', duration=0.1):
    """
    Create a GIF from the sequence of occlusion images.
    """
    image_files = sorted(Path(image_dir).glob('occlusion_day_*.png'))
    
    if not image_files:
        print("No images found to create GIF!")
        return None
    
    print(f"Creating GIF from {len(image_files)} images...")
    
    # Read all images and resize to common dimensions
    images = []
    target_size = None
    
    for i, img_file in enumerate(image_files):
        img = imageio.v2.imread(img_file)  # Use v2 to avoid deprecation warning
        
        # Set target size from first image
        if target_size is None:
            target_size = img.shape[:2]  # (height, width)
            print(f"Target image size: {target_size}")
        
        # Resize if needed
        if img.shape[:2] != target_size:
            # Resize using PIL for better quality
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            img = np.array(img_pil)
        
        images.append(img)
        
        if i % 50 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images...")
    
    # Create GIF
    output_path = Path(output_filename)
    imageio.v2.mimsave(output_path, images, duration=duration, loop=0)
    
    print(f"GIF saved as: {output_path}")
    return output_path

def main():
    """Main function to create yearly occlusion analysis GIF."""
    
    print("="*70)
    print("YEARLY OCCLUSION SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Target region coordinates (Tsushima area)
    target_sw_lat, target_sw_lon = 34.02837, 129.11613
    target_ne_lat, target_ne_lon = 34.76456, 129.55801
    
    print(f"Target Region (Tsushima area):")
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
    
    # Load data files
    print("\nLoading microplastic data files...")
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    
    if len(nc_files) == 0:
        print("Error: No NetCDF files found!")
        return
    
    print(f"Found {len(nc_files)} data files")
    
    # Create yearly forecast data
    num_forecasts = min(365, len(nc_files) // 3)  # Limit based on available data
    print(f"\nCreating {num_forecasts} forecasts for yearly analysis...")
    
    forecast_data = create_yearly_forecast_data(nc_files, num_forecasts)
    
    # Process each forecast
    print("\nRunning occlusion analysis for each forecast...")
    
    for i, forecast_point in enumerate(forecast_data):
        if i % 10 == 0:
            print(f"Processing forecast {i+1}/{len(forecast_data)} (Day {forecast_point['day']})...")
        
        # Run occlusion analysis
        sensitivity_map = run_occlusion_for_forecast(
            model, forecast_point, target_region_pixels, patch_size=4
        )
        
        # Create and save image
        image_file = create_occlusion_image(
            sensitivity_map, target_region_pixels, forecast_point['day']
        )
    
    # Create GIF from all images
    print("\nCreating GIF from all occlusion images...")
    gif_path = create_gif_from_images(
        'yearly_occlusion_frames', 
        'yearly_occlusion_analysis.gif', 
        duration=0.1  # 10 FPS
    )
    
    print("\n" + "="*70)
    print("YEARLY OCCLUSION ANALYSIS COMPLETE")
    print("="*70)
    print(f"Generated files:")
    print(f"✓ {len(forecast_data)} individual occlusion images in 'yearly_occlusion_frames/'")
    print(f"✓ yearly_occlusion_analysis.gif")
    print("\nInterpretation:")
    print("- The GIF shows how the most influential areas change throughout the year")
    print("- Brighter/hotter colors indicate areas that more strongly influence")
    print("  microplastic concentration in the Tsushima region")
    print("- The cyan dashed box shows the target region (Tsushima)")
    print("- Only the top 20% most influential areas are shown in each frame")

if __name__ == "__main__":
    main()