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
import cv2

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SA_Memory_Module_WithAttention(nn.Module):
    """Modified SA Memory Module that returns attention weights"""
    def __init__(self, input_dim, hidden_dim, patch_size=8):
        super().__init__()
        self.layer_qh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_kh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_vh = nn.Conv2d(input_dim, hidden_dim, 1)
        
        self.layer_km = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_vm = nn.Conv2d(input_dim, hidden_dim, 1)
        
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.patch_size = patch_size
        
    def forward(self, h, m, return_attention=False):
        batch_size, channel, H, W = h.shape

        # Use patch-based attention to reduce memory usage
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # Apply convolutions on original full resolution
        K_h = self.layer_kh(h)
        Q_h = self.layer_qh(h)
        V_h = self.layer_vh(h)
        
        # Work with patches for attention computation
        K_h_patches = K_h.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        Q_h_patches = Q_h.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        V_h_patches = V_h.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        
        Q_h_patches = Q_h_patches.transpose(1, 2)  # batch_size, patch_h*patch_w, hidden_dim
        
        A_h = torch.softmax(torch.bmm(Q_h_patches, K_h_patches), dim=-1)  # batch_size, patches, patches
        Z_h_patches = torch.matmul(A_h, V_h_patches.permute(0, 2, 1))  # batch_size, patches, hidden_dim

        K_m = self.layer_km(m)
        V_m = self.layer_vm(m)
        
        K_m_patches = K_m.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        V_m_patches = V_m.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        
        A_m = torch.softmax(torch.bmm(Q_h_patches, K_m_patches), dim=-1)
        Z_m_patches = torch.matmul(A_m, V_m_patches.permute(0, 2, 1))
        
        # Interpolate back to full resolution
        Z_h_patches = Z_h_patches.transpose(1, 2).view(batch_size, self.input_dim, patch_h, patch_w)
        Z_m_patches = Z_m_patches.transpose(1, 2).view(batch_size, self.input_dim, patch_h, patch_w)
        
        # Upsample patches back to original resolution
        Z_h = F.interpolate(Z_h_patches, size=(H, W), mode='bilinear', align_corners=False)
        Z_m = F.interpolate(Z_m_patches, size=(H, W), mode='bilinear', align_corners=False)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        
        ## Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.chunk(combined, chunks=3, dim=1)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m 

        if return_attention:
            # Reshape attention maps back to spatial grid for visualization
            A_h_spatial = A_h.view(batch_size, patch_h, patch_w, patch_h, patch_w)
            A_m_spatial = A_m.view(batch_size, patch_h, patch_w, patch_h, patch_w)
            return new_h, new_m, A_h_spatial, A_m_spatial
        
        return new_h, new_m

class SA_Convlstm_cell_WithAttention(nn.Module):
    """Modified SA ConvLSTM cell that can return attention weights"""
    def __init__(self, input_dim, hid_dim, patch_size=8):
        super().__init__()
        self.input_channels = input_dim
        self.hidden_dim = hid_dim
        self.kernel_size = 3
        self.padding = 1
        self.attention_layer = SA_Memory_Module_WithAttention(hid_dim, hid_dim, patch_size=patch_size)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels + self.hidden_dim, 
                     out_channels=4 * self.hidden_dim, 
                     kernel_size=self.kernel_size, padding=self.padding),
            nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim))    

    def forward(self, x, hidden, return_attention=False):
        c, h, m = hidden
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv2d(combined)
        i, f, g, o = torch.chunk(combined_conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = torch.mul(f, c) + torch.mul(i, g)
        h_next = torch.mul(o, torch.tanh(c_next))
        
        # Self-Attention
        if return_attention:
            h_next, m_next, A_h, A_m = self.attention_layer(h_next, m, return_attention=True)
            return h_next, (c_next, h_next, m_next), A_h, A_m
        else:
            h_next, m_next = self.attention_layer(h_next, m, return_attention=False)
            return h_next, (c_next, h_next, m_next)

class SA_ConvLSTM_Model_WithAttention(nn.Module):
    """Modified SA ConvLSTM model that can extract attention weights"""
    def __init__(self, args):
        super(SA_ConvLSTM_Model_WithAttention, self).__init__()
        self.batch_size = args.batch_size // args.gpu_num
        self.img_size = (args.img_size, args.img_size)
        self.cells, self.bns = [], []
        self.n_layers = args.num_layers
        self.frame_num = args.frame_num
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.patch_size = getattr(args, 'patch_size', 8)
        
        self.linear_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1, stride=1)
        
        for i in range(self.n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            hidden_dim = self.hidden_dim
            self.cells.append(SA_Convlstm_cell_WithAttention(input_dim, hidden_dim, patch_size=self.patch_size))
            self.bns.append(nn.LayerNorm((self.hidden_dim, *self.img_size)))

        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, X, hidden=None, return_attention=False):
        actual_batch_size = X.size(0)
        if hidden == None:
            hidden = self.init_hidden(batch_size=actual_batch_size, img_size=self.img_size)
        
        predict = []
        attention_maps = [] if return_attention else None
        inputs_x = None
        
        # Process sequence
        for t in range(X.size(1)):
            inputs_x = X[:, t, :, :, :]
            for i, layer in enumerate(self.cells):
                if return_attention and i == 0:  # Extract attention from first layer
                    inputs_x, hidden[i], A_h, A_m = layer(inputs_x, hidden[i], return_attention=True)
                    if attention_maps is not None:
                        attention_maps.append((A_h.detach().cpu(), A_m.detach().cpu()))
                else:
                    inputs_x, hidden[i] = layer(inputs_x, hidden[i], return_attention=False)
                inputs_x = self.bns[i](inputs_x)

        # Generate predictions
        inputs_x = X[:, -1, :, :, :]
        for t in range(X.size(1)):
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i], return_attention=False)
                inputs_x = self.bns[i](inputs_x)
                
            inputs_x = self.linear_conv(inputs_x)
            predict.append(inputs_x)
        
        predict = torch.stack(predict, dim=1)

        if return_attention:
            return torch.sigmoid(predict), attention_maps
        return torch.sigmoid(predict)

    def init_hidden(self, batch_size, img_size, device=None):
        h, w = img_size
        if device is None:
            device = next(self.parameters()).device
        hidden_state = (torch.zeros(batch_size, self.hidden_dim, h, w).to(device),
                        torch.zeros(batch_size, self.hidden_dim, h, w).to(device),
                        torch.zeros(batch_size, self.hidden_dim, h, w).to(device))
        states = [] 
        for i in range(self.n_layers):
            states.append(hidden_state)
        return states

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

def map_region_to_attention_grid(target_sw_lat, target_sw_lon, target_ne_lat, target_ne_lon, 
                                full_lats, full_lons, attention_grid_size, patch_size):
    """
    Map the target geographic region to the attention grid coordinates.
    
    Args:
        target_sw_lat, target_sw_lon: Southwest corner of target region
        target_ne_lat, target_ne_lon: Northeast corner of target region
        full_lats, full_lons: Full coordinate arrays from data
        attention_grid_size: Size of the attention grid (patches)
        patch_size: Size of each patch in the attention mechanism
    """
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
    
    # Convert to attention grid coordinates (considering downsampling to 64x64 and patch size)
    japan_shape = (japan_lat_end - japan_lat_start, japan_lon_end - japan_lon_start)
    downsampled_shape = (64, 64)  # Model uses 64x64 images
    
    # Scale factors from original Japan region to downsampled grid
    lat_scale = downsampled_shape[0] / japan_shape[0]
    lon_scale = downsampled_shape[1] / japan_shape[1]
    
    # Convert target region to downsampled grid coordinates
    target_lat_start_ds = int(target_lat_start * lat_scale)
    target_lat_end_ds = int(target_lat_end * lat_scale)
    target_lon_start_ds = int(target_lon_start * lon_scale)
    target_lon_end_ds = int(target_lon_end * lon_scale)
    
    # Convert to patch coordinates (for attention grid)
    target_patch_lat_start = target_lat_start_ds // patch_size
    target_patch_lat_end = target_lat_end_ds // patch_size
    target_patch_lon_start = target_lon_start_ds // patch_size
    target_patch_lon_end = target_lon_end_ds // patch_size
    
    return {
        'target_region_patches': {
            'lat_range': (target_patch_lat_start, target_patch_lat_end),
            'lon_range': (target_patch_lon_start, target_patch_lon_end)
        },
        'target_region_pixels': {
            'lat_range': (target_lat_start_ds, target_lat_end_ds),
            'lon_range': (target_lon_start_ds, target_lon_end_ds)
        },
        'japan_bounds': {
            'lat_range': (japan_lat_start, japan_lat_end),
            'lon_range': (japan_lon_start, japan_lon_end)
        }
    }

def analyze_attention_for_region(model, data_sequence, target_region_info, patch_size=4):
    """
    Analyze which spatial areas most influence the target region.
    
    Args:
        model: Trained SA-ConvLSTM model
        data_sequence: Input sequence for analysis
        target_region_info: Dictionary with target region coordinates
        patch_size: Patch size used in attention mechanism
    """
    model.eval()
    
    with torch.no_grad():
        # Run inference with attention extraction
        predictions, attention_maps = model(data_sequence, return_attention=True)
        
        # Extract target region coordinates
        target_patches = target_region_info['target_region_patches']
        target_lat_range = target_patches['lat_range']
        target_lon_range = target_patches['lon_range']
        
        # Analyze attention maps
        attention_analysis = []
        
        for t, (A_h, A_m) in enumerate(attention_maps):
            # A_h and A_m have shape: (batch, patch_h, patch_w, patch_h, patch_w)
            batch_size, patch_h, patch_w, _, _ = A_h.shape
            
            # Average across batch
            A_h_avg = A_h.mean(dim=0)  # (patch_h, patch_w, patch_h, patch_w)
            A_m_avg = A_m.mean(dim=0)
            
            # Extract attention weights for target region
            target_attention_h = np.zeros((patch_h, patch_w))
            target_attention_m = np.zeros((patch_h, patch_w))
            
            # Sum attention weights that influence the target region
            for target_lat in range(max(0, target_lat_range[0]), min(patch_h, target_lat_range[1] + 1)):
                for target_lon in range(max(0, target_lon_range[0]), min(patch_w, target_lon_range[1] + 1)):
                    # Attention weights showing what influences this target patch
                    target_attention_h += A_h_avg[target_lat, target_lon, :, :].numpy()
                    target_attention_m += A_m_avg[target_lat, target_lon, :, :].numpy()
            
            # Normalize attention weights
            target_attention_h = target_attention_h / (target_attention_h.sum() + 1e-8)
            target_attention_m = target_attention_m / (target_attention_m.sum() + 1e-8)
            
            attention_analysis.append({
                'timestep': t,
                'self_attention': target_attention_h,
                'cross_attention': target_attention_m,
                'combined_attention': (target_attention_h + target_attention_m) / 2
            })
    
    return attention_analysis

def visualize_attention_influence(attention_analysis, target_region_info, patch_size=4, img_size=64):
    """Visualize which areas most influence the target region"""
    
    # Average attention across all timesteps
    avg_self_attention = np.mean([a['self_attention'] for a in attention_analysis], axis=0)
    avg_cross_attention = np.mean([a['cross_attention'] for a in attention_analysis], axis=0)
    avg_combined_attention = np.mean([a['combined_attention'] for a in attention_analysis], axis=0)
    
    # Upsample attention maps to full image resolution
    attention_maps_full = {}
    for name, attention_map in [
        ('Self-Attention', avg_self_attention),
        ('Cross-Attention', avg_cross_attention),
        ('Combined Attention', avg_combined_attention)
    ]:
        # Upsample from patch grid to full image
        upsampled = cv2.resize(attention_map, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        attention_maps_full[name] = upsampled
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Spatial Attention Analysis: Which Areas Influence Target Region\n'
                 f'Target Region: SW({target_region_info["target_sw_lat"]:.4f}, {target_region_info["target_sw_lon"]:.4f}) '
                 f'NE({target_region_info["target_ne_lat"]:.4f}, {target_region_info["target_ne_lon"]:.4f})', 
                 fontsize=14)
    
    # Plot attention maps
    for i, (name, attention_map) in enumerate(attention_maps_full.items()):
        # Original attention map
        axes[0, i].imshow(np.flipud(attention_map), cmap='hot', aspect='auto')
        axes[0, i].set_title(f'{name}\n(Influence on Target Region)')
        axes[0, i].axis('off')
        
        # Add target region overlay
        target_pixels = target_region_info['target_region_pixels']
        lat_range = target_pixels['lat_range']
        lon_range = target_pixels['lon_range']
        
        # Draw target region rectangle (remember to flip coordinates for display)
        rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                           lon_range[1] - lon_range[0], 
                           lat_range[1] - lat_range[0],
                           linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[0, i].add_patch(rect)
        
        # Thresholded attention map (top 20% most influential areas)
        threshold = np.percentile(attention_map, 80)
        thresholded = attention_map.copy()
        thresholded[thresholded < threshold] = 0
        
        axes[1, i].imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
        axes[1, i].set_title(f'{name}\n(Top 20% Most Influential Areas)')
        axes[1, i].axis('off')
        
        # Add target region overlay
        rect2 = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                            lon_range[1] - lon_range[0], 
                            lat_range[1] - lat_range[0],
                            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[1, i].add_patch(rect2)
        
        # Add colorbar
        cbar = plt.colorbar(axes[0, i].images[0], ax=axes[0, i], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight')
        
        cbar2 = plt.colorbar(axes[1, i].images[0], ax=axes[1, i], fraction=0.046, pad=0.04)
        cbar2.set_label('Attention Weight')
    
    plt.tight_layout()
    plt.savefig('attention_influence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics
    print("\n" + "="*60)
    print("SPATIAL ATTENTION ANALYSIS RESULTS")
    print("="*60)
    print(f"Target Region Coordinates:")
    print(f"  SW: ({target_region_info['target_sw_lat']:.5f}, {target_region_info['target_sw_lon']:.5f})")
    print(f"  NE: ({target_region_info['target_ne_lat']:.5f}, {target_region_info['target_ne_lon']:.5f})")
    
    # Find most influential areas
    combined_attention = attention_maps_full['Combined Attention']
    
    # Get top influential patches
    patch_h, patch_w = avg_combined_attention.shape
    max_attention_coords = []
    
    # Find top 5 most influential patches
    flat_indices = np.argsort(avg_combined_attention.flatten())[-5:]
    
    for idx in reversed(flat_indices):
        patch_lat, patch_lon = np.unravel_index(idx, avg_combined_attention.shape)
        attention_weight = avg_combined_attention[patch_lat, patch_lon]
        
        # Convert patch coordinates back to approximate geographic coordinates
        # This is an approximation for interpretation
        max_attention_coords.append({
            'patch_coords': (patch_lat, patch_lon),
            'attention_weight': attention_weight,
            'pixel_coords': (patch_lat * patch_size + patch_size//2, 
                           patch_lon * patch_size + patch_size//2)
        })
    
    print(f"\nTop 5 Most Influential Areas (in attention grid coordinates):")
    for i, coord in enumerate(max_attention_coords):
        print(f"  {i+1}. Patch ({coord['patch_coords'][0]}, {coord['patch_coords'][1]}) - "
              f"Weight: {coord['attention_weight']:.4f}")
    
    print(f"\nAttention Statistics:")
    print(f"  Mean attention weight: {np.mean(combined_attention):.6f}")
    print(f"  Max attention weight: {np.max(combined_attention):.6f}")
    print(f"  Attention concentration (std/mean): {np.std(combined_attention)/np.mean(combined_attention):.4f}")
    
    return attention_maps_full, max_attention_coords

def main():
    """Main function to extract and analyze spatial attention for the specified region"""
    
    # Target region coordinates provided by user
    target_sw_lat, target_sw_lon = 34.02837, 129.11613
    target_ne_lat, target_ne_lon = 34.76456, 129.55801
    
    print("="*60)
    print("SPATIAL ATTENTION EXTRACTION FOR MICROPLASTIC ANALYSIS")
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
    
    # Set up model configuration (matching the trained model)
    class Args:
        def __init__(self):
            self.batch_size = 8
            self.gpu_num = 1
            self.img_size = 64
            self.num_layers = 1
            self.frame_num = 3
            self.input_dim = 1
            self.hidden_dim = 32
            self.patch_size = 4
    
    args = Args()
    
    # Map target region to attention grid
    print("\nMapping target region to attention grid...")
    target_region_info = map_region_to_attention_grid(
        target_sw_lat, target_sw_lon, target_ne_lat, target_ne_lon,
        full_lats, full_lons, args.img_size // args.patch_size, args.patch_size
    )
    
    # Add original coordinates to region info for reference
    target_region_info['target_sw_lat'] = target_sw_lat
    target_region_info['target_sw_lon'] = target_sw_lon
    target_region_info['target_ne_lat'] = target_ne_lat
    target_region_info['target_ne_lon'] = target_ne_lon
    
    print(f"Target region mapped to attention patches: {target_region_info['target_region_patches']}")
    print(f"Target region mapped to image pixels: {target_region_info['target_region_pixels']}")
    
    # Load trained model weights
    model_path = 'sa_convlstm_japan_microplastics.pth'
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        print("Please run the training script first to generate the model weights.")
        return
    
    # Create model with attention extraction capability
    print("\nLoading trained model...")
    model = SA_ConvLSTM_Model_WithAttention(args)
    
    try:
        # Load the trained weights into our modified model
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Use strict=False in case of minor differences
        model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load some sample data for analysis
    print("\nLoading sample data for attention analysis...")
    from sa_convlstm_microplastics import extract_data_and_clusters, create_sequences_from_data
    
    # Get data files
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    data = extract_data_and_clusters(nc_files[-10:])  # Use last 10 files for analysis
    
    # Create sample sequence
    X_sample, _ = create_sequences_from_data(data, args.frame_num)
    
    # Take a few samples for analysis
    sample_indices = [0, len(X_sample)//2, -1]  # First, middle, last
    
    all_attention_analysis = []
    
    for idx in sample_indices:
        print(f"\nAnalyzing attention for sample {idx+1}...")
        sample_data = torch.FloatTensor(X_sample[idx:idx+1]).to(device)
        
        # Analyze attention for this sample
        attention_analysis = analyze_attention_for_region(
            model, sample_data, target_region_info, args.patch_size
        )
        all_attention_analysis.extend(attention_analysis)
    
    # Visualize results
    print("\nGenerating attention visualization...")
    attention_maps_full, max_attention_coords = visualize_attention_influence(
        all_attention_analysis, target_region_info, args.patch_size, args.img_size
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- attention_influence_analysis.png")
    print("\nThe visualization shows which spatial areas most influence")
    print("microplastic concentration in your specified region.")
    print("Brighter areas indicate higher influence.")

if __name__ == "__main__":
    main() 