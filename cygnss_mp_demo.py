#!/usr/bin/env python3
"""
CYGNSS Microplastic Concentration Demo Visualization

Quick demo version that processes a subset of files for fast visualization.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Set matplotlib backend to prevent display issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')

class CYGNSSDemoVisualizer:
    def __init__(self, data_directory, max_files=20):
        """Initialize with limited number of files for demo."""
        self.data_dir = Path(data_directory)
        all_files = sorted(self.data_dir.glob("cyg.ddmi*.nc"))
        
        # Take every nth file to get a representative sample
        if len(all_files) > max_files:
            step = len(all_files) // max_files
            self.nc_files = all_files[::step][:max_files]
        else:
            self.nc_files = all_files
            
        print(f"Demo: Using {len(self.nc_files)} files out of {len(all_files)} total files")
        
        # Load coordinate data from first file
        if self.nc_files:
            with nc.Dataset(self.nc_files[0], 'r') as ds:
                self.lats = ds.variables['lat'][:]
                self.lons = ds.variables['lon'][:]
                if np.max(self.lons) > 180:
                    self.lons = np.where(self.lons > 180, self.lons - 360, self.lons)
        
    def load_single_file(self, filepath):
        """Load microplastic concentration data from a single NetCDF file."""
        with nc.Dataset(filepath, 'r') as ds:
            mp_data = ds.variables['mp_concentration'][0]
            return np.array(mp_data)
    
    def extract_date_from_filename(self, filepath):
        """Extract date from CYGNSS filename."""
        filename = Path(filepath).name
        date_str = filename.split('-')[0].split('s')[1]
        return datetime.strptime(date_str, '%Y%m%d')
    
    def create_global_map(self, data, title="CYGNSS Microplastic Concentration", 
                         save_path=None, figsize=(15, 8)):
        """Create a global map of microplastic concentration."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mesh for plotting
        lon_mesh, lat_mesh = np.meshgrid(self.lons, self.lats)
        
        # Plot microplastic concentration
        im = ax.pcolormesh(lon_mesh, lat_mesh, data, 
                          cmap='YlOrRd', alpha=0.9, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.1, shrink=0.8)
        cbar.set_label('Microplastic Concentration (count km⁻²)', fontsize=12)
        
        # Set labels and grid
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.3)
        
        # Add degree symbols to tick labels
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xticklabels([f'{x}°' for x in np.arange(-180, 181, 60)])
        ax.set_yticklabels([f'{y}°' for y in np.arange(-90, 91, 30)])
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved map to {save_path}")
        
        plt.close()  # Close figure to free memory
        return fig, ax
    
    def demo_sample_maps(self, num_samples=4):
        """Create sample maps demonstration."""
        print("Creating sample maps demonstration...")
        
        if len(self.nc_files) < num_samples:
            num_samples = len(self.nc_files)
        
        # Select evenly spaced files
        indices = np.linspace(0, len(self.nc_files)-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        vmin, vmax = None, None
        
        # First pass to get global min/max for consistent color scale
        for idx in indices:
            data = self.load_single_file(self.nc_files[idx])
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.min(valid_data)
                    vmax = np.max(valid_data)
                else:
                    vmin = min(vmin, np.min(valid_data))
                    vmax = max(vmax, np.max(valid_data))
        
        for i, idx in enumerate(indices):
            filepath = self.nc_files[idx]
            data = self.load_single_file(filepath)
            date = self.extract_date_from_filename(filepath)
            
            ax = axes[i]
            lon_mesh, lat_mesh = np.meshgrid(self.lons, self.lats)
            im = ax.pcolormesh(lon_mesh, lat_mesh, data, cmap='YlOrRd', 
                              alpha=0.9, vmin=vmin, vmax=vmax, shading='auto')
            
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{date.strftime("%Y-%m-%d")}', fontsize=12)
            
            # Simplified tick labels
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-90, -45, 0, 45, 90])
        
        # Add a common colorbar
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Microplastic Concentration (count km⁻²)', fontsize=12)
        
        plt.suptitle('CYGNSS Microplastic Concentration - Sample Time Periods', 
                     fontsize=16, fontweight='bold')
        
        # Save the figure
        plt.savefig('sample_maps.png', dpi=300, bbox_inches='tight')
        print("Saved sample_maps.png")
        plt.close()
        return fig, axes
    
    def demo_temporal_analysis(self):
        """Create temporal analysis demonstration."""
        print("Creating temporal analysis...")
        
        dates = []
        global_means = []
        global_stds = []
        
        for filepath in self.nc_files:
            data = self.load_single_file(filepath)
            date = self.extract_date_from_filename(filepath)
            
            # Calculate statistics (excluding NaN values)
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                dates.append(date)
                global_means.append(np.mean(valid_data))
                global_stds.append(np.std(valid_data))
        
        # Create temporal plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Global mean over time
        axes[0].plot(dates, global_means, 'bo-', linewidth=2, markersize=6)
        axes[0].set_title('Global Mean Microplastic Concentration Over Time')
        axes[0].set_ylabel('Concentration (count km⁻²)')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Distribution of global means
        axes[1].hist(global_means, bins=15, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True)
        axes[1].set_title('Distribution of Global Mean Concentrations')
        axes[1].set_xlabel('Mean Concentration (count km⁻²)')
        axes[1].set_ylabel('Density')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(global_means)
        std_val = np.std(global_means)
        axes[1].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_val:.1f}')
        axes[1].axvline(mean_val + std_val, color='orange', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'±1σ: {std_val:.1f}')
        axes[1].axvline(mean_val - std_val, color='orange', linestyle='--', 
                       linewidth=1, alpha=0.7)
        axes[1].legend()
        
        plt.tight_layout()
        plt.suptitle('CYGNSS Microplastic Concentration - Temporal Analysis Demo', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved temporal_analysis.png")
        plt.close()
        
        return fig, axes, {'dates': dates, 'means': global_means, 'stds': global_stds}
    
    def demo_spatial_statistics(self):
        """Create spatial statistics demonstration."""
        print("Creating spatial statistics...")
        
        # Calculate average across all demo files
        sum_data = np.zeros((len(self.lats), len(self.lons)))
        count_data = np.zeros((len(self.lats), len(self.lons)))
        
        for filepath in self.nc_files:
            data = self.load_single_file(filepath)
            valid_mask = ~np.isnan(data)
            sum_data[valid_mask] += data[valid_mask]
            count_data[valid_mask] += 1
        
        avg_data = np.divide(sum_data, count_data, 
                           out=np.full_like(sum_data, np.nan), 
                           where=count_data>0)
        
        # Create spatial analysis
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Average concentration map
        lon_mesh, lat_mesh = np.meshgrid(self.lons, self.lats)
        im1 = axes[0].pcolormesh(lon_mesh, lat_mesh, avg_data, 
                                cmap='YlOrRd', alpha=0.9, shading='auto')
        axes[0].set_title('Average Concentration')
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Latitudinal average
        lat_avg = np.nanmean(avg_data, axis=1)
        axes[1].plot(lat_avg, self.lats, 'b-', linewidth=2)
        axes[1].set_title('Latitudinal Average')
        axes[1].set_xlabel('Concentration (count km⁻²)')
        axes[1].set_ylabel('Latitude (°)')
        axes[1].grid(True, alpha=0.3)
        
        # Longitudinal average
        lon_avg = np.nanmean(avg_data, axis=0)
        axes[2].plot(self.lons, lon_avg, 'r-', linewidth=2)
        axes[2].set_title('Longitudinal Average')
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Concentration (count km⁻²)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('CYGNSS Microplastic Concentration - Spatial Analysis Demo', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.savefig('spatial_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved spatial_analysis.png")
        plt.close()
        
        return fig, axes, avg_data


def main():
    """Main demo function."""
    data_directory = "/Users/maradumitru/Downloads/CYGNSS-data"
    
    print("CYGNSS Microplastic Concentration Demo Visualization")
    print("=" * 55)
    
    # Initialize visualizer with limited files
    visualizer = CYGNSSDemoVisualizer(data_directory, max_files=20)
    
    if len(visualizer.nc_files) == 0:
        print(f"No NetCDF files found in {data_directory}")
        return
    
    print(f"\nCreating demo visualizations...")
    
    # 1. Sample maps
    print("\n1. Sample spatial maps...")
    visualizer.demo_sample_maps()
    
    # 2. Temporal analysis
    print("\n2. Temporal analysis...")
    visualizer.demo_temporal_analysis()
    
    # 3. Spatial statistics
    print("\n3. Spatial statistics...")
    visualizer.demo_spatial_statistics()
    
    # 4. Detailed map of latest data
    print("\n4. Detailed map of latest data...")
    latest_data = visualizer.load_single_file(visualizer.nc_files[-1])
    latest_date = visualizer.extract_date_from_filename(visualizer.nc_files[-1])
    visualizer.create_global_map(
        latest_data, 
        title=f"CYGNSS Microplastic Concentration - {latest_date.strftime('%Y-%m-%d')}",
        save_path="latest_detailed_map.png"
    )
    
    print("\nDemo visualization complete!")
    print(f"Processed {len(visualizer.nc_files)} files for demonstration.")
    print("\nGenerated files:")
    print("- sample_maps.png")
    print("- temporal_analysis.png") 
    print("- spatial_analysis.png")
    print("- latest_detailed_map.png")


if __name__ == "__main__":
    main() 