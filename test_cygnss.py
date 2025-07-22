#!/usr/bin/env python3
"""
Simple test script to check CYGNSS data access and create basic visualization.
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def main():
    print("CYGNSS Data Test")
    print("=" * 30)
    
    # Data directory path
    data_directory = "/Users/maradumitru/Downloads/CYGNSS-data"
    
    try:
        # Check if directory exists
        data_dir = Path(data_directory)
        if not data_dir.exists():
            print(f"ERROR: Directory {data_directory} does not exist!")
            return
        
        print(f"✓ Data directory exists: {data_directory}")
        
        # Find NetCDF files
        nc_files = sorted(data_dir.glob("cyg.ddmi*.nc"))
        print(f"✓ Found {len(nc_files)} NetCDF files")
        
        if len(nc_files) == 0:
            print("ERROR: No NetCDF files found!")
            return
        
        # Test opening first file
        first_file = nc_files[0]
        print(f"✓ Testing file: {first_file.name}")
        
        with nc.Dataset(first_file, 'r') as ds:
            print("✓ Successfully opened NetCDF file")
            
            # Check variables
            variables = list(ds.variables.keys())
            print(f"✓ Variables found: {variables}")
            
            # Check dimensions
            dims = {dim: ds.dimensions[dim].size for dim in ds.dimensions}
            print(f"✓ Dimensions: {dims}")
            
            # Load coordinate data
            lats = ds.variables['lat'][:]
            lons = ds.variables['lon'][:]
            mp_data = ds.variables['mp_concentration'][0]
            
            print(f"✓ Latitude range: {np.min(lats):.1f}° to {np.max(lats):.1f}°")
            print(f"✓ Longitude range: {np.min(lons):.1f}° to {np.max(lons):.1f}°")
            print(f"✓ Data shape: {mp_data.shape}")
            
            # Check data values
            valid_data = mp_data[~np.isnan(mp_data)]
            if len(valid_data) > 0:
                print(f"✓ Valid data points: {len(valid_data)}")
                print(f"✓ Data range: {np.min(valid_data):.1f} to {np.max(valid_data):.1f}")
            else:
                print("⚠ WARNING: No valid data found!")
                return
        
        # Create a simple test plot
        print("\n📊 Creating test visualization...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert longitude if needed
        if np.max(lons) > 180:
            lons = np.where(lons > 180, lons - 360, lons)
        
        # Create mesh and plot
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        im = ax.pcolormesh(lon_mesh, lat_mesh, mp_data, 
                          cmap='YlOrRd', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Microplastic Concentration (count km⁻²)')
        
        # Set labels
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'CYGNSS Microplastic Concentration Test\nFile: {first_file.name}')
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        output_file = 'cygnss_test_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Test plot saved: {output_file}")
        
        # Quick statistics
        print(f"\n📈 Quick Statistics:")
        print(f"   Mean concentration: {np.mean(valid_data):.1f} count km⁻²")
        print(f"   Std concentration: {np.std(valid_data):.1f} count km⁻²")
        print(f"   Min concentration: {np.min(valid_data):.1f} count km⁻²")
        print(f"   Max concentration: {np.max(valid_data):.1f} count km⁻²")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 