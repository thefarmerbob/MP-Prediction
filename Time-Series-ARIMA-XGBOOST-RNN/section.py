import sys
print(sys.executable)  # This will show which Python installation you're using

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

# Get all .nc files from the whole dataset
all_nc_files = sorted(Path("../../Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
nc_files = all_nc_files

print(f"Number of .nc files found (whole dataset): {len(nc_files)}")
print(f"First few files: {list(nc_files[:3])}")

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def process_and_plot(nc_file):
    ds = xr.open_dataset(nc_file)
    
    # First, let's examine the dataset structure to understand coordinates
    print(f"Dataset variables: {list(ds.variables.keys())}")
    print(f"Dataset dimensions: {list(ds.dims.keys())}")
    
    # Check if there are coordinate variables
    if hasattr(ds, 'coords'):
        print(f"Dataset coordinates: {list(ds.coords.keys())}")
    
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    data_array_2d_flipped = np.flipud(data_array_2d)

    # Calculate average microplastic concentration, ignoring NaNs
    average_concentration = np.nanmean(data_array_2d)
    print(f"File: {nc_file}, Average microplastic concentration: {average_concentration:.2f}")

    # Define the regions of interest with coordinates
    # Tsushima region
    # SW: 34.02837, 129.11613
    # NE: 34.76456, 129.55801
    
    # Tokyo region
    # SW: 34.69138, 138.93427
    # NE: 35.89168, 140.13448
    
    # Kyoto region
    # SW: 36.25588, 135.17307
    # NE: 36.64146, 135.96607
    
    # Osaka region
    # SW: 32.88645, 134.44579
    # NE: 33.17125, 135.09421
    
    # Try to get the actual lat/lon coordinates from the dataset
    lats = None
    lons = None
    
    # Check for different possible coordinate variable names
    possible_lat_names = ['lat', 'latitude', 'y', 'lat_1', 'lat_2']
    possible_lon_names = ['lon', 'longitude', 'x', 'lon_1', 'lon_2']
    
    for lat_name in possible_lat_names:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            print(f"Found latitude variable: {lat_name}")
            break
    
    for lon_name in possible_lon_names:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            print(f"Found longitude variable: {lon_name}")
            break
    
    if lats is not None and lons is not None:
        print(f"Latitude range: {lats.min():.4f} to {lats.max():.4f}")
        print(f"Longitude range: {lons.min():.4f} to {lons.max():.4f}")
        
        # Convert lat/lon to grid indices for Tsushima region
        tsushima_sw_lat, tsushima_sw_lon = 34.02837, 129.11613
        tsushima_ne_lat, tsushima_ne_lon = 34.76456, 129.55801
        
        tsushima_sw_lat_idx, tsushima_sw_lon_idx = lat_lon_to_indices(tsushima_sw_lat, tsushima_sw_lon, lats, lons)
        tsushima_ne_lat_idx, tsushima_ne_lon_idx = lat_lon_to_indices(tsushima_ne_lat, tsushima_ne_lon, lats, lons)
        
        # Ensure proper ordering for Tsushima
        tsushima_lat_start = min(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
        tsushima_lat_end = max(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
        tsushima_lon_start = min(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
        tsushima_lon_end = max(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
        
        # Convert lat/lon to grid indices for Tokyo region
        tokyo_sw_lat, tokyo_sw_lon = 34.69138, 138.93427
        tokyo_ne_lat, tokyo_ne_lon = 35.89168, 140.13448
        
        tokyo_sw_lat_idx, tokyo_sw_lon_idx = lat_lon_to_indices(tokyo_sw_lat, tokyo_sw_lon, lats, lons)
        tokyo_ne_lat_idx, tokyo_ne_lon_idx = lat_lon_to_indices(tokyo_ne_lat, tokyo_ne_lon, lats, lons)
        
        # Ensure proper ordering for Tokyo
        tokyo_lat_start = min(tokyo_sw_lat_idx, tokyo_ne_lat_idx)
        tokyo_lat_end = max(tokyo_sw_lat_idx, tokyo_ne_lat_idx)
        tokyo_lon_start = min(tokyo_sw_lon_idx, tokyo_ne_lon_idx)
        tokyo_lon_end = max(tokyo_sw_lon_idx, tokyo_ne_lon_idx)
        
        # Convert lat/lon to grid indices for Kyoto region
        kyoto_sw_lat, kyoto_sw_lon = 36.25588, 135.17307
        kyoto_ne_lat, kyoto_ne_lon = 36.64146, 135.96607
        kyoto_sw_lat_idx, kyoto_sw_lon_idx = lat_lon_to_indices(kyoto_sw_lat, kyoto_sw_lon, lats, lons)
        kyoto_ne_lat_idx, kyoto_ne_lon_idx = lat_lon_to_indices(kyoto_ne_lat, kyoto_ne_lon, lats, lons)
        kyoto_lat_start = min(kyoto_sw_lat_idx, kyoto_ne_lat_idx)
        kyoto_lat_end = max(kyoto_sw_lat_idx, kyoto_ne_lat_idx)
        kyoto_lon_start = min(kyoto_sw_lon_idx, kyoto_ne_lon_idx)
        kyoto_lon_end = max(kyoto_sw_lon_idx, kyoto_ne_lon_idx)
        
        # Convert lat/lon to grid indices for Osaka region
        osaka_sw_lat, osaka_sw_lon = 32.88645, 134.44579
        osaka_ne_lat, osaka_ne_lon = 33.17125, 135.09421
        osaka_sw_lat_idx, osaka_sw_lon_idx = lat_lon_to_indices(osaka_sw_lat, osaka_sw_lon, lats, lons)
        osaka_ne_lat_idx, osaka_ne_lon_idx = lat_lon_to_indices(osaka_ne_lat, osaka_ne_lon, lats, lons)
        osaka_lat_start = min(osaka_sw_lat_idx, osaka_ne_lat_idx)
        osaka_lat_end = max(osaka_sw_lat_idx, osaka_ne_lat_idx)
        osaka_lon_start = min(osaka_sw_lon_idx, osaka_ne_lon_idx)
        osaka_lon_end = max(osaka_sw_lon_idx, osaka_ne_lon_idx)
        
        print(f"Tsushima grid indices: lat[{tsushima_lat_start}:{tsushima_lat_end}], lon[{tsushima_lon_start}:{tsushima_lon_end}]")
        print(f"Tokyo grid indices: lat[{tokyo_lat_start}:{tokyo_lat_end}], lon[{tokyo_lon_start}:{tokyo_lon_end}]")
        print(f"Kyoto grid indices: lat[{kyoto_lat_start}:{kyoto_lat_end}], lon[{kyoto_lon_start}:{kyoto_lon_end}]")
        print(f"Osaka grid indices: lat[{osaka_lat_start}:{osaka_lat_end}], lon[{osaka_lon_start}:{osaka_lon_end}]")
    else:
        # Fallback to approximate indices
        tsushima_lat_start, tsushima_lat_end = 120, 170
        tsushima_lon_start, tsushima_lon_end = 380, 560
        tokyo_lat_start, tokyo_lat_end = 120, 170
        tokyo_lon_start, tokyo_lon_end = 380, 560
        kyoto_lat_start, kyoto_lat_end = 120, 170
        kyoto_lon_start, kyoto_lon_end = 380, 560
        osaka_lat_start, osaka_lat_end = 120, 170
        osaka_lon_start, osaka_lon_end = 380, 560
        print("Using approximate grid indices (no lat/lon coordinates found)")

    # Extract the regions of interest from the data
    tsushima_region_data = data_array_2d[tsushima_lat_start:tsushima_lat_end, tsushima_lon_start:tsushima_lon_end]
    tokyo_region_data = data_array_2d[tokyo_lat_start:tokyo_lat_end, tokyo_lon_start:tokyo_lon_end]
    kyoto_region_data = data_array_2d[kyoto_lat_start:kyoto_lat_end, kyoto_lon_start:kyoto_lon_end]
    osaka_region_data = data_array_2d[osaka_lat_start:osaka_lat_end, osaka_lon_start:osaka_lon_end]

    # Calculate the average concentration in all regions, ignoring NaNs
    average_concentration_tsushima = np.nanmean(tsushima_region_data)
    average_concentration_tokyo = np.nanmean(tokyo_region_data)
    average_concentration_kyoto = np.nanmean(kyoto_region_data)
    average_concentration_osaka = np.nanmean(osaka_region_data)
    print(f"File: {nc_file}, Average microplastic concentration in Tsushima: {average_concentration_tsushima:.2f}")
    print(f"File: {nc_file}, Average microplastic concentration in Tokyo: {average_concentration_tokyo:.2f}")
    print(f"File: {nc_file}, Average microplastic concentration in Kyoto: {average_concentration_kyoto:.2f}")
    print(f"File: {nc_file}, Average microplastic concentration in Osaka: {average_concentration_osaka:.2f}")

    # Create figure with proper geographic aspect ratio
    # Calculate aspect ratio based on data dimensions and geographic extent
    height, width = data_array_2d_flipped.shape
    
    # Use a more natural aspect ratio for global maps (roughly 2:1 for world maps)
    fig_width = 15
    fig_height = fig_width * (height / width) * 0.7  # Adjust the 0.5 factor for natural proportions
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Use extent if we have actual coordinates, otherwise use indices
    if lats is not None and lons is not None:
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]
        img = ax.imshow(data_array_2d_flipped, aspect='auto', cmap='viridis', extent=extent)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        
        # Draw the Tsushima region box using actual coordinates
        tsushima_rect = plt.Rectangle((tsushima_sw_lon, tsushima_sw_lat), 
                                    tsushima_ne_lon - tsushima_sw_lon, tsushima_ne_lat - tsushima_sw_lat,
                                    linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(tsushima_rect)
        
        # Draw the Tokyo region box using actual coordinates
        tokyo_rect = plt.Rectangle((tokyo_sw_lon, tokyo_sw_lat), 
                                 tokyo_ne_lon - tokyo_sw_lon, tokyo_ne_lat - tokyo_sw_lat,
                                 linewidth=2, edgecolor='pink', facecolor='none')
        ax.add_patch(tokyo_rect)
        
        # Draw the Kyoto region box using actual coordinates
        kyoto_rect = plt.Rectangle((kyoto_sw_lon, kyoto_sw_lat), 
                                 kyoto_ne_lon - kyoto_sw_lon, kyoto_ne_lat - kyoto_sw_lat,
                                 linewidth=2, edgecolor='magenta', facecolor='none')
        ax.add_patch(kyoto_rect)
        
        # Draw the Osaka region box using actual coordinates
        osaka_rect = plt.Rectangle((osaka_sw_lon, osaka_sw_lat), 
                                 osaka_ne_lon - osaka_sw_lon, osaka_ne_lat - osaka_sw_lat,
                                 linewidth=2, edgecolor='darkred', facecolor='none')
        ax.add_patch(osaka_rect)
    else:
        img = ax.imshow(data_array_2d_flipped, aspect='auto', cmap='viridis')
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")
        
        # Draw the Tsushima region box using grid indices
        tsushima_rect = plt.Rectangle((tsushima_lon_start, data_array_2d_flipped.shape[0] - tsushima_lat_end), 
                                    tsushima_lon_end - tsushima_lon_start, tsushima_lat_end - tsushima_lat_start, 
                                    linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(tsushima_rect)
        
        # Draw the Tokyo region box using grid indices
        tokyo_rect = plt.Rectangle((tokyo_lon_start, data_array_2d_flipped.shape[0] - tokyo_lat_end), 
                                 tokyo_lon_end - tokyo_lon_start, tokyo_lat_end - tokyo_lat_start, 
                                 linewidth=2, edgecolor='pink', facecolor='none')
        ax.add_patch(tokyo_rect)
        
        # Draw the Kyoto region box using grid indices
        kyoto_rect = plt.Rectangle((kyoto_lon_start, data_array_2d_flipped.shape[0] - kyoto_lat_end), 
                                 kyoto_lon_end - kyoto_lon_start, kyoto_lat_end - kyoto_lat_start, 
                                 linewidth=2, edgecolor='magenta', facecolor='none')
        ax.add_patch(kyoto_rect)
        
        # Draw the Osaka region box using grid indices
        osaka_rect = plt.Rectangle((osaka_lon_start, data_array_2d_flipped.shape[0] - osaka_lat_end), 
                                 osaka_lon_end - osaka_lon_start, osaka_lat_end - osaka_lat_start, 
                                 linewidth=2, edgecolor='darkred', facecolor='none')
        ax.add_patch(osaka_rect)
    
    cbar = fig.colorbar(img, ax=ax, label='Microplastic Concentration', shrink=0.8)
    cbar.mappable.set_clim(10000, 21000)  # Set the colorbar limits

    # Extract date and season from filename
    file_name = Path(nc_file).name
    date_str = file_name.split('.')[2][1:]
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]

    month_name = {
        '01': 'January', '02': 'February', '03': 'March',
        '04': 'April', '05': 'May', '06': 'June',
        '07': 'July', '08': 'August', '09': 'September',
        '10': 'October', '11': 'November', '12': 'December'
    }[month]

    # Determine the season
    if month in ['12', '01', '02']:
        season = 'Winter'
    elif month in ['03', '04', '05']:
        season = 'Spring'
    elif month in ['06', '07', '08']:
        season = 'Summer'
    else:
        season = 'Autumn'

    date_title = f"{month_name} {day}, {year} - {season}"
    ax.set_title(f"Microplastic Concentration Map - {date_title}", fontsize=14, pad=20)

    # Add text to the plot with concentration statistics
    textstr = f"Global Avg: {average_concentration:.2f}\nTsushima: {average_concentration_tsushima:.2f}\nTokyo: {average_concentration_tokyo:.2f}\nKyoto: {average_concentration_kyoto:.2f}\nOsaka: {average_concentration_osaka:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=props)

    # Add grid for better geographic reference
    ax.grid(True, alpha=0.3)

    return fig

# Let's test with just one file first to understand the structure
if nc_files:
    print("Testing with first file to understand dataset structure...")
    test_file = nc_files[0]
    fig = process_and_plot(test_file)
    
    # Save the test plot
    fig.savefig('test_natural_projection.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Test plot saved as 'test_natural_projection.png'")
    print("Test completed. Check the output above for coordinate information.")
else:
    print("No files found to process.")

# If test looks good, process all files
create_full_timeseries = input("Create full time series? (y/n): ").lower() == 'y'

if create_full_timeseries:
    image_files = []
    output_dir = Path("timeseries-images")  # Define the output directory
    output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    for nc_file in nc_files:
        fig = process_and_plot(nc_file)
        image_file = output_dir / f"timeseries_{Path(nc_file).stem}.png"  # Save in the timeseries-images directory
        fig.savefig(image_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        image_files.append(image_file)

    # Create the GIF
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave('microplastic_timeseries.gif', images, duration=0.8)

    print("GIF created: microplastic_timeseries.gif")
    print(f"Processed {len(image_files)} files")


