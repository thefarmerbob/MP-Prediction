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

    # Calculate GLOBAL average microplastic concentration using full dataset, ignoring NaNs
    global_average_concentration = np.nanmean(data_array_2d)
    print(f"File: {nc_file}, Global average microplastic concentration: {global_average_concentration:.2f}")

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
        print(f"Full dataset - Latitude range: {lats.min():.4f} to {lats.max():.4f}")
        print(f"Full dataset - Longitude range: {lons.min():.4f} to {lons.max():.4f}")
        
        # Define display region bounds
        # SW: 22.30649,118.07623
        # NE: 36.96467,143.9533
        display_sw_lat, display_sw_lon = 22.30649, 118.07623
        display_ne_lat, display_ne_lon = 36.96467, 143.9533
        
        # Convert display region coordinates to grid indices
        display_sw_lat_idx, display_sw_lon_idx = lat_lon_to_indices(display_sw_lat, display_sw_lon, lats, lons)
        display_ne_lat_idx, display_ne_lon_idx = lat_lon_to_indices(display_ne_lat, display_ne_lon, lats, lons)
        
        # Ensure proper ordering for display region
        display_lat_start = min(display_sw_lat_idx, display_ne_lat_idx)
        display_lat_end = max(display_sw_lat_idx, display_ne_lat_idx)
        display_lon_start = min(display_sw_lon_idx, display_ne_lon_idx)
        display_lon_end = max(display_sw_lon_idx, display_ne_lon_idx)
        
        print(f"Display region grid indices: lat[{display_lat_start}:{display_lat_end}], lon[{display_lon_start}:{display_lon_end}]")
        
        # Crop the data for display
        data_cropped = data_array_2d[display_lat_start:display_lat_end, display_lon_start:display_lon_end]
        data_cropped_flipped = np.flipud(data_cropped)
        
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
        
        # Convert Tsushima coordinates to grid indices (using full dataset)
        tsushima_sw_lat, tsushima_sw_lon = 34.02837, 129.11613
        tsushima_ne_lat, tsushima_ne_lon = 34.76456, 129.55801
        
        tsushima_sw_lat_idx, tsushima_sw_lon_idx = lat_lon_to_indices(tsushima_sw_lat, tsushima_sw_lon, lats, lons)
        tsushima_ne_lat_idx, tsushima_ne_lon_idx = lat_lon_to_indices(tsushima_ne_lat, tsushima_ne_lon, lats, lons)
        
        # Ensure proper ordering for Tsushima analysis
        tsushima_lat_start = min(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
        tsushima_lat_end = max(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
        tsushima_lon_start = min(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
        tsushima_lon_end = max(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
        
        # Convert Tokyo coordinates to grid indices (using full dataset)
        tokyo_sw_lat, tokyo_sw_lon = 34.80503,139.10416
        tokyo_ne_lat, tokyo_ne_lon = 35.30061,139.81791
        
        tokyo_sw_lat_idx, tokyo_sw_lon_idx = lat_lon_to_indices(tokyo_sw_lat, tokyo_sw_lon, lats, lons)
        tokyo_ne_lat_idx, tokyo_ne_lon_idx = lat_lon_to_indices(tokyo_ne_lat, tokyo_ne_lon, lats, lons)
        
        # Ensure proper ordering for Tokyo analysis
        tokyo_lat_start = min(tokyo_sw_lat_idx, tokyo_ne_lat_idx)
        tokyo_lat_end = max(tokyo_sw_lat_idx, tokyo_ne_lat_idx)
        tokyo_lon_start = min(tokyo_sw_lon_idx, tokyo_ne_lon_idx)
        tokyo_lon_end = max(tokyo_sw_lon_idx, tokyo_ne_lon_idx)
        
        # Convert Kyoto coordinates to grid indices (using full dataset)
        kyoto_sw_lat, kyoto_sw_lon = 36.25588,135.17307
        kyoto_ne_lat, kyoto_ne_lon = 36.64146,135.96607
        
        kyoto_sw_lat_idx, kyoto_sw_lon_idx = lat_lon_to_indices(kyoto_sw_lat, kyoto_sw_lon, lats, lons)
        kyoto_ne_lat_idx, kyoto_ne_lon_idx = lat_lon_to_indices(kyoto_ne_lat, kyoto_ne_lon, lats, lons)
        
        # Ensure proper ordering for Kyoto analysis
        kyoto_lat_start = min(kyoto_sw_lat_idx, kyoto_ne_lat_idx)
        kyoto_lat_end = max(kyoto_sw_lat_idx, kyoto_ne_lat_idx)
        kyoto_lon_start = min(kyoto_sw_lon_idx, kyoto_ne_lon_idx)
        kyoto_lon_end = max(kyoto_sw_lon_idx, kyoto_ne_lon_idx)
        
        # Convert Osaka coordinates to grid indices (using full dataset)
        osaka_sw_lat, osaka_sw_lon = 32.88645, 134.44579
        osaka_ne_lat, osaka_ne_lon = 33.17125, 135.09421
        
        osaka_sw_lat_idx, osaka_sw_lon_idx = lat_lon_to_indices(osaka_sw_lat, osaka_sw_lon, lats, lons)
        osaka_ne_lat_idx, osaka_ne_lon_idx = lat_lon_to_indices(osaka_ne_lat, osaka_ne_lon, lats, lons)
        
        # Ensure proper ordering for Osaka analysis
        osaka_lat_start = min(osaka_sw_lat_idx, osaka_ne_lat_idx)
        osaka_lat_end = max(osaka_sw_lat_idx, osaka_ne_lat_idx)
        osaka_lon_start = min(osaka_sw_lon_idx, osaka_ne_lon_idx)
        osaka_lon_end = max(osaka_sw_lon_idx, osaka_ne_lon_idx)
        
        # Extract the regions of interest from the FULL data for analysis
        tsushima_region_data = data_array_2d[tsushima_lat_start:tsushima_lat_end, tsushima_lon_start:tsushima_lon_end]
        tokyo_region_data = data_array_2d[tokyo_lat_start:tokyo_lat_end, tokyo_lon_start:tokyo_lon_end]
        kyoto_region_data = data_array_2d[kyoto_lat_start:kyoto_lat_end, kyoto_lon_start:kyoto_lon_end]
        osaka_region_data = data_array_2d[osaka_lat_start:osaka_lat_end, osaka_lon_start:osaka_lon_end]
        
        tsushima_average_concentration = np.nanmean(tsushima_region_data)
        tokyo_average_concentration = np.nanmean(tokyo_region_data)
        kyoto_average_concentration = np.nanmean(kyoto_region_data)
        osaka_average_concentration = np.nanmean(osaka_region_data)
        print(f"File: {nc_file}, Tsushima average microplastic concentration: {tsushima_average_concentration:.2f}")
        print(f"File: {nc_file}, Tokyo average microplastic concentration: {tokyo_average_concentration:.2f}")
        print(f"File: {nc_file}, Kyoto average microplastic concentration: {kyoto_average_concentration:.2f}")
        print(f"File: {nc_file}, Osaka average microplastic concentration: {osaka_average_concentration:.2f}")
        
        # Set up display extent for the cropped region
        display_extent = [lons[display_lon_start], lons[display_lon_end], 
                         lats[display_lat_start], lats[display_lat_end]]
        
        # Create figure with proper geographic aspect ratio
        height, width = data_cropped_flipped.shape
        fig_width = 12
        fig_height = fig_width * (height / width)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Display the cropped data
        img = ax.imshow(data_cropped_flipped, aspect='auto', cmap='viridis', extent=display_extent)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        
        # Draw the Tsushima region box (red box) - only if it's within the display region
        if (tsushima_sw_lon >= display_extent[0] and tsushima_ne_lon <= display_extent[1] and
            tsushima_sw_lat >= display_extent[2] and tsushima_ne_lat <= display_extent[3]):
            tsushima_rect = plt.Rectangle((tsushima_sw_lon, tsushima_sw_lat), 
                                        tsushima_ne_lon - tsushima_sw_lon, tsushima_ne_lat - tsushima_sw_lat,
                                        linewidth=2, edgecolor='red', facecolor='none', 
                                        label='Tsushima Region')
            ax.add_patch(tsushima_rect)
        
        # Draw the Tokyo region box (pink box) - only if it's within the display region
        if (tokyo_sw_lon >= display_extent[0] and tokyo_ne_lon <= display_extent[1] and
            tokyo_sw_lat >= display_extent[2] and tokyo_ne_lat <= display_extent[3]):
            tokyo_rect = plt.Rectangle((tokyo_sw_lon, tokyo_sw_lat), 
                                     tokyo_ne_lon - tokyo_sw_lon, tokyo_ne_lat - tokyo_sw_lat,
                                     linewidth=2, edgecolor='pink', facecolor='none', 
                                     label='Tokyo Region')
            ax.add_patch(tokyo_rect)
        
        # Draw the Kyoto region box (magenta box) - only if it's within the display region
        if (kyoto_sw_lon >= display_extent[0] and kyoto_ne_lon <= display_extent[1] and
            kyoto_sw_lat >= display_extent[2] and kyoto_ne_lat <= display_extent[3]):
            kyoto_rect = plt.Rectangle((kyoto_sw_lon, kyoto_sw_lat), 
                                     kyoto_ne_lon - kyoto_sw_lon, kyoto_ne_lat - kyoto_sw_lat,
                                     linewidth=2, edgecolor='magenta', facecolor='none', 
                                     label='Kyoto Region')
            ax.add_patch(kyoto_rect)
        
        # Draw the Osaka region box (dark pink box) - only if it's within the display region
        if (osaka_sw_lon >= display_extent[0] and osaka_ne_lon <= display_extent[1] and
            osaka_sw_lat >= display_extent[2] and osaka_ne_lat <= display_extent[3]):
            osaka_rect = plt.Rectangle((osaka_sw_lon, osaka_sw_lat), 
                                     osaka_ne_lon - osaka_sw_lon, osaka_ne_lat - osaka_sw_lat,
                                     linewidth=2, edgecolor='darkred', facecolor='none', 
                                     label='Osaka Region')
            ax.add_patch(osaka_rect)
        
        fallback_mode = False
    else:
        # Fallback mode without coordinates
        fallback_mode = True
        data_cropped_flipped = data_array_2d_flipped
        tsushima_average_concentration = np.nanmean(data_array_2d[120:170, 380:560])  # fallback region
        tokyo_average_concentration = np.nanmean(data_array_2d[120:170, 380:560])  # fallback region
        kyoto_average_concentration = np.nanmean(data_array_2d[120:170, 380:560])  # fallback region
        osaka_average_concentration = np.nanmean(data_array_2d[120:170, 380:560])  # fallback region
        display_extent = None
    
    if fallback_mode:
        fig, ax = plt.subplots(figsize=(12, 8))
        img = ax.imshow(data_cropped_flipped, aspect='auto', cmap='viridis')
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")
        
        # Draw fallback regional boxes
        tsushima_rect = plt.Rectangle((380, data_cropped_flipped.shape[0] - 170), 180, 50,
                                    linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(tsushima_rect)
        
        tokyo_rect = plt.Rectangle((380, data_cropped_flipped.shape[0] - 170), 180, 50,
                                 linewidth=2, edgecolor='pink', facecolor='none')
        ax.add_patch(tokyo_rect)
        
        kyoto_rect = plt.Rectangle((380, data_cropped_flipped.shape[0] - 170), 180, 50,
                                 linewidth=2, edgecolor='magenta', facecolor='none')
        ax.add_patch(kyoto_rect)
        
        osaka_rect = plt.Rectangle((380, data_cropped_flipped.shape[0] - 170), 180, 50,
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
    ax.set_title(f"Microplastic Concentration Map - {date_title}\n(East Asia Region)", fontsize=14, pad=20)

    # Add text to the plot with concentration statistics
    textstr = f"Global Avg: {global_average_concentration:.2f}\nTsushima: {tsushima_average_concentration:.2f}\nTokyo: {tokyo_average_concentration:.2f}\nKyoto: {kyoto_average_concentration:.2f}\nOsaka: {osaka_average_concentration:.2f}\nDisplay: SW(22.31°,118.08°) NE(36.96°,143.95°)"
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
    fig.savefig('test_cropped_region.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Test plot saved as 'test_cropped_region.png'")
    print("Test completed. Check the output above for coordinate information.")
else:
    print("No files found to process.")

# If test looks good, process all files
create_full_timeseries = input("Create full time series? (y/n): ").lower() == 'y'

if create_full_timeseries:
    image_files = []
    output_dir = Path("timeseries-images")  # Define the output directory
    output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
    
    # Create directory for average data files
    avg_data_dir = Path("regional-averages")
    avg_data_dir.mkdir(exist_ok=True)
    
    # Initialize lists to store averages data
    global_averages = []
    tsushima_averages = []
    tokyo_averages = []
    kyoto_averages = []
    osaka_averages = []
    dates = []

    for nc_file in nc_files:
        # Process and plot
        fig = process_and_plot(nc_file)
        image_file = output_dir / f"timeseries_{Path(nc_file).stem}.png"  # Save in the timeseries-images directory
        fig.savefig(image_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        image_files.append(image_file)
        
        # Extract date from filename and store averages data
        # We need to get the concentration values from the processing
        # Let's get them by reprocessing the file briefly
        ds = xr.open_dataset(nc_file)
        data = ds['mp_concentration']
        data_array = data.values
        data_array_2d = data_array.squeeze()
        
        # Calculate global average
        global_avg = np.nanmean(data_array_2d)
        
        # Calculate regional averages (similar to process_and_plot function)
        lats = None
        lons = None
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
        
        if lats is not None and lons is not None:
            # Tsushima
            tsushima_sw_lat, tsushima_sw_lon = 34.02837, 129.11613
            tsushima_ne_lat, tsushima_ne_lon = 34.76456, 129.55801
            tsushima_sw_lat_idx, tsushima_sw_lon_idx = lat_lon_to_indices(tsushima_sw_lat, tsushima_sw_lon, lats, lons)
            tsushima_ne_lat_idx, tsushima_ne_lon_idx = lat_lon_to_indices(tsushima_ne_lat, tsushima_ne_lon, lats, lons)
            tsushima_lat_start = min(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
            tsushima_lat_end = max(tsushima_sw_lat_idx, tsushima_ne_lat_idx)
            tsushima_lon_start = min(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
            tsushima_lon_end = max(tsushima_sw_lon_idx, tsushima_ne_lon_idx)
            tsushima_region_data = data_array_2d[tsushima_lat_start:tsushima_lat_end, tsushima_lon_start:tsushima_lon_end]
            tsushima_avg = np.nanmean(tsushima_region_data)
            
            # Tokyo
            tokyo_sw_lat, tokyo_sw_lon = 34.80503, 139.10416
            tokyo_ne_lat, tokyo_ne_lon = 35.30061, 139.81791
            tokyo_sw_lat_idx, tokyo_sw_lon_idx = lat_lon_to_indices(tokyo_sw_lat, tokyo_sw_lon, lats, lons)
            tokyo_ne_lat_idx, tokyo_ne_lon_idx = lat_lon_to_indices(tokyo_ne_lat, tokyo_ne_lon, lats, lons)
            tokyo_lat_start = min(tokyo_sw_lat_idx, tokyo_ne_lat_idx)
            tokyo_lat_end = max(tokyo_sw_lat_idx, tokyo_ne_lat_idx)
            tokyo_lon_start = min(tokyo_sw_lon_idx, tokyo_ne_lon_idx)
            tokyo_lon_end = max(tokyo_sw_lon_idx, tokyo_ne_lon_idx)
            tokyo_region_data = data_array_2d[tokyo_lat_start:tokyo_lat_end, tokyo_lon_start:tokyo_lon_end]
            tokyo_avg = np.nanmean(tokyo_region_data)
            
            # Kyoto
            kyoto_sw_lat, kyoto_sw_lon = 36.25588, 135.17307
            kyoto_ne_lat, kyoto_ne_lon = 36.64146, 135.96607
            kyoto_sw_lat_idx, kyoto_sw_lon_idx = lat_lon_to_indices(kyoto_sw_lat, kyoto_sw_lon, lats, lons)
            kyoto_ne_lat_idx, kyoto_ne_lon_idx = lat_lon_to_indices(kyoto_ne_lat, kyoto_ne_lon, lats, lons)
            kyoto_lat_start = min(kyoto_sw_lat_idx, kyoto_ne_lat_idx)
            kyoto_lat_end = max(kyoto_sw_lat_idx, kyoto_ne_lat_idx)
            kyoto_lon_start = min(kyoto_sw_lon_idx, kyoto_ne_lon_idx)
            kyoto_lon_end = max(kyoto_sw_lon_idx, kyoto_ne_lon_idx)
            kyoto_region_data = data_array_2d[kyoto_lat_start:kyoto_lat_end, kyoto_lon_start:kyoto_lon_end]
            kyoto_avg = np.nanmean(kyoto_region_data)
            
            # Osaka
            osaka_sw_lat, osaka_sw_lon = 32.88645, 134.44579
            osaka_ne_lat, osaka_ne_lon = 33.17125, 135.09421
            osaka_sw_lat_idx, osaka_sw_lon_idx = lat_lon_to_indices(osaka_sw_lat, osaka_sw_lon, lats, lons)
            osaka_ne_lat_idx, osaka_ne_lon_idx = lat_lon_to_indices(osaka_ne_lat, osaka_ne_lon, lats, lons)
            osaka_lat_start = min(osaka_sw_lat_idx, osaka_ne_lat_idx)
            osaka_lat_end = max(osaka_sw_lat_idx, osaka_ne_lat_idx)
            osaka_lon_start = min(osaka_sw_lon_idx, osaka_ne_lon_idx)
            osaka_lon_end = max(osaka_sw_lon_idx, osaka_ne_lon_idx)
            osaka_region_data = data_array_2d[osaka_lat_start:osaka_lat_end, osaka_lon_start:osaka_lon_end]
            osaka_avg = np.nanmean(osaka_region_data)
        else:
            # Fallback values
            tsushima_avg = np.nanmean(data_array_2d[120:170, 380:560])
            tokyo_avg = np.nanmean(data_array_2d[120:170, 380:560])
            kyoto_avg = np.nanmean(data_array_2d[120:170, 380:560])
            osaka_avg = np.nanmean(data_array_2d[120:170, 380:560])
        
        # Extract date from filename
        file_name = Path(nc_file).name
        date_str = file_name.split('.')[2][1:]  # Remove 's' prefix
        
        # Store the data
        dates.append(date_str)
        global_averages.append(global_avg)
        tsushima_averages.append(tsushima_avg)
        tokyo_averages.append(tokyo_avg)
        kyoto_averages.append(kyoto_avg)
        osaka_averages.append(osaka_avg)
        
        ds.close()
    
    # Save all averages to separate files
    with open(avg_data_dir / "mp-global-avg.txt", "w") as f:
        for date, avg in zip(dates, global_averages):
            f.write(f"{date}: {avg:.2f}\n")
    
    with open(avg_data_dir / "mp-tsushima-avg.txt", "w") as f:
        for date, avg in zip(dates, tsushima_averages):
            f.write(f"{date}: {avg:.2f}\n")
    
    with open(avg_data_dir / "mp-tokyo-avg.txt", "w") as f:
        for date, avg in zip(dates, tokyo_averages):
            f.write(f"{date}: {avg:.2f}\n")
    
    with open(avg_data_dir / "mp-kyoto-avg.txt", "w") as f:
        for date, avg in zip(dates, kyoto_averages):
            f.write(f"{date}: {avg:.2f}\n")
    
    with open(avg_data_dir / "mp-osaka-avg.txt", "w") as f:
        for date, avg in zip(dates, osaka_averages):
            f.write(f"{date}: {avg:.2f}\n")

    # Create the GIF
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave('microplastic_timeseries_cropped.gif', images, duration=0.8)

    print("GIF created: microplastic_timeseries_cropped.gif")
    print(f"Processed {len(image_files)} files")
    print(f"Regional averages saved to '{avg_data_dir}' directory:")
    print(f"  - mp-global-avg.txt")
    print(f"  - mp-tsushima-avg.txt")
    print(f"  - mp-tokyo-avg.txt")
    print(f"  - mp-kyoto-avg.txt")
    print(f"  - mp-osaka-avg.txt")


