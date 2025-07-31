import sys
print(sys.executable)  # This will show which Python installation you're using


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from sklearn.cluster import DBSCAN
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import imageio.v2 as imageio  # Updated to avoid deprecation warning
from scipy.spatial import ConvexHull

# Try multiple possible paths for CYGNSS data
possible_paths = [
    Path("../../CYGNSS-data"),  # Based on project structure
    Path("../CYGNSS-data"), 
    Path("../../Downloads/CYGNSS-data"),
    Path("../Downloads/CYGNSS-data")
]

nc_files = []
for path in possible_paths:
    if path.exists():
        nc_files = sorted(path.glob("cyg.ddmi*.nc"))
        if nc_files:
            print(f"Found CYGNSS data at: {path}")
            break

if not nc_files:
    print("ERROR: No CYGNSS data files found!")
    print("Checked paths:")
    for path in possible_paths:
        print(f"  - {path.absolute()} (exists: {path.exists()})")
    sys.exit(1)

print(f"Number of .nc files found: {len(nc_files)}")
print(f"First few files: {list(nc_files[:3])}")

# Check dimensions of all files before processing
print("\nChecking file dimensions...")
for i, nc_file in enumerate(nc_files[:5]):  # Check first 5 files
    try:
        ds = xr.open_dataset(nc_file)
        data = ds['mp_concentration']
        shape = data.values.squeeze().shape
        print(f"File {i+1}: {Path(nc_file).name} - Shape: {shape}")
        ds.close()
    except Exception as e:
        print(f"Error reading {nc_file}: {e}")
print()

def process_and_plot(nc_file, threshold=15000, eps=10, min_samples=10):
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    
    # Preprocessing: Ensure consistent dimensions
    target_height, target_width = 320, 1440  # Standard global grid dimensions
    h_offset, w_offset = 0, 0  # Track offsets for coordinate adjustment
    orig_h, orig_w = data_array_2d.shape  # Store original dimensions
    
    if data_array_2d.shape != (target_height, target_width):
        print(f"File: {nc_file}, Original shape: {data_array_2d.shape}, Target shape: ({target_height}, {target_width})")
        
        # Create a new array with target dimensions, filled with NaN
        standardized_array = np.full((target_height, target_width), np.nan)
        
        # Copy the original data to the standardized array (center or align as needed)
        
        # Calculate offsets to center the data if it's smaller
        h_offset = max(0, (target_height - orig_h) // 2)
        w_offset = max(0, (target_width - orig_w) // 2)
        
        # Determine the actual copy dimensions
        copy_h = min(orig_h, target_height)
        copy_w = min(orig_w, target_width)
        
        # Copy data with proper bounds checking
        standardized_array[h_offset:h_offset+copy_h, w_offset:w_offset+copy_w] = \
            data_array_2d[:copy_h, :copy_w]
        
        data_array_2d = standardized_array
    else:
        # If no resizing needed, use original dimensions
        orig_h, orig_w = target_height, target_width
    
    # Use original orientation for consistency
    data_array_2d_display = data_array_2d

    # Calculate average microplastic concentration, ignoring NaNs
    average_concentration = np.nanmean(data_array_2d)
    print(f"File: {nc_file}, Average microplastic concentration: {average_concentration:.2f}")

    # Define the region of interest (adjusted for standardized coordinates)
    lon_start, lon_end = 380 + w_offset, 560 + w_offset
    lat_start, lat_end = 120 + h_offset, 170 + h_offset

    # Extract the region of interest from the data
    region_data = data_array_2d[lat_start:lat_end, lon_start:lon_end]

    # Calculate the average concentration in the region, ignoring NaNs
    average_concentration_region = np.nanmean(region_data)
    print(f"File: {nc_file}, Average microplastic concentration in region: {average_concentration_region:.2f}")

    # Apply DBSCAN to the display data, excluding NaN values
    # Create mask for valid data points above threshold
    valid_data_mask = ~np.isnan(data_array_2d_display)
    threshold_mask = data_array_2d_display > threshold
    combined_mask = valid_data_mask & threshold_mask
    
    filtered_indices = np.where(combined_mask)
    X = np.column_stack((filtered_indices[0], filtered_indices[1]))

    if len(X) == 0:
        print(f"File: {nc_file}, No data points above threshold")
        # Create empty plot
        fig, ax = plt.subplots(figsize=(16, 8))
        img = ax.imshow(data_array_2d_display, aspect='equal', cmap='viridis', origin='lower')
        ax.set_title(f"Map with Clustering (No clusters found)")
        return fig

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"File: {nc_file}, Estimated clusters: {n_clusters_}, Noise: {n_noise_}")

    unique_labels = set(labels)
    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]

    fig, ax = plt.subplots(figsize=(16, 8))  # Changed to 2:1 ratio for global accuracy
    img = ax.imshow(data_array_2d_display, aspect='equal', cmap='viridis', origin='lower')
    
    # Set consistent axis limits to match array coordinates
    ax.set_xlim(0, target_width)
    ax.set_ylim(0, target_height)
    
    cbar = fig.colorbar(img, ax=ax, label='Concentration')
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
    ax.set_title(f"Map with Clustering (Clusters: {n_clusters_}, Date: {date_title})")
    ax.set_xlabel("Longitude Index")
    ax.set_ylabel("Latitude Index")

    # Draw a red box around the region of interest
    # With origin='lower' and original data orientation, coordinates are direct
    rect = plt.Rectangle((lon_start, lat_start), 
                         lon_end - lon_start, lat_end - lat_start, 
                         linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Add text to the plot
    textstr = f"Avg. Conc: {average_concentration:.2f}\nAvg. Conc (Region): {average_concentration_region:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = X[class_member_mask]
        
        # Draw circles around clusters (coordinates are now correctly positioned)
        if k != -1:
            if len(xy) >= 3:  # Need at least 3 points to form a hull
                # Check if all x-coordinates or all y-coordinates are the same
                if np.all(xy[:, 0] == xy[0, 0]) or np.all(xy[:, 1] == xy[0, 1]):
                    print(f"Skipping ConvexHull for cluster {k}: Points are collinear")
                    # Optionally, handle this case differently, e.g., draw a line
                else:
                    hull = ConvexHull(xy)
                    # Plot the convex hull (coordinates are now correctly positioned)
                    for simplex in hull.simplices:
                        ax.plot(xy[simplex, 1], xy[simplex, 0], 'r-', lw=2)
            else:
                # If less than 3 points, just draw a circle around the points
                center_x, center_y = np.mean(xy[:, 1]), np.mean(xy[:, 0])
                radius = np.max(np.sqrt((xy[:, 1] - center_x)**2 + (xy[:, 0] - center_y)**2))
                circle = plt.Circle((center_x, center_y), radius, color='r', fill=False, linewidth=2)
                ax.add_patch(circle)

    return fig

image_files = []
output_dir = Path("gif-images2")  # Define the output directory
output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

for nc_file in nc_files:
    try:
        print(f"Processing: {Path(nc_file).name}")
        fig = process_and_plot(nc_file)
        image_file = output_dir / f"cluster_plot_{Path(nc_file).stem}.png"  # Save in the gif-images directory
        fig.savefig(image_file, dpi=100, bbox_inches='tight')  # Added dpi and bbox_inches for better quality
        plt.close(fig)
        image_files.append(image_file)
        print(f"  -> Saved: {image_file.name}")
    except Exception as e:
        print(f"ERROR processing {nc_file}: {e}")
        continue

#_______________________________

# Check if we have any images before creating GIF
if not image_files:
    print("ERROR: No images were created! Check if NetCDF files are being processed correctly.")
    sys.exit(1)

print(f"Created {len(image_files)} images, now creating GIF...")

images = []
for file in image_files:
    images.append(imageio.imread(file))
    
if images:
    imageio.mimsave('clustering_red_timeseries.gif', images, duration=0.5)
    print("GIF created: clustering_red_timeseries.gif")
else:
    print("ERROR: No images to create GIF with!")


