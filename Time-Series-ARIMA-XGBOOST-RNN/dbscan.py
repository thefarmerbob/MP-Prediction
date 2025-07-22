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
import imageio
from scipy.spatial import ConvexHull

nc_files = sorted(Path("../Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))  # Corrected glob pattern
print(f"Number of .nc files found: {len(nc_files)}")
print(f"First few files: {list(nc_files[:3])}")

def process_and_plot(nc_file, threshold=15000, eps=10, min_samples=10):
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    data_array_2d_flipped = np.flipud(data_array_2d)

    # Calculate average microplastic concentration, ignoring NaNs
    average_concentration = np.nanmean(data_array_2d)
    print(f"File: {nc_file}, Average microplastic concentration: {average_concentration:.2f}")

    # Define the region of interest
    lon_start, lon_end = 380, 560
    lat_start, lat_end = 120, 170

    # Extract the region of interest from the data
    region_data = data_array_2d[lat_start:lat_end, lon_start:lon_end]

    # Calculate the average concentration in the region, ignoring NaNs
    average_concentration_region = np.nanmean(region_data)
    print(f"File: {nc_file}, Average microplastic concentration in region: {average_concentration_region:.2f}")

    filtered_indices = np.where(data_array_2d > threshold)
    X = np.column_stack((filtered_indices[0], filtered_indices[1]))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"File: {nc_file}, Estimated clusters: {n_clusters_}, Noise: {n_noise_}")

    unique_labels = set(labels)
    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(data_array_2d_flipped, aspect='auto', cmap='twilight_shifted')
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
    rect = plt.Rectangle((lon_start, data_array_2d_flipped.shape[0] - lat_end), 
                         lon_end - lon_start, lat_end - lat_start, 
                         linewidth=1, edgecolor='r', facecolor='none')
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
        flipped_y = data_array_2d_flipped.shape[0] - xy[:, 0]

        # Draw circles around clusters
        if k != -1:
            if len(xy) >= 3:  # Need at least 3 points to form a hull
                # Check if all x-coordinates or all y-coordinates are the same
                if np.all(xy[:, 0] == xy[0, 0]) or np.all(xy[:, 1] == xy[0, 1]):
                    print(f"Skipping ConvexHull for cluster {k}: Points are collinear")
                    # Optionally, handle this case differently, e.g., draw a line
                else:
                    hull = ConvexHull(xy)
                    # Plot the convex hull
                    for simplex in hull.simplices:
                        ax.plot(xy[simplex, 1], data_array_2d_flipped.shape[0] - xy[simplex, 0], 'r-', lw=2)
            else:
                # If less than 3 points, just draw a circle around the points
                center_x, center_y = np.mean(xy[:, 1]), data_array_2d_flipped.shape[0] - np.mean(xy[:, 0])
                radius = np.max(np.sqrt((xy[:, 1] - center_x)**2 + (data_array_2d_flipped.shape[0] - xy[:, 0] - center_y)**2))
                circle = plt.Circle((center_x, center_y), radius, color='r', fill=False, linewidth=2)
                ax.add_patch(circle)

    return fig

image_files = []
output_dir = Path("gif-images2")  # Define the output directory
output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

for nc_file in nc_files:
    fig = process_and_plot(nc_file)
    image_file = output_dir / f"cluster_plot_{Path(nc_file).stem}.png"  # Save in the gif-images directory
    fig.savefig(image_file)
    plt.close(fig)
    image_files.append(image_file)

#_______________________________

images = []
for file in image_files:
    images.append(imageio.imread(file))
imageio.mimsave('clustering_red_timeseries.gif', images, duration=0.5)

print("GIF created: clustering_red_timeseries.gif")


