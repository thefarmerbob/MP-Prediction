import imageio.v2 as imageio
from pathlib import Path
import sys

# Get all the cluster plot images from gif-images2 directory
image_dir = Path("gif-images2")
if not image_dir.exists():
    print("ERROR: gif-images2 directory not found!")
    sys.exit(1)

# Get all PNG files and sort them by date
image_files = sorted(image_dir.glob("cluster_plot_*.png"))

if not image_files:
    print("ERROR: No cluster plot images found in gif-images2/")
    sys.exit(1)

print(f"Found {len(image_files)} images to create GIF...")

# Create the GIF
images = []
for i, file in enumerate(image_files):
    print(f"Loading image {i+1}/{len(image_files)}: {file.name}")
    images.append(imageio.imread(file))

# Save the new GIF
output_filename = 'clustering_viridis_timeseries.gif'
print(f"Creating GIF: {output_filename}")
imageio.mimsave(output_filename, images, duration=0.5)

print(f"✅ New GIF created: {output_filename}")
print(f"   - {len(images)} frames")
print(f"   - With viridis colormap and standardized dimensions") 