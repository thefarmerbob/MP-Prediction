import imageio.v2 as imageio
from pathlib import Path
import sys

# Get all the cluster plot images from gif-images2 directory
image_dir = Path("gif-images2")
if not image_dir.exists():
    print("ERROR: gif-images2 directory not found!")
    sys.exit(1)

# Get all PNG files and sort them by date
all_image_files = sorted(image_dir.glob("cluster_plot_*.png"))

if not all_image_files:
    print("ERROR: No cluster plot images found in gif-images2/")
    sys.exit(1)

print(f"Found {len(all_image_files)} total images")

# Take only the last 1000 images (most recent data)
max_images = 1000
image_files = all_image_files[-max_images:] if len(all_image_files) > max_images else all_image_files

print(f"Using last {len(image_files)} images (most recent data)")

# Create the GIF with the selected images
images = []
for i, file in enumerate(image_files):
    if i % 50 == 0:  # Progress update every 50 images
        print(f"Loading image {i+1}/{len(image_files)}")
    
    try:
        img = imageio.imread(file)
        images.append(img)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue

# Save the new GIF
output_filename = 'clustering_viridis_recent.gif'
print(f"Creating GIF with {len(images)} frames...")

try:
    imageio.mimsave(output_filename, images, duration=0.3)  # Slightly slower for better viewing
    print(f"✅ New GIF created: {output_filename}")
    print(f"   - {len(images)} frames (last {max_images} from {len(all_image_files)} total)")
    print(f"   - With viridis colormap and standardized dimensions")
    print(f"   - Showing most recent time period")
except Exception as e:
    print(f"Error creating GIF: {e}") 