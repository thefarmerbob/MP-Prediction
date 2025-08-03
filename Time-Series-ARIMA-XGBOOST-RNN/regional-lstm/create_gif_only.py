import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

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
        img = imageio.imread(img_file)
        
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
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    
    print(f"GIF saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Creating GIF from existing occlusion images...")
    gif_path = create_gif_from_images(
        'yearly_occlusion_frames', 
        'yearly_occlusion_analysis.gif', 
        duration=0.1  # 10 FPS
    )
    
    if gif_path:
        print(f"\n✓ Success! GIF created: {gif_path}")
    else:
        print("\n✗ Failed to create GIF")