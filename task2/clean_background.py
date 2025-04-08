import os
import rembg
from PIL import Image
import numpy as np
from tqdm import tqdm

# Constants
DATASET_DIR = r".\paired_dataset"
OUTPUT_DIR = r".\clean_animal_pokemon_sprite_dataset"
ANIMAL_SUBDIR = "animal"
SPLITS = ["train", "val", "test"]

def ensure_dir(directory):
    """Make sure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_image(image_path, output_path):
    """Process a single image to remove background using rembg."""
    # Load image
    input_img = Image.open(image_path).convert("RGB")
    
    # Remove background
    output = rembg.remove(
        np.array(input_img),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10
    )
    
    # Convert output to RGBA
    output_img = Image.fromarray(output)
    
    # Create a white background
    white_bg = Image.new("RGBA", output_img.size, (255, 255, 255, 255))
    
    # Paste the foreground onto the white background
    white_bg.paste(output_img, (0, 0), output_img)
    
    # Convert back to RGB and save
    white_bg.convert("RGB").save(output_path)

def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        process_image(input_path, output_path)

def main():
    """Main function to process all animal images."""
    print("Using rembg library for background removal...")
    
    # Process each split
    for split in SPLITS:
        input_dir = os.path.join(DATASET_DIR, ANIMAL_SUBDIR, split)
        output_dir = os.path.join(OUTPUT_DIR, ANIMAL_SUBDIR, split)
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        print(f"\nProcessing {split} images...")
        process_directory(input_dir, output_dir)
    
    # Also create the necessary directories for the pokemon images
    for split in SPLITS:
        pokemon_output_dir = os.path.join(OUTPUT_DIR, "pokemon", split)
        ensure_dir(pokemon_output_dir)
        
        # Copy pokemon images (not modifying these)
        pokemon_input_dir = os.path.join(DATASET_DIR, "pokemon", split)
        if os.path.exists(pokemon_input_dir):
            print(f"\nCopying pokemon {split} images...")
            pokemon_files = [f for f in os.listdir(pokemon_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in tqdm(pokemon_files, desc=f"Copying {split} pokemon"):
                src_path = os.path.join(pokemon_input_dir, img_file)
                dst_path = os.path.join(pokemon_output_dir, img_file)
                # Use PIL to open and save to ensure consistency
                img = Image.open(src_path)
                img.save(dst_path)
    
    print("\nBackground removal complete!")
    print(f"Processed images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()