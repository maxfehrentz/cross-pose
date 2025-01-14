import os
from PIL import Image
import argparse
import numpy as np

def create_masks_from_alpha(folder_path: str, prefix: str):
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and filename.startswith(prefix):
            # Open the image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Ensure the image has an alpha channel
            if image.mode == 'RGBA':
                # Extract the alpha channel
                alpha_channel = image.split()[-1]

                # Convert alpha channel to binary mask (black and white)
                alpha_array = np.array(alpha_channel)
                binary_mask = (alpha_array == 255).astype(np.uint8) * 255

                # Create a new image from the binary mask
                mask_image = Image.fromarray(binary_mask, mode='L')

                # Create the new filename by replacing the prefix with 'mask_'
                new_filename = filename.replace(prefix, 'mask_')
                mask_image_path = os.path.join(folder_path, new_filename)

                # Save the mask image
                mask_image.save(mask_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create masks from alpha channels of PNG images.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing PNG images.")
    parser.add_argument("prefix", type=str, help="Prefix of the PNG files to process.")
    args = parser.parse_args()

    create_masks_from_alpha(args.folder_path, args.prefix)