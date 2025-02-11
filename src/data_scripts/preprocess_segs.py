import os
from PIL import Image
import argparse
import numpy as np
import json

def create_binary_masks_from_segs(folder_path: str, transforms_path: str, target_color: tuple):
    # Load transforms.json
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Track modified filenames to update transforms.json
    filename_mapping = {}
    
    # Iterate through all segmentation files in transforms
    for frame in transforms['frames']:
        seg_file = frame.get('segmentation_path')
        if not seg_file:
            continue
            
        # Open the segmentation image
        image_path = os.path.join(folder_path, seg_file)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found")
            continue
            
        image = Image.open(image_path)
        
        # Convert image to numpy array
        img_array = np.array(image)

        # Only for debugging; printing the last column of the image
        print(img_array[:, -1])
        
        # Mask out the target color
        binary_mask = np.all(img_array != target_color, axis=2).astype(np.uint8) * 255
        
        # Create a new image from the binary mask
        mask_image = Image.fromarray(binary_mask, mode='L')
        
        # Create new filename
        new_filename = 'binary_' + os.path.basename(seg_file)
        new_filepath = os.path.join(folder_path, new_filename)
        
        # Save the mask image
        mask_image.save(new_filepath)
        
        # Track filename mapping
        filename_mapping[seg_file] = new_filename
    
    # Update transforms.json with new filenames
    for frame in transforms['frames']:
        if frame.get('segmentation_path') in filename_mapping:
            frame['segmentation_path'] = filename_mapping[frame['segmentation_path']]
    
    # Save updated transforms.json
    new_transforms_path = transforms_path.replace('.json', '_binary.json')
    with open(new_transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create binary masks from RGB segmentation images.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing segmentation images.")
    parser.add_argument("transforms_path", type=str, help="Path to transforms.json file.")
    parser.add_argument("--color", nargs=3, type=int, default=[255, 0, 0], 
                        help="Target RGB color to convert to white in binary mask (default: red)")
    args = parser.parse_args()

    target_color = tuple(args.color)
    create_binary_masks_from_segs(args.folder_path, args.transforms_path, target_color)
