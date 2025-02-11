import os
from PIL import Image
import argparse
import numpy as np
import json
from pathlib import Path

def find_largest_rectangle(alpha_array):
    """Find the largest rectangle in binary mask using a dynamic programming approach"""
    rows, cols = alpha_array.shape
    heights = np.zeros((rows + 1, cols), dtype=np.int32)
    
    # For each row, calculate height of 1's above
    for i in range(rows):
        for j in range(cols):
            if alpha_array[i,j] == 255:
                heights[i+1,j] = heights[i,j] + 1
    
    max_area = 0
    best_rect = (0, 0, 0, 0)  # x, y, width, height
    
    # For each row, find largest rectangle
    for i in range(1, rows + 1):
        stack = []  # stack of indices
        for j in range(cols + 1):
            start = j
            while stack and (j == cols or heights[i][j] < heights[i][stack[-1]]):
                height = heights[i][stack.pop()]
                width = j - (stack[-1] + 1 if stack else 0)
                area = width * height
                if area > max_area:
                    max_area = area
                    best_rect = (
                        stack[-1] + 1 if stack else 0,  # x
                        i - height,                     # y
                        width,                          # width
                        height                          # height
                    )
            if j < cols:
                stack.append(j)
    
    return best_rect

def process_transforms(transforms_path: str, output_folder: str):
    """Process images based on transforms.json and create new transforms file"""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read transforms.json
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Get all image paths from transforms
    image_paths = [frame["file_path"] for frame in transforms["frames"]]
    
    if not image_paths:
        print("No image paths found in transforms.json")
        return
    
    # Get base directory of transforms.json for relative paths
    base_dir = os.path.dirname(transforms_path)
    
    # Get rectangle from first frame
    first_image_path = os.path.join(base_dir, image_paths[0])
    image = Image.open(first_image_path)
    
    if image.mode != 'RGBA':
        print(f"First frame {first_image_path} doesn't have an alpha channel")
        return
    
    # Extract alpha channel and find largest rectangle
    alpha_channel = np.array(image.split()[-1])
    x, y, width, height = find_largest_rectangle(alpha_channel)
    print(f"Found rectangle at ({x},{y}) with size {width}x{height} from first frame")
    
    # Create new transforms data
    new_transforms = transforms.copy()
    new_transforms["frames"] = []
    
    # Process all frames
    for frame in transforms["frames"]:
        image_path = os.path.join(base_dir, frame["file_path"])
        image = Image.open(image_path)
        
        if image.mode == 'RGBA':
            # Create new filename and path for image
            old_path = Path(frame["file_path"])
            new_filename = f'cropped_{old_path.name}'
            new_rel_path = new_filename  # Just the filename, no subdirectory
            new_abs_path = os.path.join(output_folder, new_filename)
            
            # Crop and save the image
            cropped_image = image.crop((x, y, x + width, y + height))
            cropped_image.save(new_abs_path)
            print(f"Processed {image_path}")

            # Create new frame entry with updated paths
            new_frame = frame.copy()
            
            # Handle segmentation if it exists
            if "segmentation_path" in frame:
                seg_path = os.path.join(base_dir, frame["segmentation_path"])
                if os.path.exists(seg_path):
                    # Load and crop segmentation
                    seg_image = Image.open(seg_path)
                    cropped_seg = seg_image.crop((x, y, x + width, y + height))
                    
                    # Create new segmentation filename and path
                    old_seg_path = Path(frame["segmentation_path"])
                    new_seg_filename = f'cropped_{old_seg_path.name}'
                    new_seg_rel_path = new_seg_filename
                    new_seg_abs_path = os.path.join(output_folder, new_seg_filename)
                    
                    # Save cropped segmentation
                    cropped_seg.save(new_seg_abs_path)
                    print(f"Processed segmentation {seg_path}")
                    
                    # Update segmentation path in frame
                    new_frame["segmentation_path"] = new_seg_rel_path
                else:
                    print(f"Warning: Segmentation file not found: {seg_path}")
            

            new_frame["file_path"] = new_rel_path
            new_frame.pop("mask_path", None)  # Remove if exists
            
            # Update intrinsics
            # Convert numpy types to native Python types
            new_frame['cx'] = float(frame['cx'] - x)
            new_frame['cy'] = float(frame['cy'] - y)
            
            # Update image dimensions to cropped size
            new_frame['w'] = int(width)
            new_frame['h'] = int(height)
            
            new_transforms["frames"].append(new_frame)
    
    # Save new transforms.json
    new_transforms_path = os.path.join(output_folder, 'transforms.json')
    with open(new_transforms_path, 'w') as f:
        json.dump(new_transforms, f, indent=2)
    print(f"Saved new transforms to {new_transforms_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images based on transforms.json and create cropped version.")
    parser.add_argument("transforms_path", type=str, help="Path to the transforms.json file.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for cropped images and new transforms.")
    args = parser.parse_args()

    process_transforms(args.transforms_path, args.output_folder)