# Load a transforms.json and create a copy of a subset of the frames, specified by input args that give starts and ends

import argparse
import json
import os

def extract_frame_number(file_path):
    # Get the filename without extension
    filename = file_path.split('/')[-1].split('.')[0]
    # Extract the last sequence of digits
    digits = ''.join(c for c in filename[::-1] if c.isdigit())[::-1]
    return int(digits) if digits else 0

def split_transforms(transforms_path, start_frame, end_frame, output_name):
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    # Create a new transforms.json with only the frames between start_frame and end_frame
    new_transforms = {
        'frames': [frame for frame in transforms['frames'] 
                  if start_frame <= extract_frame_number(frame['file_path']) <= end_frame]
    }

    # Save the new transforms.json in the same folder
    folder = os.path.dirname(transforms_path)
    with open(os.path.join(folder, output_name), 'w') as f:
        json.dump(new_transforms, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split transforms.json into a subset of frames.")
    parser.add_argument("transforms_path", type=str, help="Path to transforms.json file.")
    parser.add_argument("start_frame", type=int, help="Start frame number.")
    parser.add_argument("end_frame", type=int, help="End frame number.")
    parser.add_argument("output_name", type=str, help="New output filename")

    args = parser.parse_args()

    split_transforms(args.transforms_path, args.start_frame, args.end_frame, args.output_name)