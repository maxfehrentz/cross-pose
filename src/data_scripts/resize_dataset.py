import os
import json
import cv2
import argparse

def resize_images_and_update_intrinsics(input_folder, input_transform, output_folder, output_transform):
    # Load transforms.json
    with open(os.path.join(input_folder, input_transform), 'r') as f:
        data = json.load(f)

    # Create new folder if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through frames and resize images
    for frame in data['frames']:
        img_path = os.path.join(input_folder, frame['file_path'])
        mask_path = img_path.replace('img_', 'mask_')
        frame['mask_path'] = mask_path.split('/')[-1]

        print(f"Processing:")
        print(f"  Image: {frame['file_path']}")
        print(f"  Mask: {frame['mask_path']}")
        print(f"  Segmentation: {frame['segmentation_path']}")

        seg_path = os.path.join(input_folder, frame['segmentation_path'])

        # Resize image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        new_h = h // 2
        new_w = w // 2
        resized_img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(os.path.join(output_folder, frame['file_path']), resized_img)

        # Resize masks
        mask = cv2.imread(mask_path)
        resized_mask = cv2.resize(mask, (new_w, new_h))
        cv2.imwrite(os.path.join(output_folder, frame['mask_path']), resized_mask)

        # Resize segmentation masks
        seg_mask = cv2.imread(seg_path)
        resized_seg_mask = cv2.resize(seg_mask, (new_w, new_h))
        cv2.imwrite(os.path.join(output_folder, frame['segmentation_path']), resized_seg_mask)

        # Update intrinsics
        frame['fl_x'] = frame['fl_x'] / 2
        frame['fl_y'] = frame['fl_y'] / 2
        frame['cx'] = frame['cx'] / 2
        frame['cy'] = frame['cy'] / 2

        # Update image dimensions
        frame['w'] = new_w
        frame['h'] = new_h

    # Save updated transforms
    with open(os.path.join(output_folder, output_transform), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize dataset images and update camera parameters.")
    parser.add_argument("input_folder", type=str, help="Path to input folder containing images")
    parser.add_argument("input_transform", type=str, help="Input transforms.json filename")
    parser.add_argument("output_folder", type=str, help="Path to output folder for resized images")
    parser.add_argument("output_transform", type=str, help="Output transforms.json filename")
    
    args = parser.parse_args()

    resize_images_and_update_intrinsics(
        args.input_folder,
        args.input_transform,
        args.output_folder,
        args.output_transform
    )