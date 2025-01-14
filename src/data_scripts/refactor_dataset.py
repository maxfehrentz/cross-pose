import os
import json
import cv2


def resize_images_and_update_intrinsics(folder, file_path, new_folder, new_file_path):
    # Load transforms.json
    with open(os.path.join(folder, file_path), 'r') as f:
        data = json.load(f)

    # Create new folder if necessary
    os.makedirs(new_folder, exist_ok=True)

    # Iterate through frames and resize images
    for frame in data['frames']:
        img_path = os.path.join(folder, frame['file_path'])

        # update the mask path; replace img_ with mask_
        mask_path = img_path
        mask_path = mask_path.replace('img_', 'mask_')
        # Get rid of folder
        frame['mask_path'] = mask_path.split('/')[-1]

        print(f"img path: {frame['file_path']}")
        print(f"mask path: {frame['mask_path']}")

        # Resize image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        new_h = h // 2
        new_w = w // 2
        resized_img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(os.path.join(new_folder, frame['file_path']), resized_img)

        # Resize mask as well
        mask = cv2.imread(mask_path)
        resized_mask = cv2.resize(mask, (new_w, new_h))
        cv2.imwrite(os.path.join(new_folder, frame['mask_path']), resized_mask)

        # Update intrinsics
        frame['fl_x'] = frame['fl_x'] / 2
        frame['fl_y'] = frame['fl_y'] / 2
        frame['cx'] = frame['cx'] / 2
        frame['cy'] = frame['cy'] / 2

        # Update image data
        frame['w'] = new_w
        frame['h'] = new_h

    # Save updated transforms
    with open(os.path.join(new_folder, new_file_path), 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
resize_images_and_update_intrinsics('/home/fsc/max/Data/Pigs/dynamic_breathing', 'transform.json', '/home/fsc/max/Data/Pigs/dynamic_breathing_half_res', 'transforms.json')