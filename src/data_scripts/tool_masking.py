import os
import torch
import numpy as np
from PIL import Image
import json
from torch.autograd import Variable
import argparse

# Surgical SAM is in another project, add it to the path
import sys
sys.path.append('/home/fsc/max/SurgicalSAM')
from segment_anything import sam_model_registry, SamPredictor
from surgicalSAM.model import Prototype_Prompt_Encoder, Learnable_Prototypes
from surgicalSAM.model_forward import model_forward_function
import cv2

def create_masks_from_model(folder_path: str, transforms_path: str, sam_checkpoint: str, surgical_sam_checkpoint: str):
    # Load transforms.json
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Initialize SAM models
    model_type = "vit_h"  # Using full SAM model to get image features
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    sam_predictor = SamPredictor(sam)
    
    # Initialize no-image-encoder SAM components
    model_type_no_encoder = "vit_h_no_image_encoder"
    sam_prompt_encoder, sam_decoder = sam_model_registry[model_type_no_encoder](checkpoint=sam_checkpoint)
    sam_prompt_encoder.cuda()
    sam_decoder.cuda()
    
    # Initialize Surgical SAM specific models
    # num_tokens = 2  # for endovis18
    num_tokens = 4  # for endovis17
    learnable_prototypes_model = Learnable_Prototypes(
        num_classes=7, 
        feat_dim=256  # Match checkpoint dimensions
    ).cuda()
    
    prototype_prompt_encoder = Prototype_Prompt_Encoder(
        feat_dim=256,  # SAM feature dimension
        hidden_dim_dense=128,
        hidden_dim_sparse=128,
        size=64,  # This controls the spatial dimension
        num_tokens=num_tokens
    ).cuda()
    
    # Load weights
    checkpoint = torch.load(surgical_sam_checkpoint)
    prototype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])
    
    # Set to eval mode
    prototype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()
    
    # # Debug prints for tensor shapes
    # with torch.no_grad():
    #     prototypes = learnable_prototypes_model()
    #     print(f"Prototypes shape: {prototypes.shape}")
        
    #     # First image processing
    #     image_path = os.path.join(folder_path, transforms['frames'][0].get('file_path', ''))
    #     if os.path.exists(image_path):
    #         sam_feats, original_size = preprocess_image(image_path, sam_predictor)
    #         print(f"Input image path: {image_path}")
    #         print(f"SAM features shape: {sam_feats.shape}")
    #         print(f"SAM features min/max: {sam_feats.min():.4f}/{sam_feats.max():.4f}")
    #         cls_ids = torch.tensor([7], device='cuda')
    #         print(f"Class IDs shape: {cls_ids.shape}")

    prototypes = learnable_prototypes_model()
    
    # Process each image
    for frame in transforms['frames']:
        image_path = os.path.join(folder_path, frame.get('file_path', ''))
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found")
            continue
        
        # Get SAM features - keep in [B, C, H, W] format
        sam_feats, original_size = preprocess_image(image_path, sam_predictor)
        
        # Set class IDs - shape should be [B] (not [B,1])
        cls_ids = torch.tensor([5], device='cuda')  # Single dimension tensor
        print(f"Class IDs shape: {cls_ids.shape}")
        
        # Run inference
        with torch.no_grad():
            prototypes = learnable_prototypes_model()
            print(f"Prototypes shape: {prototypes.shape}")
            print(f"SAM features shape: {sam_feats.shape}")
            
            preds, preds_quality = model_forward_function(
                prototype_prompt_encoder, 
                sam_prompt_encoder, 
                sam_decoder, 
                sam_feats, 
                prototypes, 
                cls_ids
            )
        
        # Convert predictions to binary mask; where preds > 0, set to 0, else 255
        print(f"shape of preds: {preds.shape}")

        binary_mask = torch.ones_like(preds)
        binary_mask[preds > 0] = 0
        binary_mask = binary_mask.float().cpu().numpy()[0] * 255

        # We have to erode to enlargen the mask since it is actually the 0 values
        kernel = np.ones((30, 30), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
        
        # Create mask image
        mask_image = Image.fromarray(binary_mask.astype(np.uint8), mode='L')

        print(f"shape of mask_image: {mask_image.size}")
        
        # Create new filename
        new_filename = 'color_seg_' + os.path.basename(image_path)
        new_filepath = os.path.join(folder_path, new_filename)
        
        # Save mask
        mask_image.save(new_filepath)

        # Also save the image with the mask applied
        image = Image.open(image_path)
        image.putalpha(mask_image)
        image.save(new_filepath.replace('.png', '_masked.png'))

        frame['segmentation_path'] = new_filename
    
    # Save updated transforms.json
    new_transforms_path = transforms_path.replace('.json', '_color_seg.json')
    with open(new_transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)

def preprocess_image(image_path, sam_predictor):
    """
    Extract SAM features from image using SAM's image encoder
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    image = cv2.resize(image, (1024, 1024))
    
    # Get SAM features
    sam_predictor.set_image(image)
    features = sam_predictor.features.squeeze().permute(1, 2, 0).unsqueeze(0)
    
    print(f"Features final shape: {features.shape}")
    return features, original_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create binary masks using Surgical SAM model.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("transforms_path", type=str, help="Path to transforms.json file.")
    parser.add_argument("sam_checkpoint", type=str, help="Path to SAM checkpoint.")
    parser.add_argument("surgical_sam_checkpoint", type=str, help="Path to Surgical SAM checkpoint.")
    args = parser.parse_args()

    create_masks_from_model(args.folder_path, args.transforms_path, 
                          args.sam_checkpoint, args.surgical_sam_checkpoint)
