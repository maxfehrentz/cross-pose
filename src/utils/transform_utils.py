import torch


# Apparently necessary to be in correct metric; guessing mm to cm
SCALE = 10

# Reorient camera in camera space; flip y and z axes (OpenGL/Blender/Nerfstudio to COLMAP/OpenCV convention)
FLIP_CAM_AXES = torch.tensor([[1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]], dtype=torch.float32)

# From same nerfstudio conventions source: "Our world space is oriented such that the up vector is +Z. The XY
#  plane is parallel to the ground plane." In their colmap -> nerfstudio conversion, they swap x and y axes
#  in the world space and then flip z. There is no explanation provided but my hypothesis is that this brings it
#  to COLMAP/OpenCV world space convention which I could not find online.
#  See also code here: https://github.com/nerfstudio-project/nerfstudio/blob/079f419e544b8d2aa6aad6f1a31decf0e06cb88c/nerfstudio/process_data/colmap_utils.py#L620-L621
SWAP_AND_FLIP_WORLD_AXES = torch.tensor([[0, 1, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 0, 0, 1]], dtype=torch.float32)