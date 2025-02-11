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

def reverse_cam_convention_changes(c2w):
    # Swap and flip axes back from opencv to opengl convention; see datasets.py for explanation
    swap_and_flip_world_axes = SWAP_AND_FLIP_WORLD_AXES.to(c2w.device)
    c2w = swap_and_flip_world_axes @ c2w

    # Flip y and z axis again to go back to nerfstudio convention
    flip_cam_axes = FLIP_CAM_AXES.to(c2w.device)
    c2w = c2w @ flip_cam_axes

    # Divide the translation by scale again
    c2w[:3, 3] = c2w[:3, 3] / SCALE

    return c2w