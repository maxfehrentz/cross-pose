#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from src.scene.gaussian_model import GaussianModel
from src.utils.sh_utils import RGB2SH
from src.utils.mesh_utils import register_mesh
from src.utils.transform_utils import SWAP_AND_FLIP_WORLD_AXES, FLIP_CAM_AXES, SCALE

# Imports for dealing with the mesh
import pyvista as pv
import vtk

# TODO: Read camera poses, only for debugging
import json
import numpy as np


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


def add_mesh_to_plotter(mesh, registration, plotter):
    register_mesh(mesh, registration)
    # Add the mesh to the plotter
    _ = plotter.add_mesh(mesh, smooth_shading=True, lighting=True, diffuse=1, opacity=1)


# TODO: remove later, only for debugging
def add_cameras_to_plotter(plotter):
    # TODO: hardcoded for now, method just for debugging
    with open('/home/fsc/max/Data/Pigs/posed_data_half_res/transforms_cleaned.json', 'r') as f:
        data = json.load(f)

    frames = data['frames']
    poses = np.array([frame['transform_matrix'] for frame in frames], dtype=float)

    # For the poses during optimization
    pts = [pose[:3, 3] for pose in poses]
    dirs = [-pose[:3, 2] for pose in poses]
    ups = [pose[:3, 1] for pose in poses]

    for i, (p, dir) in enumerate(zip(pts, dirs)):
        a = pv.Cone(p, -dir, radius=.01, height=.01, resolution=4, capping=True)
        _ = plotter.add_mesh(a, color='white', show_edges=True, style='wireframe', line_width=4)


def set_initial_camera(viewpoint_camera, plotter):
    # Get the current c2w matrix
    c2w = viewpoint_camera.c2w
    # Bring back to OpenGL convention
    c2w = reverse_cam_convention_changes(c2w)
    # Direction and up vector follow because each column in the rotation matrix represents where the unit vector of that
    #  axis would end up after the rotation; e.g, first column of R is where the x-axis would end up after the rotation.
    #  In VTK convention, the camera looks towards -Z and Y is up, that's why the second and (negative) third column
    #  represent the view-up and direction after rotation.
    pos = c2w[:3, 3]
    dir = -c2w[:3, 2]
    focal_point = pos + dir
    up = c2w[:3, 1]

    # Get vtk camera
    cam = plotter.camera
    # Set camera extrinsics
    cam.SetPosition(pos[0].item(), pos[1].item(), pos[2].item())
    cam.SetFocalPoint(focal_point[0].item(), focal_point[1].item(), focal_point[2].item())
    cam.SetViewUp(up[0].item(), up[1].item(), up[2].item())
    plotter.camera = cam


# TODO: remove later, only for debugging
def debug_mesh(viewpoint_camera, mesh, registration):
    plotter = pv.Plotter(off_screen=False, window_size=[viewpoint_camera.image_width, viewpoint_camera.image_height])
    plotter.set_background('black')

    add_mesh_to_plotter(mesh, plotter)
    add_cameras_to_plotter(plotter)
    set_initial_camera(viewpoint_camera, plotter)

    # Add axes for debugging
    axes_actor = pv.Axes().axes_actor
    axes_actor.shaft_length = 1
    _ = plotter.add_actor(axes_actor)

    # Show the plotter
    plotter.show(auto_close=False, interactive_update=True)

    # Keep rendering and listening to key events
    while True:
        plotter.update()
    return


def render_mesh(viewpoint_camera, mesh, registration):
    plotter = pv.Plotter(off_screen=True, window_size=[viewpoint_camera.image_width, viewpoint_camera.image_height])
    plotter.set_background('black')

    add_mesh_to_plotter(mesh, registration, plotter)
    set_initial_camera(viewpoint_camera, plotter)

    # Get vtk camera
    cam = plotter.camera

    # Get intrinsics and set them in the vtk camera as well, converting radians to degrees
    cam.view_angle = viewpoint_camera.FoVy * 180 / math.pi

    plotter.camera = cam

    # Show the plotter
    plotter.show()

    mesh_surface = plotter.screenshot(filename="output/ATLAS/screenshot.png", transparent_background=True,
                                      return_img=True)
    # Get depth image, scaling to get what I believe is cm; - to get the depth in the right direction
    mesh_depth = -10 * plotter.get_image_depth()

    return mesh_surface, mesh_depth



def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, mesh = None, registration = None, scaling_modifier = 1.0, override_color = None, deform=True, render_deformation=False, deformed_mesh=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(), # seems like a 4x4 matrix
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    xyz, scales, rots, opacity, shs, semantics = pc(deform)
    if render_deformation:
        # set deformation as color
        mean_def = pc._deformation.get_mean_def(pc.get_xyz).abs()
        mean_def = mean_def / (torch.quantile(mean_def, 0.99)+1e-12)
        shs = RGB2SH(mean_def[:, None])
        opacity = opacity.clamp(0, 0.9)
    means2D = screenspace_points

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, rendered_semantics = rasterizer(
        means3D = xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rots,
        cov3D_precomp = None,
        semantics = semantics)
    rendered_image = rendered_image.permute(1, 2, 0)
    depth = depth.squeeze(0)
    # spotlight light source model
    rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # If a meshes are available, render both original and deformed meshes
    if mesh is not None and registration is not None and deformed_mesh is not None:
        # Render both meshes
        mesh_surface, mesh_depth = render_mesh(viewpoint_camera, mesh, registration)
        # Use identity registration for deformed mesh because it was already registered in the Gaussian Model
        deformed_surface, deformed_depth = render_mesh(viewpoint_camera, deformed_mesh, torch.eye(4))
        
        # TODO: Debug rendered outputs; what is the issue here?
        print("\nRendered outputs stats:")
        print(f"Original mesh depth range: [{mesh_depth.min()}, {mesh_depth.max()}]")
        print(f"Deformed mesh depth range: [{deformed_depth.min()}, {deformed_depth.max()}]")
        
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": depth,
                "alpha": alpha.squeeze(0),
                "semantics": rendered_semantics,
                "mesh_surface": mesh_surface,
                "mesh_depth": mesh_depth,
                "deformed_surface": deformed_surface,
                "deformed_depth": deformed_depth}

    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                "alpha": alpha.squeeze(0),
                "semantics": rendered_semantics,
                "mesh_surface": None,
                "mesh_depth": None}

