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
from icomma_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from src.scene.gaussian_model import GaussianModel, MeshAwareGaussianModel, HyperMeshAwareGaussianModel
from src.utils.sh_utils import RGB2SH
from src.utils.mesh_utils import register_mesh
from src.utils.transform_utils import reverse_cam_convention_changes, SWAP_AND_FLIP_WORLD_AXES
import pyvista as pv
import numpy as np
from PIL import Image


def add_mesh_to_plotter(mesh, registration, plotter, wireframe=False, opacity=1):
    register_mesh(mesh, registration)
    # Add the mesh to the plotter
    if wireframe:
        _ = plotter.add_mesh(mesh, smooth_shading=True, lighting=True, diffuse=1, opacity=opacity, style='wireframe')
    else:
        _ = plotter.add_mesh(mesh, smooth_shading=True, lighting=True, diffuse=1, opacity=opacity)

def set_initial_camera(viewpoint_camera, plotter, scale):
    # Get the current c2w matrix
    c2w = viewpoint_camera.c2w
    # Bring back to OpenGL convention
    c2w = reverse_cam_convention_changes(c2w, scale)
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


def render_mesh_deformations(viewpoint_camera, mesh, deformed_mesh, registration, control_vertices, control_def, scale):
    """Render mesh with control points colored by deformation magnitude"""
    plotter = pv.Plotter(off_screen=True, window_size=[viewpoint_camera.image_width, viewpoint_camera.image_height])
    plotter.set_background('black')

    # # Make copy of the mesh to not modify the original mesh
    # mesh_copy = mesh.copy()
    # # Add the mesh to the plotter
    # add_mesh_to_plotter(mesh_copy, registration, plotter, opacity=0.5)

    # Add the deformed mesh to the plotter as wireframe, registration is identity
    add_mesh_to_plotter(deformed_mesh, torch.eye(4), plotter, wireframe=True)
    set_initial_camera(viewpoint_camera, plotter, scale)
    # Get vtk camera
    cam = plotter.camera
    # Get intrinsics and set them in the vtk camera as well, converting radians to degrees
    cam.view_angle = viewpoint_camera.FoVy * 180 / math.pi
    # # Zoom out slightly
    # cam.zoom(.6)
    plotter.camera = cam
    # Compute deformation magnitude for each control point
    deform_magnitudes = np.linalg.norm(control_def, axis=1)
    # Normalize magnitudes to [0,1] for coloring
    if deform_magnitudes.max() > 0:
        deform_magnitudes = deform_magnitudes / deform_magnitudes.max()
    # Reverse the coordinate system conventions, going back from 3DGS world space to original mesh space
    transform = SWAP_AND_FLIP_WORLD_AXES
    control_vertices = (transform[:3, :3] @ control_vertices.T).T / scale
    control_def = (transform[:3, :3] @ control_def.T).T / scale
    # Create PolyData for control points
    points = pv.PolyData(control_vertices.numpy())
    points["deformation"] = deform_magnitudes
    
    # Add arrows at control points, showing deformation direction and magnitude
    points["vectors"] = control_def  # Add vectors as point data
    _ = plotter.add_mesh(
        points.glyph(
            orient=True,  # Use vectors for orientation
            scale='deformation',  # Scale by deformation magnitude
            factor=0.05,  # Scale factor for arrows
            geom=pv.Arrow(  # Arrow glyph settings
                shaft_radius=0.01,
                tip_radius=0.03,
                tip_length=0.05
            )
        ),
        scalars='deformation',
        cmap='RdYlGn_r',  # Red (high deformation) to Green (low deformation)
        clim=[0, 1],
        show_scalar_bar=False,
        scalar_bar_args={'title': 'Deformation Magnitude'}
    )

    # Add spheres at control points
    _ = plotter.add_mesh(
        points.glyph(
            scale=False,  # Don't scale by magnitude
            geom=pv.Sphere(radius=0.0025),  # Fixed size spheres
        ),
        scalars="deformation",
        cmap='RdYlGn_r',  # Red (high deformation) to Green (low deformation)
        clim=[0, 1],
        show_scalar_bar=False,
        opacity=0.8  # Slightly transparent to see arrows better
    )

    plotter.show()
    image = plotter.screenshot(None, return_img=True)
    plotter.close()
    return Image.fromarray(image)


def render_mesh(viewpoint_camera, mesh, registration, scale, wireframe=False):
    plotter = pv.Plotter(off_screen=True, window_size=[viewpoint_camera.image_width, viewpoint_camera.image_height])
    plotter.set_background('black')

    # Make copy of the mesh to not modify the original mesh
    mesh_copy = mesh.copy()
    # Add the mesh to the plotter
    add_mesh_to_plotter(mesh_copy, registration, plotter, wireframe=wireframe)

    set_initial_camera(viewpoint_camera, plotter, scale)
    # Get vtk camera
    cam = plotter.camera
    # Get intrinsics and set them in the vtk camera as well, converting radians to degrees
    cam.view_angle = viewpoint_camera.FoVy * 180 / math.pi
    plotter.camera = cam

    # Show the plotter
    plotter.show()
    mesh_surface = plotter.screenshot(filename=None, transparent_background=True,
                                      return_img=True)
    # - to get the depth in the right direction, pyvista returns negative depth due to right-handed coordinate system
    mesh_depth = -scale * plotter.get_image_depth()
    plotter.close()

    return mesh_surface, mesh_depth


def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scale, mesh = None, registration = None, scaling_modifier = 1.0, override_color = None, deform=True, render_deformation=False, deformed_mesh=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if isinstance(pc, MeshAwareGaussianModel) or isinstance(pc, HyperMeshAwareGaussianModel):
        screenspace_points = torch.zeros(size=(pc._parent_faces.shape[0], 3), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    else:
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
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform, # seems like a 4x4 matrix
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        compute_grad_cov2d=True,
        proj_k=viewpoint_camera.projection_matrix
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    xyz, scales, rots, opacity, shs, semantics = pc(deform)
    means2D = screenspace_points

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rots,
        cov3D_precomp = None,
        camera_center = viewpoint_camera.camera_center,
        camera_pose = viewpoint_camera.world_view_transform)
    rendered_image = rendered_image.permute(1, 2, 0)

    # depth = depth.squeeze(0)
    # spotlight light source model
    # TODO: adding again later, simplifying for now
    # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # If meshes are available, render both original and deformed meshes
    if mesh is not None and registration is not None and deformed_mesh is not None:
        # Render both meshes
        mesh_surface, mesh_depth = render_mesh(viewpoint_camera, mesh, registration, scale)
        # Use identity registration for deformed mesh because it was already registered in the Gaussian Model
        deformed_surface, deformed_depth = render_mesh(viewpoint_camera, deformed_mesh, torch.eye(4), scale)
        # Render initial mesh as wireframe
        init_wireframe, _ = render_mesh(viewpoint_camera, mesh, registration, scale, wireframe=True)
        deformed_wireframe, _ = render_mesh(viewpoint_camera, deformed_mesh, torch.eye(4), scale, wireframe=True)
        
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "mesh_surface": mesh_surface,
                "mesh_depth": mesh_depth,
                "deformed_surface": deformed_surface,
                "deformed_depth": deformed_depth,
                "init_wireframe": init_wireframe,
                "deformed_wireframe": deformed_wireframe}

    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

