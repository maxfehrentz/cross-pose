import os
import torch
import numpy as np
from imageio import imsave
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from src.utils.renderer import render, render_mesh_deformations
from src.utils.camera import Camera
from src.utils.semantic_utils import SemanticDecoder
import vtk


class FrameVisualizer(object):
    """
    Visualizes itermediate results, render out depth and color images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    Args:

    """

    def __init__(self, outpath, cfg, net):
        self.outmap = os.path.join(outpath, 'mapping')
        # self.outsem = os.path.join(outpath, 'semantic')
        self.outoverlay = os.path.join(outpath, 'overlays')
        self.outdeformation = os.path.join(outpath, 'deformation')
        self.outmesh = os.path.join(outpath, 'mesh')
        self.outct = os.path.join(outpath, 'ct')
        self.outsplat = os.path.join(outpath, 'splats')
        os.makedirs(self.outmap, exist_ok=True)
        # os.makedirs(self.outsem , exist_ok=True)
        os.makedirs(self.outoverlay, exist_ok=True)
        os.makedirs(self.outdeformation, exist_ok=True)
        os.makedirs(self.outmesh, exist_ok=True)
        os.makedirs(self.outct, exist_ok=True)
        os.makedirs(self.outsplat, exist_ok=True)
        self.camera = Camera(cfg['cam'])
        self.widefield_camera = Camera(cfg['cam_widefield'])
        self.decoder = SemanticDecoder()
        self.net = net
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    def save_imgs(self, idx, gt_depth, gt_color, c2w, scale, mesh=None, registration=None, deformed_mesh=None, first_render_debug=False):
        """
        Visualization of depth and color images and save to file.
        Args:

        """
        self.camera.set_c2w(c2w)
        render_pkg = render(self.camera, self.net, self.background, scale, mesh=mesh, registration=registration, deform=False, deformed_mesh=deformed_mesh)

        if first_render_debug:
            splatted_image = render_pkg['render'].cpu().numpy()
            splatted_image = np.clip(splatted_image, 0, 1)
            splat_plot = self.plot_splat(c2w, splatted_image)
            splat_filename = f'{idx:05d}_first_render.jpg'
            outsplat = os.path.join(self.outsplat, splat_filename)
            imsave(outsplat, splat_plot)
            return None, None

        # renderer for pose estimation does not support depth
        # splat_depth = render_pkg['depth']
        splat_depth = torch.ones_like(gt_depth)
        mesh_depth = render_pkg['mesh_depth']
        deformed_mesh_depth = render_pkg['deformed_depth']

        self.plot_mapping(splat_depth, render_pkg['render'], gt_depth, gt_color, 
                         render_pkg['mesh_surface'], mesh_depth,
                         render_pkg['deformed_surface'], deformed_mesh_depth)
        outmap = os.path.join(self.outmap,f'{idx:05d}.jpg')
        plt.savefig(outmap, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close()

        # Create two more figures, alpha blending the rendered image, the deformed mesh and the original mesh, both are already images
        alpha = 0.25
        fig, ax = plt.subplots(1, 1)
        ax.imshow(gt_color.squeeze(0).cpu().numpy())
        ax.imshow(render_pkg['init_wireframe'], alpha=alpha)
        ax.axis('off')
        outmap_overlay_deformed = os.path.join(self.outoverlay,f'{idx:05d}_overlay_init.jpg')
        plt.savefig(outmap_overlay_deformed, bbox_inches='tight', pad_inches=0, dpi=300)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(gt_color.squeeze(0).cpu().numpy())
        ax.imshow(render_pkg['deformed_wireframe'], alpha=alpha)
        ax.axis('off')
        outmap_overlay_undeformed = os.path.join(self.outoverlay, f'{idx:05d}_overlay_warped.jpg')
        plt.savefig(outmap_overlay_undeformed, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        # Create two more figures, deformed and undeformed mesh in surface rendering, not wireframe
        fig, ax = plt.subplots(1, 1)
        ax.imshow(render_pkg['deformed_surface'])
        ax.axis('off')
        outmesh_deformed = os.path.join(self.outmesh, f'{idx:05d}_deformed.jpg')
        plt.savefig(outmesh_deformed, bbox_inches='tight', pad_inches=0, dpi=300)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(render_pkg['mesh_surface'])
        ax.axis('off')
        outmesh_undeformed = os.path.join(self.outmesh, f'{idx:05d}_undeformed.jpg')
        plt.savefig(outmesh_undeformed, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        splatted_image = render_pkg['render'].cpu().numpy()
        splatted_image = np.clip(splatted_image, 0, 1)
        splat_plot = self.plot_splat(c2w, splatted_image)
        outsplat = os.path.join(self.outsplat,f'{idx:05d}.jpg')
        imsave(outsplat, splat_plot)

        return outmap, outsplat
    
    def save_mesh_deformations(self, idx, c2w, mesh, deformed_mesh, registration, control_vertices, control_def, scale):
        self.widefield_camera.set_c2w(c2w)
        deformation_vis = render_mesh_deformations(self.widefield_camera, mesh, deformed_mesh, registration, control_vertices, control_def, scale)
        outdeformation = os.path.join(self.outdeformation, f'{idx:05d}.jpg')
        imsave(outdeformation, deformation_vis)

    def plot_mapping(self, depth, color, gt_depth, gt_color, mesh_surface=None, mesh_depth=None, deformed_surface=None, deformed_depth=None):
        # Already relative depth, came from depth estimation model
        gt_depth_np = gt_depth.squeeze(0).cpu().numpy()

        gt_color_np = gt_color.squeeze(0).cpu().numpy()

        depth_np = depth.squeeze(0).cpu().numpy()
        # Convert to relative depth
        depth_np = depth_np / np.nanmax(depth_np)

        color_np = color.squeeze(0).cpu().numpy()

        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # Convert to relative depth
        depth_residual = depth_residual / np.nanmax(depth_residual)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0

        if mesh_surface is not None and mesh_depth is not None:
            mesh_surface_np = mesh_surface
            mesh_depth_np = mesh_depth / np.nanmax(mesh_depth)
            deformed_surface_np = deformed_surface
            deformed_depth_np = deformed_depth / np.nanmax(deformed_depth)
            fig, axs = plt.subplots(2, 5)
        else:
            fig, axs = plt.subplots(2, 3)

        max_depth = 1.0

        axs[0, 0].imshow(gt_depth_np, vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[0, 1].imshow(depth_np, vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        axs[0, 2].imshow(depth_residual, vmin=0, vmax=max_depth)
        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)

        axs[1, 0].imshow(gt_color_np)
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])

        axs[1, 1].imshow(color_np)
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])

        axs[1, 2].imshow(color_residual)
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        if mesh_surface is not None and mesh_depth is not None:
            axs[0, 3].imshow(mesh_depth_np, vmin=0, vmax=max_depth)
            axs[0, 3].set_title('Original Mesh Depth')
            axs[0, 3].set_xticks([])
            axs[0, 3].set_yticks([])

            axs[1, 3].imshow(mesh_surface_np)
            axs[1, 3].set_title('Original Mesh')
            axs[1, 3].set_xticks([])
            axs[1, 3].set_yticks([])

            axs[0, 4].imshow(deformed_depth_np, vmin=0, vmax=max_depth)
            axs[0, 4].set_title('Deformed Mesh Depth')
            axs[0, 4].set_xticks([])
            axs[0, 4].set_yticks([])

            axs[1, 4].imshow(deformed_surface_np)
            axs[1, 4].set_title('Deformed Mesh')
            axs[1, 4].set_xticks([])
            axs[1, 4].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        return fig, axs


    @torch.no_grad()
    def plot_semantics(self, c2w, scale, thr=0.8):
        # TODO: currently not using widefield but could be useful for debugging
        self.widefield_camera.set_c2w(c2w)
        self.camera.set_c2w(c2w)
        render_pkg = render(self.camera, self.net, self.background, scale)
        # mask gaussians that are in areas with very low opacity
        render_pkg['render'][render_pkg['alpha'] < thr] = 0.0
        semantics = torch.argmax(render_pkg['semantics'], dim=0)
        semantics[render_pkg['alpha'] < thr] = 0
        semantics = self.decoder.colorize_label(semantics.cpu().numpy()) / 255.0
        # TODO: revert this later, abusing this for debugging for nice visualization without semantics
        #vis_img = 0.7*render_pkg['render'].cpu().numpy() + 0.3*semantics
        vis_img = 1.0*render_pkg['render'].cpu().numpy() + 0.0*semantics
        vis_img = (255*vis_img).clip(0, 255).astype(np.uint8)
        return vis_img

    def plot_splat(self, c2w, splatted_image):
        # TODO: currently not using widefield but could be useful for debugging
        self.widefield_camera.set_c2w(c2w)
        self.camera.set_c2w(c2w)
        vis_img = splatted_image
        vis_img = (255*vis_img).clip(0, 255).astype(np.uint8)
        return vis_img

    def visualize_ct_slice(self, ct_image, slice_idx=None, axis=2, filename=None):
        """
        Visualize a slice of the CT volume
        axis: 0=sagittal, 1=coronal, 2=axial
        """
        
        # If no slice specified, use middle slice
        if slice_idx is None:
            dims = ct_image.GetDimensions()
            slice_idx = dims[axis] // 2
            
        # Extract the slice
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(ct_image)
        
        if axis == 0:  # Sagittal
            extractVOI.SetVOI(slice_idx, slice_idx, 0, ct_image.GetDimensions()[1]-1, 0, ct_image.GetDimensions()[2]-1)
        elif axis == 1:  # Coronal
            extractVOI.SetVOI(0, ct_image.GetDimensions()[0]-1, slice_idx, slice_idx, 0, ct_image.GetDimensions()[2]-1)
        else:  # Axial
            extractVOI.SetVOI(0, ct_image.GetDimensions()[0]-1, 0, ct_image.GetDimensions()[1]-1, slice_idx, slice_idx)
        extractVOI.Update()
        
        if axis == 0 or axis == 1:
            # If sagittal or coronal, need to reslice the axes
            axes_reslice = vtk.vtkImageReslice()
            axes_reslice.SetInputData(extractVOI.GetOutput())

            # Create vtk 4x4 matrix that swaps the axes
            matrix = vtk.vtkMatrix4x4()
            # See doc of reslice axes for 4th column: sets origin of the axes, last element must be 1
            if axis == 0:
                # Swap x and z for sagittal
                transform = [
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]
                ]
                for i in range(4):
                    for j in range(4):
                        matrix.SetElement(i, j, transform[i][j])
            else:
                # Swap y and z for coronal
                transform = [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ]
                for i in range(4):
                    for j in range(4):
                        matrix.SetElement(i, j, transform[i][j])
            axes_reslice.SetResliceAxes(matrix)
            axes_reslice.Update()

        # Convert to displayable image
        cast = vtk.vtkImageCast()
        if axis == 0 or axis == 1:
            cast.SetInputConnection(axes_reslice.GetOutputPort())
        else:
            cast.SetInputConnection(extractVOI.GetOutputPort())
        cast.SetOutputScalarTypeToUnsignedChar()
        cast.Update()
        
        # Save to PNG
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(cast.GetOutputPort())
        writer.SetFileName(os.path.join(self.outct, filename))
        writer.Write()

    def visualize_volume_with_points(self, ct_image, points, output_path, color=(1,0,0)):
        """
        Simple volume rendering with points overlay
        """
        
        # Get CT bounds
        ct_bounds = ct_image.GetBounds()
        print(f"CT volume bounds: [{ct_bounds[0]:.1f}, {ct_bounds[1]:.1f}], [{ct_bounds[2]:.1f}, {ct_bounds[3]:.1f}], [{ct_bounds[4]:.1f}, {ct_bounds[5]:.1f}]")
        
        # Get points bounds
        points_array = np.array(points)
        points_min = np.min(points_array, axis=0)
        points_max = np.max(points_array, axis=0)
        print(f"Points bounds: [{points_min[0]:.1f}, {points_max[0]:.1f}], [{points_min[1]:.1f}, {points_max[1]:.1f}], [{points_min[2]:.1f}, {points_max[2]:.1f}]")
        
        # Create renderer and window
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 800)
        
        # Create interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # Set interactor style
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        
        # Set up volume rendering
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(ct_image)
        
        # Set up volume properties (basic CT preset)
        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.1)
        volume_property.SetDiffuse(0.9)
        volume_property.SetSpecular(0.2)
        volume_property.SetSpecularPower(10.0)
        
        # Transfer functions for CT
        color_transfer = vtk.vtkColorTransferFunction()

        color_transfer.AddRGBPoint(-1024, 0.0, 0.0, 0.0)    # Air
        color_transfer.AddRGBPoint(-600, 0.0, 0.0, 0.0)     # Lungs
        color_transfer.AddRGBPoint(-400, 0.15, 0.15, 0.15)  # Lung tissue
        color_transfer.AddRGBPoint(-100, 0.3, 0.3, 0.3)     # Fat
        color_transfer.AddRGBPoint(40, 0.8, 0.8, 0.8)       # Soft tissue
        color_transfer.AddRGBPoint(400, 1.0, 1.0, 1.0)      # Bone
        color_transfer.AddRGBPoint(3000, 1.0, 1.0, 1.0)     # Contrast/Metal
        
        opacity_transfer = vtk.vtkPiecewiseFunction()

        opacity_transfer.AddPoint(-1024, 0.0)
        opacity_transfer.AddPoint(-600, 0.0)                 # Complete transparency for air
        opacity_transfer.AddPoint(-400, 0.0)                 # Start showing very faint lung tissue
        opacity_transfer.AddPoint(-100, 0.05)                # Slight opacity for fat
        opacity_transfer.AddPoint(40, 0.2)                   # More visible soft tissue
        opacity_transfer.AddPoint(400, 0.8)                  # Dense bone
        opacity_transfer.AddPoint(3000, 1.0)                 # Metal/contrast
        
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        # Add points
        points_vtk = vtk.vtkPoints()
        for p in points:
            points_vtk.InsertNextPoint(p)
            
        point_poly = vtk.vtkPolyData()
        point_poly.SetPoints(points_vtk)
        
        # Make points visible as spheres
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(2.0)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(point_poly)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        
        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputConnection(glyph.GetOutputPort())
        
        point_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)
        point_actor.GetProperty().SetColor(color)
        
        # Add actors to renderer
        renderer.AddVolume(volume)
        renderer.AddActor(point_actor)
        renderer.ResetCamera()
        
        # Initialize interactor and start
        interactor.Initialize()
        render_window.Render()
        
        # Define key press callback for saving
        def key_press_callback(obj, event):
            key = obj.GetKeystroke()
            if key == 's':
                # Save current view to PNG
                w2i = vtk.vtkWindowToImageFilter()
                w2i.SetInput(render_window)
                w2i.Update()
                
                writer = vtk.vtkPNGWriter()
                writer.SetFileName(output_path)
                writer.SetInputConnection(w2i.GetOutputPort())
                writer.Write()
                print(f"Saved view to {output_path}")
        
        # Add observer for key press
        interactor.AddObserver("KeyPressEvent", key_press_callback)
        
        # Start interaction
        interactor.Start()


    def warp_ct_volume(self, ct_volume, deformed_mesh, original_mesh, control_ids):
        """
        Warp the CT volume using the deformation field from mesh vertices
        """

        # Get the patient position and translate the mesh points
        patient_position = ct_volume['position']
        # Correct the points in both meshes
        original_mesh.points -= patient_position
        deformed_mesh.points -= patient_position

        # Place the mesh in the center of the CT volume to swap the axes and switch from RAS to LPS
        ct_bounds = ct_volume['image'].GetBounds()
        x_offset = (ct_bounds[1] - ct_bounds[0]) / 2
        y_offset = (ct_bounds[3] - ct_bounds[2]) / 2
        z_offset = (ct_bounds[5] - ct_bounds[4]) / 2
        original_mesh.points[:, 0] -= x_offset
        original_mesh.points[:, 1] -= y_offset
        original_mesh.points[:, 2] -= z_offset
        deformed_mesh.points[:, 0] -= x_offset
        deformed_mesh.points[:, 1] -= y_offset
        deformed_mesh.points[:, 2] -= z_offset
        # Flip the axes
        original_mesh.points[:, 1] = original_mesh.points[:, 1] * -1
        original_mesh.points[:, 2] = original_mesh.points[:, 2] * -1
        deformed_mesh.points[:, 1] = deformed_mesh.points[:, 1] * -1
        deformed_mesh.points[:, 2] = deformed_mesh.points[:, 2] * -1
        # Translate back
        original_mesh.points[:, 0] += x_offset
        original_mesh.points[:, 1] += y_offset
        original_mesh.points[:, 2] += z_offset
        deformed_mesh.points[:, 0] += x_offset
        deformed_mesh.points[:, 1] += y_offset
        deformed_mesh.points[:, 2] += z_offset

        # Optional: can use this visualization to debug, will render vertices together with the CT volume
        # self.visualize_volume_with_points(
        #     self.ct_volume['image'],
        #     original_mesh.points,
        #     f'{self.output}/original_mesh_vertices.png',
        #     color=(0, 0, 1)
        # )
        # self.visualize_volume_with_points(
        #     self.ct_volume['image'],
        #     deformed_mesh.points[control_ids],
        #     f'{self.output}/deformed_mesh_vertices.png',
        #     color=(1, 0, 0)
        # )

        # Get the bounds of our deformation field
        points_array = np.array(original_mesh.points)
        min_bound = np.min(points_array, axis=0)
        max_bound = np.max(points_array, axis=0)
        
        # Create fixed boundary points
        boundary_points = []
        for x in [min_bound[0], max_bound[0]]:
            for y in [min_bound[1], max_bound[1]]:
                for z in [min_bound[2], max_bound[2]]:
                    boundary_points.append([x, y, z])
        boundary_points = np.array(boundary_points)

        # Convert numpy points to vtk points
        source_points = vtk.vtkPoints()
        target_points = vtk.vtkPoints()
        
        # Insert points from numpy arrays
        for control_id in control_ids:
            source_points.InsertNextPoint(original_mesh.points[control_id])
            target_points.InsertNextPoint(deformed_mesh.points[control_id])

        # Add boundary points (same position in both source and target to keep them fixed)
        for p in boundary_points:
            source_points.InsertNextPoint(p)
            target_points.InsertNextPoint(p)

        # Create a transform function from the deformation field
        print("Creating thing plate spline transform")
        transform = vtk.vtkThinPlateSplineTransform()
        transform.SetSigma(.1)
        transform.SetSourceLandmarks(source_points)
        transform.SetTargetLandmarks(target_points)
        transform.SetBasisToR()
        # This inverse is very slow but necessary! mentioned here: https://examples.vtk.org/site/Cxx/PolyData/ThinPlateSplineTransform/
        transform.Inverse()
        
        # Create a filter to actually transform the volume
        print("Creating reslice filter")
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(ct_volume['image'])
        print("Setting reslice transform")
        reslice.SetResliceTransform(transform)
        reslice.SetInterpolationModeToCubic()
        reslice.SetOutputSpacing(ct_volume['spacing'])
        reslice.SetOutputOrigin(ct_volume['origin'])
        
        # Use the original extent
        extent = ct_volume['extent']
        reslice.SetOutputExtent(extent)
        reslice.Update()
        
        output = reslice.GetOutput()
        if not output.GetNumberOfPoints():
            raise ValueError("Warped volume has no points")
            
        return output
