import numpy as np
import os
import torch
import torchvision
import wandb
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from scipy.spatial import KDTree
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
from torch.utils.data import DataLoader
from src.utils.CT_reader import read_ct_volume
from src.utils.camera import Camera
from src.utils.FrameVisualizer import FrameVisualizer
from src.utils.flow_utils import get_depth_from_raft
from src.utils.datasets import MonoMeshMIS, NeuroWrapper
from src.utils.loss_utils import l1_loss, VGGPerceptualLoss, GaussianAppearanceRegularizer
from src.utils.renderer import render
from src.scene.gaussian_model import MeshAwareGaussianModel, HyperMeshAwareGaussianModel
from src.utils.mesh_utils import preprocess_mesh, undo_preprocess_mesh, undo_register_mesh
from src.utils.transform_utils import SWAP_AND_FLIP_WORLD_AXES

# For Depth Anything V2
import cv2
import sys
sys.path.append('Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# For the preoperative CT
import pyvista as pv

# For benchmarking
from torch.profiler import profile, record_function, ProfilerActivity

# Initialize tensorboard writer
from torch.utils.tensorboard import SummaryWriter

# For VTK
import vtk

# For pose estimation
from src.scene.camera_pose import CameraPose

class SceneOptimizer():
    def __init__(self, cfg, args):
        self.total_iters = 0
        self.cfg = cfg
        self.args = args
        self.visualize = args.visualize
        self.eval = cfg['eval']
        self.scale = cfg['scale']
        self.device = cfg['device']
        self.output = cfg['data']['output']

        if self.cfg['dataset'] == 'ATLAS' or self.cfg['dataset'] == 'SOFA':
            self.frame_reader = MonoMeshMIS(cfg, args, scale=self.scale)
        elif self.cfg['dataset'] == 'NEURO':
            self.frame_reader = NeuroWrapper(cfg, args, scale=self.scale)
        else:
            raise ValueError(f"Dataset {self.cfg['dataset']} not supported")
        
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=0 if args.debug else 4)
        # self.net = GaussianModel(cfg=cfg['model'])
        # self.net = MeshAwareGaussianModel(cfg=cfg['model'])
        self.net = HyperMeshAwareGaussianModel(cfg=cfg['model'], hypernet_cfg=None)
        self.camera = Camera(cfg['cam'])
        if self.cfg['model']['ARAP']['active']:
            self.output = f"{self.output}_ARAP"
        self.visualizer = FrameVisualizer(self.output, cfg, self.net)

        self.log_freq = args.log_freq
        self.log = args.log is not None
        self.run_id = wandb.util.generate_id()
        log_cfg = cfg.copy()
        log_cfg.update(vars(args))
        if self.log:
            wandb.init(id=self.run_id, name=args.log, config=log_cfg, project='gtracker', group=args.log_group)
        self.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        self.dbg = args.debug

        # Checking for baseline, not available if monocular dataset
        self.baseline = cfg['cam'].get('stereo_baseline', None)
        if self.baseline is not None:
            self.baseline = self.baseline / 1000.0 * self.scale

        # Checking for mesh
        self.mesh = cfg.get('data', {}).get('mesh', None)
        # Load mesh
        if self.mesh is not None:
            self.mesh = pv.read(self.mesh)
            # Preprocesses mesh in place; dataset is relevant as different datasets have different mesh coordinate system conventions
            preprocess_mesh(self.mesh, dataset=self.cfg['dataset'])

        # Initialize Depth Anything V2 model
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        encoder = 'vitl'
        self.depth_anything = DepthAnythingV2(**model_configs[encoder])
        self.depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()

        # Initialize tensorboard writer
        self.writer = SummaryWriter(os.path.join(self.output, 'tensorboard'))

        # Load CT if provided
        self.ct_volume = None
        # Safely check if key exists in cfg['data']
        CT_folder = cfg['data'].get('CT', None)
        if self.visualize and CT_folder is not None:
            # Contains volumetric and metadata, check method for details
            self.ct_volume = read_ct_volume(CT_folder)

    
    def estimate_pose(self, target, max_iter=500, camera_idx=0, lr_pose=0.1, lr_model=1):
        '''
        Estimate the pose of the target image, initialzed from camera_idx
        '''
        # Load a target image, target is the path to that image
        target_img_original = cv2.imread(target)

        target_img = cv2.cvtColor(target_img_original, cv2.COLOR_BGR2RGB)
        target_img = torch.from_numpy(target_img).cuda() / 255.0  # Normalize to [0,1]

        # Use the frameloader to get camera pose
        # TODO: dangerous for this not to be a dict, fix later
        self.camera.set_c2w(self.frame_reader[camera_idx][3].to(self.device))

        # Invert the current camera pose
        current_w2c = torch.inverse(self.camera.c2w)
        init_pose = CameraPose(current_w2c.cuda(), self.camera.FoVx, self.camera.FoVy, self.camera.image_width, self.camera.image_height)
        init_pose = init_pose.cuda()

        # Add pose and self.net parameters to the optimizer
        optimizer = torch.optim.Adam([
            {'params': init_pose.parameters(), 'lr': lr_pose},
            {'params': self.net.parameters(), 'lr': lr_model}
        ])

        # Disable all gradients for the model
        self.net.train()
        for param in self.net.parameters():
            param.requires_grad = False

        for i in tqdm(range(max_iter), desc="Pose estimation"):
            optimizer.zero_grad()

            # TODO: this should kick in at the very end of optimization
            # if i == max_iter - 10:
            if i == 0:
                # Activate gradients for the appearance related ones
                for name, param in self.net.named_parameters():
                    if '_features_dc' in name or '_features_rest' in name:
                        print(f"Enabling gradient for {name}")
                        param.requires_grad = True

            # Render image
            render_pkg = render(init_pose, self.net, self.background, self.scale, deform=False)
            rendered_img = render_pkg['render']

            if i == 0:
                # Check if the target image is in the same format as the rendered image, if not resize
                # TODO: this step would mess up intrinsics if we were to estimate them
                if target_img.shape != rendered_img.shape:
                    print(f"Target image shape: {target_img.shape}")
                    print(f"Rendered image shape: {rendered_img.shape}")
                    print("Resizing target image to match rendered image shape")
                    target_img_original = cv2.resize(target_img_original, (rendered_img.shape[0], rendered_img.shape[1]))
                    target_img = cv2.cvtColor(target_img_original, cv2.COLOR_BGR2RGB)
                    target_img = torch.from_numpy(target_img).cuda() / 255.0  # Normalize to [0,1]
                    print(f"shape of target image after resize: {target_img.shape}")

            # TODO: compute the style loss here
            perceptual_loss = VGGPerceptualLoss()
            # For perceptual loss, have to swap the color channels to be in the right position
            style_loss = perceptual_loss(rendered_img.permute(2, 0, 1).unsqueeze(0), target_img.permute(2, 0, 1).unsqueeze(0), feature_layers=[0, 1, 2, 3], style_layers=[0, 1, 2, 3])

            style_loss.backward(retain_graph=True)
             # Make sure those grads do not affect the camera pose
            for param in init_pose.parameters():
                param.grad = None
            # Store network gradients
            model_grads = {}
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    model_grads[name] = param.grad.clone()
                    param.grad = None

            # Save original images before blurring
            original_rendered_img = rendered_img.clone()
            original_target_img = target_img.clone()

            # Reshape from (H, W, C) to (C, H, W) for the blurring
            rendered_img = rendered_img.permute(2, 0, 1)
            target_img = target_img.permute(2, 0, 1)
            rendered_img = torchvision.transforms.GaussianBlur(kernel_size=15, sigma=10)(rendered_img)
            target_img = torchvision.transforms.GaussianBlur(kernel_size=15, sigma=10)(target_img)
            # Reshape back to (H, W, C)
            rendered_img = rendered_img.permute(1, 2, 0)
            target_img = target_img.permute(1, 2, 0)

            # Compute l1 loss
            loss = l1_loss(rendered_img, target_img)
            print(f"Pixel loss: {loss.item()}")
            
            loss.backward()

            # Overwrite the model grads
            for name, param in self.net.named_parameters():
                if name in model_grads:
                    param.grad = model_grads[name]

            optimizer.step()

            # Update pose parameters
            init_pose(current_w2c)

            # TODO: all the visualization below should get its own visualization method

            # Save the rendered image, blend the target image in the background
            rendered_img_vis = np.clip(original_rendered_img.detach().cpu().numpy(), 0, 1)
            rendered_img_vis = (255*rendered_img_vis).clip(0, 255).astype(np.uint8)
            rendered_img_vis = cv2.cvtColor(rendered_img_vis, cv2.COLOR_RGB2BGR)
            # Blend the target image in the background
            blended_img = (0.5*rendered_img_vis + 0.25*target_img_original).astype(np.uint8)
            cv2.imwrite(f"{self.output}/pose_estimation_blended_{i}.png", blended_img)

            # Save the blurred rendered image as well
            rendered_img_blur = np.clip(rendered_img.detach().cpu().numpy(), 0, 1)
            rendered_img_blur = (255*rendered_img_blur).clip(0, 255).astype(np.uint8)
            rendered_img_blur = cv2.cvtColor(rendered_img_blur, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self.output}/pose_estimation_blurred_{i}.png", rendered_img_blur)

            # Save the original rendered image as well
            cv2.imwrite(f"{self.output}/pose_estimation_original_{i}.png", rendered_img_vis)


    def fit(self, frame, iters, incremental):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]) as prof:
        self.net.reset_optimizer()
        idx, gt_color, gt_depth, gt_c2w, tool_mask = frame

        self.camera.set_c2w(gt_c2w)
        for iter in range(1, iters+1):
            if self.cfg['training']['spherical_harmonics'] and iter > iters/2:
                self.net.enable_spherical_harmonics()
            self.total_iters += 1
            self.net.train(iter == 1)

            with record_function("render"):
                render_pkg = render(self.camera, self.net, self.background, self.scale, deform=incremental)
            self.net.eval()
            color = render_pkg['render'][None, ...]
            # Currently used renderer does not support depth
            # depth = render_pkg['depth'][None, ...]
            depth = torch.ones_like(color)

            # Convert rendered depth map to a relative depth map
            depth = depth / depth.max()

            # Loss
            with record_function("photometric loss computation"):
                Ll1 = self.cfg['training']['w_color']*l1_loss(color[tool_mask], gt_color[tool_mask])

            loss = Ll1

            if isinstance(self.net, MeshAwareGaussianModel) or isinstance(self.net, HyperMeshAwareGaussianModel):
                with record_function("compute regularization terms"):
                    reg_arap, reg_scale, reg_rigid_loc, reg_rigid_rot, reg_iso, reg_visibility = self.net.compute_regularization(render_pkg["visibility_filter"])
                    l_arap = reg_arap * self.cfg['training']['w_def']['w_arap']
                    l_scale = reg_scale * self.cfg['training']['w_def']['w_scale']
                    l_rigid_loc = reg_rigid_loc * self.cfg['training']['w_def']['rigid']
                    l_rigid_rot = reg_rigid_rot * self.cfg['training']['w_def']['rot']
                    l_iso = reg_iso * self.cfg['training']['w_def']['iso']
                    l_visibility = reg_visibility * self.cfg['training']['w_def']['nvisible']
                    loss += l_arap + l_scale + l_rigid_loc + l_rigid_rot + l_iso + l_visibility
            else:
                l_rigidtrans, l_rigidrot, l_iso, l_visible = self.net.compute_regulation(render_pkg["visibility_filter"])
                def_loss = self.cfg['training']['w_def']['rigid']*l_rigidtrans + self.cfg['training']['w_def']['iso']*l_iso+ self.cfg['training']['w_def']['rot']*l_rigidrot+ self.cfg['training']['w_def']['nvisible']*l_visible
                loss += def_loss
            
            loss.backward()

            viewspace_point_tensor_grad = torch.zeros_like(render_pkg["viewspace_points"])
            viewspace_point_tensor_grad += render_pkg["viewspace_points"].grad

            with torch.no_grad():
                #Optimizer step
                if iter < iters:
                    self.net.optimizer.step()
                    self.net.optimizer.zero_grad(set_to_none=True)

                # Log scale of the Gaussians
                scales = self.net.get_scaling
                median_scale = torch.median(scales)
                mean_scale = torch.mean(scales)
                max_scale = torch.max(scales)
                min_scale = torch.min(scales)
                self.writer.add_scalar('Scale/Median', median_scale.item(), self.total_iters)
                self.writer.add_scalar('Scale/Mean', mean_scale.item(), self.total_iters)
                self.writer.add_scalar('Scale/Max', max_scale.item(), self.total_iters)
                self.writer.add_scalar('Scale/Min', min_scale.item(), self.total_iters)

                # Log all loss components, with weights
                self.writer.add_scalar('Loss/Photometric', Ll1.item(), self.total_iters)
                # self.writer.add_scalar('Loss/Depth', Ll1_depth.item(), self.total_iters)
                self.writer.add_scalar('Loss/ARAP', l_arap.item(), self.total_iters)
                self.writer.add_scalar('Loss/RigidLoc', l_rigid_loc.item(), self.total_iters)
                self.writer.add_scalar('Loss/RigidRot', l_rigid_rot.item(), self.total_iters)
                self.writer.add_scalar('Loss/ISO', l_iso.item(), self.total_iters)
                self.writer.add_scalar('Loss/Scale', l_scale.item(), self.total_iters)
                self.writer.add_scalar('Loss/Visibility', l_visibility.item(), self.total_iters)
                self.writer.add_scalar('Loss/Total', loss.item(), self.total_iters)

        # # Write profiling results to console and export trace to file
        # tqdm.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=400))
        # prof.export_chrome_trace(f'{self.output}/trace.json')


    def get_deformed_mesh(self, mesh, pc):
        # Get the deformed mesh vertices from the Gaussian model
        deformed_mesh = mesh.copy()
        deformed_vertices = pc.get_mesh_vertices.detach().cpu().numpy()

        # Transform to original mesh space! Bear in mind that those vertices do not come from the original mesh
        #  but from the optimization in the world space of the Gaussians. Need to apply the same transform
        #  (or rather its inverse) to go back to the original mesh space that this rendering assumes
        transform = SWAP_AND_FLIP_WORLD_AXES
        deformed_vertices = (transform[:3, :3] @ deformed_vertices.T).T / self.scale
        # Ensure array is contiguous and in the correct dtype
        deformed_vertices = np.ascontiguousarray(deformed_vertices, dtype=np.float32)
        deformed_mesh.points = deformed_vertices

        # Have to recompute normals after changing the vertex positions
        deformed_mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,  # Looks better with True, but causes issues with number of vertices
            inplace=True
        )

        return deformed_mesh
    

    def run(self):
        torch.cuda.empty_cache()

        # Collect metrics for evaluation, e.g., when ground truth meshes are available
        if self.eval:   
            mesh_error_l2 = []
            mesh_error_std = []
            global_max_error = 0

        for ids, gt_color, gt_color_r, gt_c2w, registration, tool_mask, semantics, intrinsics, gt_mesh_path in tqdm(self.frame_loader, total=self.n_img):
            # TODO: remove later, hack for now
            if ids.item() == 10:
                # # Save the net
                # print("Saving the model...")
                # torch.save(self.net.state_dict(), f'{self.output}/net.pth')
                return
            
            gt_color = gt_color.cuda()
            gt_color_r = gt_color_r.cuda() if gt_color_r is not None else None
            gt_c2w = gt_c2w.cuda()
            tool_mask = tool_mask.cuda() if tool_mask is not None else None
            semantics = semantics.float().cuda() if semantics is not None else None
            with torch.no_grad():
                # Check if stereo is available
                if gt_color_r is not None:
                    gt_depth, flow_valid = get_depth_from_raft(self.raft, gt_color, gt_color_r, self.baseline)
                else:
                    # Reversing preprocessing
                    raw_gt_color = gt_color.squeeze().cpu().numpy() * 255
                    raw_gt_color = raw_gt_color.astype(np.uint8)
                    raw_gt_color = cv2.cvtColor(raw_gt_color, cv2.COLOR_RGB2BGR)

                    # Use Depth Anything V2 to infer depth on monocular datasets; requires raw image data as np array
                    input_size_depth = 512

                    # Getting to relative depth
                    gt_disparity = self.depth_anything.infer_image(raw_gt_color, input_size=input_size_depth)
                    gt_disparity = torch.from_numpy(gt_disparity).cuda().unsqueeze(0)
                    gt_depth = 1 / gt_disparity

                    # Compute relative depth
                    gt_depth = gt_depth / gt_depth.max()

            frame = ids, gt_color, gt_depth, gt_c2w, tool_mask

            # Set cam intrinsics if available
            if intrinsics is not None:
                intrinsics = torch.squeeze(intrinsics)
                self.camera.set_intrinsics(fx=intrinsics[0], fy=intrinsics[1], cx=intrinsics[2], cy=intrinsics[3])

            if ids.item() == 0:
                # Note that we are using the matrix coming from the first frame; some datasets provide a registration
                #  matrix for each frame, but we are using the first frame's matrix for all frames since they do not
                #  change significantly unless the operating table is moved
                self.net.create_from_mesh(self.mesh.copy(), registration, gt_depth, self.scale)

                # Render once to check for visibility filter and initialize the deformation field based on that
                with torch.no_grad():
                    self.camera.set_c2w(gt_c2w)

                    render_pkg = render(self.camera, self.net, self.background, self.scale, deform=False)
                    visibility_filter = render_pkg['visibility_filter']
                    self.net.setup_deformation_field(visibility_filter, self.mesh.copy())

                    # TODO: this should happen somewhere else, method is a mess
                    if os.path.exists(f'{self.output}/net.pth'):
                        print("Loading the model...")
                        state_dict = torch.load(f'{self.output}/net.pth')
                        print("Saved state dict keys:", state_dict.keys())
                        print("Current model state dict keys:", self.net.state_dict().keys())
                        self.net.load_state_dict(state_dict)
                        return
                    else:
                        print("No model found, starting from scratch")

                # Save the original mesh without any preprocessing
                slicer_mesh = self.mesh.copy()
                undo_preprocess_mesh(slicer_mesh, self.cfg['dataset'])
                pv.save_meshio(f'{self.output}/original_mesh.obj', slicer_mesh)

                # Setup for training
                self.net.training_setup(self.cfg['training'])
                # Fit first frame
                self.fit(frame, iters=self.cfg['training']['iters_first'], incremental=False)
            else:
                self.fit(frame, iters=self.cfg['training']['iters'], incremental=False)

            # eval
            with torch.no_grad():
                if self.visualize:
                    # Pass a copy of mesh, will be modified by the visualizer
                    mesh_copy = self.mesh.copy() if self.mesh is not None else None
                    
                    # Get the deformed mesh from the Gaussian model
                    deformed_mesh = self.get_deformed_mesh(mesh_copy, self.net) if self.mesh is not None else None
                    
                    # Get the current deformation control vertices and their deformation values, then visualize them
                    control_vertices_indices = self.net.mesh_deformation.control_ids
                    control_vertices = self.net.original_mesh_vertices[control_vertices_indices].detach().cpu().numpy()
                    control_def = self.net.mesh_deformation.control_def.detach().cpu().numpy()

                    # Save the deformed mesh
                    if deformed_mesh is not None:
                        # Modifies in place, have to create a copy
                        slicer_deformed_mesh = deformed_mesh.copy()
                        # We have to go back to the original mesh/CT space; the registration was applied in the Gaussian model, have to go back
                        undo_register_mesh(slicer_deformed_mesh, registration)
                        undo_preprocess_mesh(slicer_deformed_mesh, self.cfg['dataset'])
                        pv.save_meshio(f'{self.output}/deformed_mesh_{ids.item()}.obj', slicer_deformed_mesh)

                    # General visualization of the current iteration
                    _, _ = self.visualizer.save_imgs(ids.item(), gt_depth, gt_color,
                                                                        gt_c2w, self.scale, mesh=mesh_copy,
                                                                        registration=registration, deformed_mesh=deformed_mesh)
                    # # Visualization of the deformations in particular
                    # self.visualizer.save_mesh_deformations(ids.item(), gt_c2w, mesh_copy, deformed_mesh, registration, control_vertices, control_def, self.scale)

                    if self.ct_volume is not None:
                        if ids.item() == 0: 
                            # Save some slices from different orientations
                            self.visualizer.visualize_ct_slice(
                                self.ct_volume['image'],
                                axis=2,
                                filename=f'ct_axial_middle.png'
                            )
                            self.visualizer.visualize_ct_slice(
                                self.ct_volume['image'],
                                axis=1,
                                filename=f'ct_coronal_middle.png'
                            )
                            self.visualizer.visualize_ct_slice(
                                self.ct_volume['image'],
                                axis=0,
                                filename=f'ct_sagittal_middle.png'
                            )
                        
                        # Warp the CT based on the deformed mesh
                        try:
                            warped_ct = self.visualizer.warp_ct_volume(self.ct_volume, slicer_deformed_mesh.copy(), slicer_mesh.copy(), control_vertices_indices.cpu().numpy())
                            
                            # Save slices of the warped CT
                            self.visualizer.visualize_ct_slice(
                                warped_ct,
                                axis=2,
                                filename=f'warped_ct_axial_middle_{ids.item()}.png'
                            )
                            self.visualizer.visualize_ct_slice(
                                warped_ct,
                                axis=1,
                                filename=f'warped_ct_coronal_middle_{ids.item()}.png'
                            )
                            self.visualizer.visualize_ct_slice(
                                warped_ct,
                                axis=0,
                                filename=f'warped_ct_sagittal_middle_{ids.item()}.png'
                            )
                        except Exception as e:
                            print(f"Error during warping: {e}")
                            import traceback
                            traceback.print_exc()

                if self.eval:
                    # Load the ground truth mesh
                    # Have to use the first element because of this DataLoader unexpected behavior: https://stackoverflow.com/questions/64883998/pytorch-dataloader-shows-odd-behavior-with-string-dataset
                    gt_mesh = pv.read(gt_mesh_path[0])

                    # TODO: fix this later, can lead to crash if self.visualize is False because the slicer_deformed_mesh is not available
                    deformed_points = slicer_deformed_mesh.points
                    gt_points = gt_mesh.points
                    diff_vertices = deformed_points - gt_points
                    l2_norms = np.linalg.norm(diff_vertices, axis=1)

                    # Mean and std of the L2 norms
                    mean_l2_error = np.mean(l2_norms)
                    std_l2_error = np.std(l2_norms)
                    max_l2_error = np.max(l2_norms)

                    print(f"Frame {ids.item()}:")
                    print(f'Mean L2 error: {mean_l2_error}')
                    print(f'Std L2 error: {std_l2_error}')
                    print(f'Max L2 error: {max_l2_error}\n')

                    mesh_error_l2.append(mean_l2_error)
                    mesh_error_std.append(std_l2_error)
                    if max_l2_error > global_max_error:
                        global_max_error = max_l2_error

        if self.eval:
            print(f"Global metrics over all frames:")
            print(f'Mean L2 error: {np.mean(mesh_error_l2)}')
            print(f'Std L2 error: {np.mean(mesh_error_std)}')
            print(f'Max L2 error: {global_max_error}')
        print('...finished')


if __name__ == "__main__":
    # Set up command line argument parser
    from src.config import load_config
    import random

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('config', type=str)
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--log', type=str)
    parser.add_argument('--log_group', type=str, default='default')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/base.yaml')
    cfg['data']['output'] = args.output if args.output else cfg['data']['output']

    trainer = SceneOptimizer(cfg, args)

    print("Start training...")
    trainer.run()
    print("Training finished")

    # Estimate pose for the first frame
    print("Start pose estimation...")
    trainer.estimate_pose(cfg['data']['target_image'], max_iter=500, camera_idx=67)
    print("Pose estimation finished")
