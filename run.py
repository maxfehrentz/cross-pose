import numpy as np
import os
import torch
import wandb
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from scipy.spatial import KDTree
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
from torch.utils.data import DataLoader
from src.utils.camera import Camera
from src.utils.FrameVisualizer import FrameVisualizer
from src.utils.flow_utils import get_scene_flow, get_depth_from_raft
from src.utils.PointTracker import PointTracker, mte, surv_2d, delta_2d
from src.utils.datasets import StereoMIS, Atlas
from src.utils.loss_utils import l1_loss
from src.utils.renderer import render
from src.scene.gaussian_model import GaussianModel, MeshAwareGaussianModel
from src.utils.mesh_utils import preprocess_mesh
from src.utils.transform_utils import SCALE, SWAP_AND_FLIP_WORLD_AXES

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


class SceneOptimizer():
    def __init__(self, cfg, args):
        self.total_iters = 0
        self.cfg = cfg
        self.args = args
        self.visualize = args.visualize
        self.scale = SCALE
        self.device = cfg['device']
        self.output = cfg['data']['output']

        self.frame_reader = Atlas(cfg, args, scale=self.scale)
        # self.frame_reader = StereoMIS(cfg, args, scale=self.scale)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=0 if args.debug else 4)
        # self.net = GaussianModel(cfg=cfg['model'])
        self.net = MeshAwareGaussianModel(cfg=cfg['model'])
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

        self.pt_tracker = None
        # track_file = os.path.join(cfg['data']['input_folder'], 'track_pts.pckl')
        # if os.path.isfile(track_file):
        #     self.pt_tracker = PointTracker(cfg, self.net, track_file)
        # self.last_frame = None
        # self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        # self.raft = self.raft.eval()

        # Checking for baseline, not available if monocular dataset
        self.baseline = cfg['cam'].get('stereo_baseline', None)
        if self.baseline is not None:
            self.baseline = self.baseline / 1000.0 * self.scale

        # Checking for mesh
        self.mesh = cfg.get('data', {}).get('mesh', None)
        # Load mesh
        if self.mesh is not None:
            self.mesh = pv.read(self.mesh)
            # Preprocesses mesh in place
            preprocess_mesh(self.mesh)

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
                render_pkg = render(self.camera, self.net, self.background, deform=incremental)
            self.net.eval()
            color = render_pkg['render'][None, ...]
            depth = render_pkg['depth'][None, ...]

            # Convert rendered depth map to a relative depth map
            depth = depth / depth.max()

            # Loss
            with record_function("photometric loss computation"):
                Ll1 = self.cfg['training']['w_color']*l1_loss(color[tool_mask], gt_color[tool_mask])
                # Ll1_depth = self.cfg['training']['w_depth']*l1_loss(depth[tool_mask], gt_depth[tool_mask])

            # loss = Ll1 + Ll1_depth
            loss = Ll1

            if isinstance(self.net, MeshAwareGaussianModel):
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
                # Saving
                self.net.add_densification_stats(viewspace_point_tensor_grad, render_pkg["visibility_filter"])

                # Densification
                if iter > self.cfg["training"]["densify_from_iter"] and iter % self.cfg["training"]["densification_interval"] == 0:
                    self.net.densify(self.cfg["training"]["densify_grad_threshold"])

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
        deformed_vertices = (transform[:3, :3] @ deformed_vertices.T).T / SCALE
        # Ensure array is contiguous and in the correct dtype
        deformed_vertices = np.ascontiguousarray(deformed_vertices, dtype=np.float32)
        deformed_mesh.points = deformed_vertices

        # Have to recompute normals after changing the vertex positions
        deformed_mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=True,  # Crucial here, otherwise normals are not computed correctly
            inplace=True
        )

        return deformed_mesh
    

    def run(self):
        torch.cuda.empty_cache()
        pt_track_stats = {"pred_2d": []}

        for ids, gt_color, gt_color_r, gt_c2w, registration, tool_mask, semantics, intrinsics in tqdm(self.frame_loader, total=self.n_img):
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
                self.net.create_from_mesh(self.mesh.copy(), registration, gt_depth)

                # Render once to check for visibility filter and initialize the deformation field based on that
                with torch.no_grad():
                    self.camera.set_c2w(gt_c2w)
                    render_pkg = render(self.camera, self.net, self.background, deform=False)
                    visibility_filter = render_pkg['visibility_filter']
                    self.net.setup_deformation_field(visibility_filter, self.mesh.copy())

                # Setup for training
                self.net.training_setup(self.cfg['training'])
                # Fit first frame
                self.fit(frame, iters=self.cfg['training']['iters_first'], incremental=True)
            else:
                if ids.item() == 1:
                    if self.cfg['training']['grad_weighing']:
                        self.net.enable_grad_weighing(True)
                self.fit(frame, iters=self.cfg['training']['iters'], incremental=True)

            self.last_frame = gt_color.detach()

            # eval
            with torch.no_grad():
                if self.visualize:
                    # Pass a copy of mesh, will be modified by the visualizer
                    mesh_copy = self.mesh.copy() if self.mesh is not None else None
                    
                    # Get the deformed mesh from the Gaussian model
                    deformed_mesh = self.get_deformed_mesh(mesh_copy, self.net) if self.mesh is not None else None
                    
                    # Get the current deformation control vertices and their deformation values, then visualize them by rendering the mesh, the deformed mesh, and add vectors to the control vertices
                    control_vertices_indices = self.net.mesh_deformation.control_ids
                    control_vertices = self.net.original_mesh_vertices[control_vertices_indices].detach().cpu().numpy()
                    control_def = self.net.mesh_deformation.control_def.detach().cpu().numpy()

                    outmap, outsem = self.visualizer.save_imgs(ids.item(), gt_depth, gt_color,
                                                                        gt_c2w, mesh=mesh_copy,
                                                                        registration=registration, deformed_mesh=deformed_mesh)
                    # Render and save the deformations
                    self.visualizer.save_mesh_deformations(ids.item(), gt_c2w, mesh_copy, deformed_mesh, registration, control_vertices, control_def)

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
    trainer.run()
