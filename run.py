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
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


# For the preoperative CT
import pyvista as pv


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
        track_file = os.path.join(cfg['data']['input_folder'], 'track_pts.pckl')
        if os.path.isfile(track_file):
            self.pt_tracker = PointTracker(cfg, self.net, track_file)
        self.last_frame = None
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.raft = self.raft.eval()

        # Checking for baseline, not available if monocular dataset
        self.baseline = cfg['cam'].get('stereo_baseline', None)
        if self.baseline is not None:
            self.baseline = self.baseline / 1000.0 * self.scale

        # Checking for mesh
        self.mesh = cfg.get('mesh', None)
        # Load mesh
        if self.mesh is not None:
            self.mesh = pv.read(self.mesh)
            # Preprocesses mesh in place
            preprocess_mesh(self.mesh)

        # Initialize Depth Anything V2 model; using metric depth estimation as described here
        # https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        encoder = 'vitl'
        dataset = 'hypersim' # 'hypersim' or 'vkitti
        # TODO: document the "scaling factor" max depth in the Readme if this is used in the end
        self.depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': .2})
        self.depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()


    def fit(self, frame, iters, incremental):
        self.net.reset_optimizer()
        av_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
        idx, gt_color, gt_depth, gt_c2w, tool_mask = frame
        self.camera.set_c2w(gt_c2w)
        for iter in range(1, iters+1):
            if self.cfg['training']['spherical_harmonics'] and iter > iters/2:
                self.net.enable_spherical_harmonics()
            self.total_iters += 1
            self.net.train(iter == 1)
            render_pkg = render(self.camera, self.net, self.background, deform=incremental)
            self.net.eval()
            color = render_pkg['render'][None, ...]
            depth = render_pkg['depth'][None, ...]

            # Loss
            Ll1 = self.cfg['training']['w_color']*l1_loss(color[tool_mask], gt_color[tool_mask])
            Ll1_depth = self.cfg['training']['w_depth']*l1_loss(depth[tool_mask]/self.scale, gt_depth[tool_mask]/self.scale)
            # TODO: depth loss taken out for now, come back to this
            #loss = Ll1 + Ll1_depth
            loss = Ll1

            if incremental:
                l_rigidtrans, l_rigidrot, l_iso, l_visible = self.net.compute_regulation(render_pkg["visibility_filter"])
                def_loss = self.cfg['training']['w_def']['rigid']*l_rigidtrans + self.cfg['training']['w_def']['iso']*l_iso+ self.cfg['training']['w_def']['rot']*l_rigidrot+ self.cfg['training']['w_def']['nvisible']*l_visible
                loss += def_loss
            else:
                l_rigidtrans, l_rigidrot, l_iso, l_visible = torch.zeros_like(Ll1), torch.zeros_like(Ll1), torch.zeros_like(Ll1), torch.zeros_like(Ll1)
            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(render_pkg["viewspace_points"])
            viewspace_point_tensor_grad += render_pkg["viewspace_points"].grad

            ########### Logging & Evaluation ###################
            with torch.no_grad():
                av_loss[0] += Ll1.item()
                av_loss[1] += Ll1_depth.item()
                av_loss[2] += l_rigidtrans.item()
                av_loss[3] += l_rigidrot.item()
                av_loss[4] += l_iso.item()
                av_loss[5] += l_visible.item()
                av_loss[-1] += 1
                if ((self.total_iters % self.log_freq) == 0) and self.log:
                    wandb.log({'color_loss': av_loss[0] / av_loss[-1],
                               'depth_loss': av_loss[1] / av_loss[-1],
                               'rigidtrans_loss': av_loss[2] / av_loss[-1],
                               'rigidrot_loss': av_loss[3] / av_loss[-1],
                               'iso_loss': av_loss[4] / av_loss[-1],
                               'visible_loss': av_loss[5] / av_loss[-1],
                               'loss': sum(av_loss[:-1]) / av_loss[-1]}, step=self.total_iters)
                    av_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]

                self.net.add_densification_stats(viewspace_point_tensor_grad, render_pkg["visibility_filter"])

                # Densification
                if iter > self.cfg["training"]["densify_from_iter"] and iter % self.cfg["training"]["densification_interval"] == 0:
                    self.net.densify(self.cfg["training"]["densify_grad_threshold"])

                # Optimizer step
                if iter < iters:
                    self.net.optimizer.step()
                    self.net.optimizer.zero_grad(set_to_none=True)


    def get_deformed_mesh(self, mesh, pc):
        # Get the deformed mesh vertices from the Gaussian model
        deformed_mesh = mesh.copy()
        deformed_vertices = pc.mesh_vertices.detach().cpu().numpy()

        # Transform to original mesh space! Bear in mind that those vertices do not come from the original mesh
        #  but from the optimization in the world space of the Gaussians. Need to apply the same transform
        #  (actually its inverse) to go back to the original mesh space that this rendering assumes
        transform = SWAP_AND_FLIP_WORLD_AXES
        deformed_vertices = (transform[:3, :3] @ deformed_vertices.T).T / SCALE
        # Ensure array is contiguous and in the correct dtype
        deformed_vertices = np.ascontiguousarray(deformed_vertices, dtype=np.float32)
        deformed_mesh.points = deformed_vertices
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

                    # Multiplying by 100 to get centimeters, although some github issues claim it outputs dm
                    # Also, beware that the metric depth estimation of this model seems to be highly specific to the
                    #  datasets it was trained on since it only outputs 0-1 values and is then scaled by max_depth
                    #  which was fixed for each dataset. Therefore, it is highly unlikely that we have a real metric
                    #  depth estimation on our OOD data here. We just got lucky with choosing the 'right' although
                    #  potentially metrically meaningless max_depth value for our dataset.
                    gt_depth = self.depth_anything.infer_image(raw_gt_color, input_size=input_size_depth) * 100

                    # Make sure the depth map is in the correct format
                    gt_depth = torch.from_numpy(gt_depth).cuda().unsqueeze(0)

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
                self.net.training_setup(self.cfg['training'])
                self.fit(frame, iters=self.cfg['training']['iters_first'], incremental=False)
            else:
                if ids.item() == 1:
                    if self.cfg['training']['grad_weighing']:
                        self.net.enable_grad_weighing(True)

                #with torch.no_grad():
                #    # add new points
                #    self.camera.set_c2w(gt_c2w)
                #    # Have to render to get the current alpha values; only adding Gaussians if opactiy < threshold
                #    render_pkg = render(self.camera, self.net, self.background, deform=True)
                #    mask = render_pkg['alpha'][None,...,None].squeeze(-1) < 0.95
                #    mask &= tool_mask
                #    self.net.add_from_pcd(gt_color, gt_depth, gt_c2w, self.camera, mask, semantics=semantics) if self.cfg['training']['add_points'] else 0.0

                self.fit(frame, iters=self.cfg['training']['iters'], incremental=True)
            self.last_frame = gt_color.detach()

            # eval
            with torch.no_grad():
                log_dict = {}
                if self.pt_tracker is not None:
                        if not self.pt_tracker.is_initialized():
                            self.pt_tracker.init_tracking_points(gt_c2w)
                        pts_3d_gt, pts_3d, pts_2d, l2_3d, l2_2d, pts_2d_gt = self.pt_tracker.eval(gt_c2w, ids.item())
                        pt_track_stats["pred_2d"].append(pts_2d.cpu().numpy())
                        log_dict.update({'pt_track_l2_2d': l2_2d, 'frame': ids[0].item()})
                else:
                    pts_2d, pts_2d_gt = None, None
                if self.visualize:
                    # Pass a copy of mesh, will be modified by the visualizer
                    mesh_copy = self.mesh.copy() if self.mesh is not None else None
                    
                    # Get the deformed mesh vertices from the Gaussian model
                    deformed_mesh = self.get_deformed_mesh(mesh_copy, self.net) if self.mesh is not None else None

                    outmap, outsem, outrack = self.visualizer.save_imgs(ids.item(), gt_depth, gt_color,
                                                                        gt_c2w, pts_2d, pts_2d_gt, mesh=mesh_copy,
                                                                        registration=registration, deformed_mesh=deformed_mesh)
                    if self.log:
                        log_dict.update({'mapping': wandb.Image(outmap),
                                         'tracking': wandb.Image(outrack) if outrack is not None else None,
                                         'semantic': wandb.Image(outsem)})
                if self.log:
                    wandb.log(log_dict)

        if self.log:
            # eval point tracking
            gt_2d, valid = self.pt_tracker.get_gt_2d_pts()
            pred_2d = np.stack(pt_track_stats["pred_2d"], axis=1)
            H, W = self.camera.get_params()[:2]
            wandb.summary['MTE_2D'] = mte(pred_2d, gt_2d, valid)
            wandb.summary['delta_2D'] = delta_2d(pred_2d, gt_2d, valid, H, W)
            wandb.summary['survival_2D'] = surv_2d(pred_2d, gt_2d, valid, H, W)
        with open(os.path.join(self.output, 'tracked.pckl'), 'wb') as f:
            pickle.dump(pt_track_stats, f)
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
