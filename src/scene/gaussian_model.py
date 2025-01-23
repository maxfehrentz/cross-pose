import torch
from torch import nn
import numpy as np
from src.utils.sh_utils import RGB2SH
from src.utils.mesh_utils import register_mesh
from src.utils.transform_utils import SWAP_AND_FLIP_WORLD_AXES, SCALE
from simple_knn._C import distCUDA2
from src.utils.general_utils import strip_symmetric, build_scaling_rotation, build_inv_cov, inverse_sigmoid, build_rotation
from src.scene.deformation import ExplicitDeformation, ExplicitSparseDeformation, MeshSparseDeformation
from src.utils.flow_utils import get_surface_pts
from functools import partial
from vtk.util.numpy_support import vtk_to_numpy
import os
from torch.autograd import Function
import arap_cpp


class GaussianModel(nn.Module):
    def __init__(self, cfg, n_classes=7):
        super().__init__()
        self.cfg = cfg
        self.active_sh_degree = 0
        self.max_sh_degree = 1
        self._xyz = torch.empty(0)
        self._semantics = torch.empty(0)
        if cfg["deform_network"]['model'] == 'sparse':
            self._deformation = ExplicitSparseDeformation(subsample=cfg['deform_network']['subsample'])
        elif cfg["deform_network"]['model'] == 'dense':
            self._deformation = ExplicitDeformation()
        else:
            raise NotImplementedError
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.hooks = None
        self.n_classes = n_classes

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def forward(self, deform=True):
        """
            apply deformation and return 3D Gaussians for rasterization
        """
        # TODO: not sure why this yields so much better results, come back to this
        # Disable gradient for scaling
        self._scaling.requires_grad_(False)

        if deform:
            xyz, scales, rots = self._deformation(self._xyz, self._scaling, self._rotation, init=self.training)
        else:
            xyz, scales, rots = self._xyz, self._scaling, self._rotation
        scales = self.scaling_activation(scales)
        rots = self.rotation_activation(rots)
        opacity = self.opacity_activation(self._opacity)
        return xyz, scales, rots, opacity, self.get_features, self.get_semantics

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_semantics(self):
        return self._semantics

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def enable_spherical_harmonics(self):
        self.active_sh_degree = 1

    def add_from_pcd(self, rgb, depth, c2w, camera, mask, downsample: int=2, semantics=None):
        semantics = torch.ones((*depth.shape, self.n_classes)).squeeze(0) if semantics is None else semantics.squeeze(0)
        # reproject points to 3D
        H, W, fx, fy, cx, cy = camera.get_params()
        points = get_surface_pts(depth, fx, fy, cx, cy, c2w, 'cuda')
        point_cloud = points[:, ::downsample, ::downsample].reshape(-1, 3).cuda()
        # filter points in already visited areas
        color = RGB2SH(rgb[:, ::downsample, ::downsample].reshape(-1, 3).cuda())
        features = torch.zeros((color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = color
        features[:, 3:, 1:] = 0.0
        semantics = semantics[::downsample, ::downsample].reshape(-1,self.n_classes).cuda()

        dist2 = torch.clamp_min(distCUDA2(point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.6*torch.ones((point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        selected_pts_mask = mask[:, ::downsample, ::downsample].reshape(-1).cuda()

        if not selected_pts_mask.float().mean() > 0.0:
            return 0.0

        self.densification_postfix(point_cloud[selected_pts_mask],
                                   features[:,:,0:1][selected_pts_mask].transpose(1, 2),
                                   features[:,:,1:][selected_pts_mask].transpose(1, 2), opacities[selected_pts_mask],
                                   scales[selected_pts_mask], rots[selected_pts_mask],
                                   semantics[selected_pts_mask])
        return selected_pts_mask.float().mean()

    def create_from_mesh(self, mesh, registration, depth, spatial_lr_scale=1.0):
        self.spatial_lr_scale = spatial_lr_scale
        register_mesh(mesh, registration)
        # Compute triangle centers
        triangle_centers = torch.tensor(mesh.cell_centers().points, dtype=torch.float32).reshape(-1,3).cuda()
        # Have to apply the same world coordinate convention change and scaling to go to Nerfstudio convention as the
        #  cameras, see dataloader
        transform = SWAP_AND_FLIP_WORLD_AXES.to(triangle_centers.device)
        triangle_centers = (transform[:3, :3] @ triangle_centers.T).T * SCALE
        points = triangle_centers
        rgb_norm = torch.ones_like(points)
        # TODO: Fused stuff here came initally from additional tool masking; have to add that later, see create_from_pcd
        fused_point_cloud = points
        fused_color = RGB2SH(rgb_norm)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        # Computing distance between points, using as heuristic to compute scale
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.6*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # TODO: fix _semantics later if necessary, see create_from_pcd
        # TODO: unintuitive that we need depth to intialize the shape here, should change that as well. Misleading since
        #  we do not need the actual depth
        self._semantics = torch.ones((*depth.shape, self.n_classes)).squeeze(0).cuda()
        self._deformation = self._deformation.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._deformation.add_gaussians(self._xyz.shape[0], self._xyz)

    def create_from_pcd(self, rgb, depth, c2w, camera, tool_mask=None, spatial_lr_scale : float=1.0, downsample: int=2, semantics=None):
        with torch.no_grad():
            tool_mask = torch.ones_like(depth).bool() if tool_mask is None else tool_mask
            semantics = torch.ones((*depth.shape, self.n_classes)).squeeze(0).cuda() if semantics is None else semantics.squeeze(0)
            self.spatial_lr_scale = spatial_lr_scale
            # reproject points to 3D
            H, W, fx, fy, cx, cy = camera.get_params()
            points = get_surface_pts(depth, fx, fy, cx, cy, c2w, depth.device).squeeze(0)
            rgb_norm = camera.inverse_splotlight_render(rgb.cuda(), points[...,2].cuda()).squeeze(0)
            fused_point_cloud = points[::downsample, ::downsample][tool_mask.squeeze(0)[::downsample, ::downsample]].reshape(-1,3).cuda()
            fused_color = RGB2SH(rgb_norm[::downsample, ::downsample][tool_mask.squeeze(0)[::downsample, ::downsample]].reshape(-1,3))
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            semantics = semantics[::downsample, ::downsample][tool_mask.squeeze(0)[::downsample, ::downsample]].reshape(-1,self.n_classes).cuda()

            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.6*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._semantics = semantics
        self._deformation = self._deformation.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._deformation.add_gaussians(self._xyz.shape[0], self._xyz)

    def training_setup(self, training_args):
        self.percent_dense = training_args["percent_dense"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args["position_lr_init"] * self.spatial_lr_scale, "name": "xyz"},
            {'params': self._deformation.parameters(), 'lr': training_args["deformation_lr_init"] * self.spatial_lr_scale, "name": "deformation"},
            {'params': [self._features_dc], 'lr': training_args["feature_lr"], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args["feature_lr"] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args["opacity_lr"], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args["scaling_lr"], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args["rotation_lr"], "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            optimizable_tensors[group["name"]] = []
            for idx in range(len(group['params'])):
                if len(mask) != len(group['params'][idx]):
                    continue
                stored_state = self.optimizer.state.get(group['params'][idx], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][idx]]
                    group["params"][idx] = nn.Parameter((group["params"][idx][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][idx]] = stored_state

                    optimizable_tensors[group["name"]].append(group["params"][idx])
                else:
                    group["params"][idx] = nn.Parameter(group["params"][idx][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]].append(group["params"][idx])
        # squeeze tensors
        for key in optimizable_tensors:
            if len(optimizable_tensors[key]) == 1:
                optimizable_tensors[key] = optimizable_tensors[key][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        grad_weighing = self.hooks is not None
        self.enable_grad_weighing(False)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.gradient_accum = self.gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        # TODO: fix semantics later
        # self._semantics = self._semantics[valid_points_mask]
        self.enable_grad_weighing(grad_weighing)
        self._deformation.replace(optimizable_tensors['deformation'], optimizable_tensors['xyz'], reinit=True)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue
            optimizable_tensors[group["name"]] = []
            if len(group['params']) == 1:
                extension_tensors = [tensors_dict[group["name"]]]
            else:
                extension_tensors = tensors_dict[group["name"]]
            for idx, extension_tensor in enumerate(extension_tensors):
                stored_state = self.optimizer.state.get(group['params'][idx], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][idx]]
                    group["params"][idx] = nn.Parameter(torch.cat((group["params"][idx], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][idx]] = stored_state

                    optimizable_tensors[group["name"]].append(group["params"][idx])
                else:
                    group["params"][idx] = nn.Parameter(torch.cat((group["params"][idx], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]].append(group["params"][idx])
        # squeeze tensors
        for key in optimizable_tensors:
            if len(optimizable_tensors[key]) == 1:
                optimizable_tensors[key] = optimizable_tensors[key][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantics):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation
       }

        if isinstance(self._deformation, ExplicitDeformation):
            new_deformation = self._deformation.get_new_params(new_xyz.shape)
            d['deformation'] = new_deformation
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        grad_weighing = self.hooks is not None
        self.enable_grad_weighing(False)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gradient_accum = torch.cat((self.gradient_accum, torch.zeros((new_xyz.shape[0], 1), device="cuda")), dim=0)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # TODO: fix semantics later
        # self._semantics = torch.cat((self._semantics, new_semantics), dim=0)
        self.enable_grad_weighing(grad_weighing)
        self._deformation.replace(optimizable_tensors['deformation'], optimizable_tensors['xyz'], reinit=False)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # exclude Gaussians which are settled based on grad_weighing
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.gradient_accum.squeeze() < self.cfg['visit_offset'])

        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # TODO: fix the semantics issues later if necessary
        # new_semantics = self._semantics[selected_pts_mask].repeat(N,1)
        new_semantics = self._semantics
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantics)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # exclude Gaussians which are settled based on grad_weighing
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.gradient_accum.squeeze() < self.cfg['visit_offset'])
        new_xyz = self._xyz[selected_pts_mask] 
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # TODO: fix this semantics issue later
        # new_semantics = self._semantics[selected_pts_mask]
        new_semantics = self._semantics

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantics)

    def densify(self, max_grad):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, 1.0)
        self.densify_and_split(grads, max_grad, 1.0)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.gradient_accum[update_filter] += 1

    def compute_regulation(self, visibility_filter):
        return self._deformation.reg_loss(visibility_filter)

    def get_closest_gaussian(self, point, use_cov=False, k=1):
        xyz, scales, rots, opacity, features, _ = self.forward()
        delta = point - xyz
        if use_cov:
            inv_cov = build_inv_cov(scales, rots)
            affinity = (opacity*torch.exp(-(delta[..., None, :] @ inv_cov @ delta[..., None]).squeeze(1)/2)).squeeze(1)
        else:
            affinity = -torch.linalg.norm(delta, dim=-1, ord=2)#(opacity * torch.exp(-(delta[..., None, :] @ delta[..., None]).squeeze(1) / 2)).squeeze(1)
        idx = torch.topk(affinity, k=k).indices
        return self._xyz[idx].mean(dim=0), idx, xyz[idx].mean(dim=0), affinity[idx]

    def enable_grad_weighing(self, enable=True):
        visit_alpha = self.cfg['visit_alpha']
        visit_offset = self.cfg['visit_offset']
        if visit_alpha is not None:
            if enable and (self.hooks is None):
                norm = self.grad_weighing(torch.ones(1, device="cuda"),
                                          torch.zeros(1, device="cuda"),
                                          visit_alpha, visit_offset,
                                          torch.ones(1, device="cuda"))
                # register grad hook
                hooks = []
                for param in [self._xyz, self._rotation, self._scaling, self._opacity, self._features_dc]:
                    hooks.append(param.register_hook(
                        partial(self.grad_weighing, visited=self.gradient_accum, visit_alpha=visit_alpha,
                                visit_offset=visit_offset, norm=norm)))
                hooks.append(self._features_rest.register_hook(
                    partial(self.grad_weighing, visited=self.gradient_accum, visit_alpha=visit_alpha,
                            visit_offset=visit_offset, norm=norm, offset=0.001)))
                self.hooks = hooks
            else:
                if self.hooks is not None:
                    [h.remove() for h in self.hooks]
                    self.hooks = None

    @staticmethod
    def grad_weighing(grad, visited, visit_alpha, visit_offset, norm, offset=0.0):
        """
            weight gradient by visit function -> points that have often been updated will get smaller gradient
        """
        #ToDo make broadcasting without transpose as it uses non contiguous views
        return (grad.transpose(0, -1) * ((1.0+offset) - torch.sigmoid(visit_alpha * (visited.squeeze() - visit_offset))) / norm).transpose(0, -1)

    def reset_optimizer(self):
        for group in self.optimizer.param_groups:
            for idx in range(len(group['params'])):
                stored_state = self.optimizer.state.get(group['params'][idx], None)
                if stored_state is not None:
                    stored_state["exp_avg"][:] = 0.0
                    stored_state["exp_avg_sq"][:] = 0.0
                    stored_state["state_step"] = 0
                    self.optimizer.state[group['params'][idx]] = stored_state


# Can sublcass autograd.Function to implement custom forward AND backward pass
class ArapFunction(Function):
    @staticmethod
    def forward(ctx, vertices, original_vertices, neighbors, neighbor_mask):
        # Ensure inputs are on CPU and contiguous
        # TODO: change that later when we have a GPU version
        # TODO: depending on the implementation, we may not need the neighbor padding and neighbor_mask; adjacency lists could work
        vertices = vertices.cpu().contiguous()
        original_vertices = original_vertices.cpu().contiguous()
        neighbors = neighbors.cpu().contiguous()
        neighbor_mask = neighbor_mask.cpu().contiguous()
        
        energy, rotations = arap_cpp.compute_arap_energy(
            vertices, original_vertices, neighbors, neighbor_mask)
        
        # TODO: same here, check if padded neighbors and neighbor_mask are needed
        # Save everything needed for backward
        ctx.save_for_backward(vertices, original_vertices, neighbors, 
                            neighbor_mask, rotations)
        return energy.cuda()

    @staticmethod
    def backward(ctx, grad_output):
        vertices, original_vertices, neighbors, neighbor_mask, rotations = ctx.saved_tensors
        
        # Compute gradients using C++ implementation
        grad_vertices = arap_cpp.compute_arap_gradient(
            vertices, original_vertices, neighbors, neighbor_mask,
            rotations, grad_output)
        
        # Note that we are ignoring the gradients of R w.r.t. the vertices since its an independent parameter and only depends on
        #  the vertices due to the flip-flop trick
        return grad_vertices.cuda(), None, None, None


class MeshAwareGaussianModel(GaussianModel):
    def __init__(self, cfg, n_classes=7):
        super().__init__(cfg, n_classes)
        self.current_mesh_vertices = None
        self.mesh_faces = None
        self.barycentric_coords = None
        self._xyz = torch.empty(0)
        self.original_mesh_vertices = None
        self.original_edges = None
        self.neighbors = None
        self.mesh_deformation = None

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_mesh_vertices(self):
        return self.current_mesh_vertices
        
    def create_from_mesh(self, mesh, registration, depth, spatial_lr_scale=1.0):
        self.spatial_lr_scale = spatial_lr_scale

        register_mesh(mesh, registration)
        
        # Get vertices
        vertices = torch.tensor(mesh.points, dtype=torch.float32)
        # Reorient and scale
        transform = SWAP_AND_FLIP_WORLD_AXES
        vertices = (transform[:3, :3] @ vertices.T).T * SCALE
        vertices = vertices.cuda()

        # Get faces (see explanation here https://stackoverflow.com/questions/51201888/retrieving-facets-and-point-from-vtk-file-in-python)
        triangle_cells = mesh.GetPolys()
        face_array = triangle_cells.GetData()
        faces = torch.tensor(vtk_to_numpy(face_array).reshape(-1, 4)[:, 1:], dtype=torch.int64).cuda()
        
        # Keep track of current vertices and original vertices
        self.current_mesh_vertices = vertices
        self.original_mesh_vertices = vertices
        # Topology is constant
        n_triangles = faces.shape[0]
        self.mesh_faces = faces
        
        # TODO: change later, fixing this and not making it a parameter for now
        # self.barycentric_coords = nn.Parameter(torch.ones(
        #     n_triangles, 3).cuda() / 3.0)   
        self.barycentric_coords = torch.ones(n_triangles, 3).cuda() / 3.0

        # Compute initial positions, no deformation yet
        xyz, _ = self.compute_positions_and_normals(deform=False)

        # TODO: remove later if possible, super hacky; for some reason, just allocating a dummy tensor fixes the illegal memory access issues?? memory alignment issue?
        dummy_tensor = torch.zeros((xyz.shape[0], 3), dtype=torch.float32, device="cuda")
        
        # Initialize other parameters as before
        rgb_norm = torch.ones((n_triangles, 3)).cuda()
        fused_color = RGB2SH(rgb_norm)
        features = torch.zeros((n_triangles, 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        # Initialize scales, rotations, opacities
        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((n_triangles, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.6*torch.ones((n_triangles, 1), device="cuda"))

        self._xyz = xyz
        # Store as parameters with explicit contiguous memory layout
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # Initialize semantics if needed
        self._semantics = torch.ones((*depth.shape, self.n_classes), device="cuda").squeeze(0)

        
        print("Precomputing neighbor information...")
        self.padded_neighbors, self.neighbor_mask = self.build_neighbors(faces)

        print("Precomputing original edges...")
        # Precomputing the original edges, used to regularize and won't change
        #   Unsqueezing along dim 1 -> (#vertices, broadcasted to #neighbors, 3)
        #   Indexing with padded neighbors -> (#vertices, #max_neighbors, 3)
        #   Can then subtract to get all edge vectors, shape (#vertices, #max_neighbors, 3)
        self.original_edges = self.original_mesh_vertices.unsqueeze(1) - self.original_mesh_vertices[self.padded_neighbors]

        # Initialize mesh deformation
        self.mesh_deformation = MeshSparseDeformation(self.original_mesh_vertices, subsample=64)

    def training_setup(self, training_args):
        self.percent_dense = training_args["percent_dense"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': self.mesh_deformation.parameters(), 'lr': training_args["deformation_lr_init"] * self.spatial_lr_scale, "name": "deformation"},
            {'params': [self._features_dc], 'lr': training_args["feature_lr"], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args["feature_lr"] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args["opacity_lr"], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args["scaling_lr"], "name": "scaling"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
    def compute_positions_and_normals(self, deform):
        if deform:
            vertices = self.mesh_deformation(self.original_mesh_vertices)
            self.current_mesh_vertices = vertices
        else:
            vertices = self.current_mesh_vertices
        
        # Get vertices for each triangle face
        v0 = vertices[self.mesh_faces[:, 0]]
        v1 = vertices[self.mesh_faces[:, 1]]
        v2 = vertices[self.mesh_faces[:, 2]]

        # Compute triangle centers as position for each Gaussian
        positions = ((v0 + v1 + v2) / 3.0)
        self._xyz = positions

        # Compute face normals
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.linalg.cross(edge1, edge2, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        
        return positions, face_normals
    
    def compute_rotations_from_normals(self, normals):
        # Reference direction (z-axis)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device)
        
        # Compute rotation axis (cross product)
        rotation_axis = torch.linalg.cross(z_axis.expand_as(normals), normals)
        
        # Handle case where normal is parallel to z-axis; x-axis is used as fallback
        # TODO: make sure this does not cause bugs, not sure about x-axis fallback
        parallel_mask = torch.norm(rotation_axis, dim=1) < 1e-6
        rotation_axis[parallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=normals.device)

        # Normalize rotation axis
        rotation_axis = torch.nn.functional.normalize(rotation_axis, dim=1)
        
        # Compute rotation angle; cos(angle) is the dot product between normals and z-axis, acos retrieves angle in radians
        cos_angle = torch.sum(normals * z_axis.expand_as(normals), dim=1)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        
        # Convert to quaternion (w, x, y, z format) where  q = (w, x, y, z) = (cos(θ/2), axsin(θ/2), aysin(θ/2), azsin(θ/2)) with ax, ay, az being the rotation axis
        sin_half_angle = torch.sin(angle/2)[:, None]
        cos_half_angle = torch.cos(angle/2)[:, None]
        quat = torch.cat([
            cos_half_angle,  # w component first
            rotation_axis * sin_half_angle  # xyz components
        ], dim=1)
        
        return quat
        
    def forward(self, deform=True):
        # Deactivate learning for scale
        self._scaling.requires_grad = False
        # Compute Gaussian positions from mesh and parameters
        xyz, normals = self.compute_positions_and_normals(deform)
        rots = self.rotation_activation(self.compute_rotations_from_normals(normals))
        scales = self.scaling_activation(self._scaling)
        opacity = self.opacity_activation(self._opacity)
        return xyz, scales, rots, opacity, self.get_features, self.get_semantics
    
    def compute_arap_energy_python(self):
        curr_pos = self.mesh_deformation(self.original_mesh_vertices)

        # Compute all edges at once using broadcasting and precomputed padded neighbors
        #   Unsqueezing along dim 1 -> (#vertices, broadcasted to #neighbors, 3)
        #   Indexing with padded neighbors -> (#vertices, #max_neighbors, 3)
        #   Can then subtract to get all edge vectors, shape (#vertices, #max_neighbors, 3)
        # TODO: potential bug: what happens here where the padded_neighbor values are -1? will just give last vertex??
        curr_edges = curr_pos.unsqueeze(1) - curr_pos[self.padded_neighbors]
        
        # Compute covariance matrices for each vertex (Si in the original ARAP paper), assuming uniform weights
        #   Have to transpose the original edges because contrary to the the paper, our edges are stored as rows, not columns 
        covariance_matrices = torch.matmul(
            self.original_edges.transpose(2, 1),  # [#vertices, 3, #max_neighbors]
            curr_edges  # [#vertices, #max_neighbors, 3]
        )  # [#vertices, 3, 3]

        # TODO: return to this with non-uniform weights at some point, maybe based on visibility criterion, difference in rendered and relative depth map, ...
        
        # Compute optimal rotations using SVD
        U, _, V = torch.svd(covariance_matrices)
        
        # Compute rotation matrices Ri = V * U^T
        rot_matrices = torch.matmul(V, U.transpose(2, 1))  # [N, 3, 3]
        
        # Handle reflection case by ensuring det(Ri) > 0
        dets = torch.linalg.det(rot_matrices)
        
        # Create reflection matrix diag(1,1,-1)
        reflection_matrix = torch.eye(3, device='cuda').unsqueeze(0)
        reflection_matrix[:, 2, 2] = -1
        
        # Apply reflection where det < 0
        rot_matrices = torch.where(
            dets.unsqueeze(-1).unsqueeze(-1) < 0,
            torch.matmul(rot_matrices, reflection_matrix),
            rot_matrices
        )
        
        # Apply rotations to all edges at once using batch matrix multiplication
        # Unsqueezing rotations along dim 1 -> (#vertices, 1, 3, 3) again so that we can broadcast along dim 1
        # Unsqueezing prev_edges along dim -1 -> (#vertices, #max_neighbors, 3, 1)
        # Then matmul will broadcast the rotations to (#vertices, #max_neighbors, 3, 3) and we can matmul to get (#vertices, #max_neighbors, 3, 1)
        # Removing the superfluous last dim gives us the rotated edges (#vertices, #max_neighbors, 3)
        rotated_edges = torch.matmul(
            rot_matrices.unsqueeze(1),
            self.original_edges.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute differences and apply precomputed mask
        edge_diff = curr_edges - rotated_edges
        # Summing over the last dim gives us the l2 norm squared for each edge, shaped (#vertices, #max_neighbors)
        squared_diff = torch.sum(edge_diff * edge_diff, dim=-1)
        # Apply mask to zero out padded entries
        masked_diff = squared_diff * self.neighbor_mask
        # Sum up all contributions
        arap_energy = masked_diff.sum()
        
        return arap_energy
        
    def compute_arap_energy_cpp(self):
        curr_pos = self.mesh_deformation(self.original_mesh_vertices)
        return ArapFunction.apply(curr_pos, 
                        self.original_mesh_vertices,
                        self.padded_neighbors,
                        self.neighbor_mask)
        
    # TODO: add more regularization terms here
    def compute_regulation(self, visibility_filter):
        arap_energy = self.compute_arap_energy_cpp()
        return arap_energy

    def build_neighbors(self, faces):
        # Load torch tensors from a file if they exist
        # TODO: hack to debug faster, remove later
        if os.path.exists("padded_neighbors.pt") and os.path.exists("neighbor_mask.pt"):
            print("Loading padded_neighbors and neighbor_mask from file")
            padded_neighbors = torch.load("padded_neighbors.pt")
            neighbor_mask = torch.load("neighbor_mask.pt")
            return padded_neighbors, neighbor_mask
        
        num_vertices = self.current_mesh_vertices.shape[0]
        
        # Create adjacency lists
        adjacency = [[] for _ in range(num_vertices)]
        
        # Each face is [v0, v1, v2]. Add edges both ways.
        for f in faces:
            v0, v1, v2 = f
            adjacency[v0].append(v1)
            adjacency[v0].append(v2)
            adjacency[v1].append(v0)
            adjacency[v1].append(v2)
            adjacency[v2].append(v0)
            adjacency[v2].append(v1)
        
        # Remove duplicates
        for i in range(num_vertices):
            adjacency[i] = list(set(adjacency[i]))
        
        # Build padded neighbor arrays
        max_neighbors = max(len(n) for n in adjacency)
        padded_neighbors = torch.full((num_vertices, max_neighbors), -1, device='cuda', dtype=torch.long)
        neighbor_mask = torch.zeros((num_vertices, max_neighbors), device='cuda', dtype=torch.bool)
        
        for i in range(num_vertices):
            n_list = adjacency[i]
            padded_neighbors[i, :len(n_list)] = torch.tensor(n_list, device='cuda', dtype=torch.long)
            neighbor_mask[i, :len(n_list)] = True

        # Save torch tensors to a file
        print("Saving padded_neighbors and neighbor_mask to file")
        torch.save(padded_neighbors, "padded_neighbors.pt")
        torch.save(neighbor_mask, "neighbor_mask.pt")
        
        return padded_neighbors, neighbor_mask


