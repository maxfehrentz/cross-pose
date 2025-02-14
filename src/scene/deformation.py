import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import KDTree
from src.utils.general_utils import build_rotation
from math import ceil


class ExplicitDeformation(nn.Module):
    def __init__(self):
        super().__init__()
        self.means_def = torch.nn.Parameter(torch.zeros(0, 3))
        self.rot_def = torch.nn.Parameter(torch.zeros(0, 4))
        self.cuda()
        self.means_cache, self.rot_cache = [None,None], [None,None]
        self.neighbour_dists = None
        self.neighbours = None
        self.neighbour_weights = None

    def forward(self, means, scales, rot, init=False):
        means = means + self.means_def
        rot = rot + self.rot_def
        if init:
            self.means_cache[1] = means.detach()
            self.rot_cache[1] = rot.detach()
        self.means_cache[0] = means
        self.rot_cache[0] = rot
        return means, scales, rot

    def get_mean_def(self, means):
        return self.means_def

    def get_deformed_means(self, means):
        return means + self.means_def

    def add_gaussians(self, n_new_gaussians: int, means):
        self.means_def = torch.nn.Parameter(torch.cat((self.means_def.data, torch.zeros(n_new_gaussians, 3, device='cuda')), dim=0))
        self.rot_def = torch.nn.Parameter(torch.cat((self.rot_def.data, torch.zeros(n_new_gaussians, 4, device='cuda')), dim=0))
        self.update_topology(means) if self.neighbours is not None else self.init_topology(means)

    def replace(self, param_list, means, reinit):
        self.means_def = param_list[0]
        self.rot_def = param_list[1]
        if reinit or self.neighbours is None:
            self.init_topology(means)
        else:
            self.update_topology(means)

    @torch.no_grad()
    def init_topology(self, means, k=20):
        """
            init topology finding k-nearest neighbours for each point to regularize deformation
        """
        tree = KDTree(means.detach().cpu().numpy())
        neighbour_dists, neighbours = tree.query(means.detach().cpu().numpy(), k=k)#, eps=0.1)
        self.neighbour_dists = torch.tensor(neighbour_dists[:, 1:], device="cuda")
        self.neighbours = torch.tensor(neighbours[:, 1:], device="cuda")
        self.neighbour_weights = torch.exp(-50*(self.neighbour_dists))
        self.means_cache, self.rot_cache = [None, None], [None, None]

    @torch.no_grad()
    def update_topology(self, means, k=20):
        # assume that new points are added to the end of the tensor and none are removed
        new_means = means[self.neighbours.shape[0]:]
        if new_means.shape[0] > 0:
            tree = KDTree(means.detach().cpu().numpy())
            neighbour_dists, neighbours = tree.query(new_means.detach().cpu().numpy(), k=k)#, eps=0.1)
            self.neighbour_dists = torch.cat((self.neighbour_dists, torch.tensor(neighbour_dists[:, 1:], device="cuda")), dim=0)
            self.neighbours = torch.cat((self.neighbours, torch.tensor(neighbours[:, 1:], device="cuda")), dim=0)
            self.neighbour_weights = torch.exp(-10.0*(self.neighbour_dists))
            self.means_cache, self.rot_cache = [None, None], [None, None]

    def reg_loss(self, visibility_filter):
        if self.means_cache[1] is None:
            l = torch.zeros(1, device="cuda").squeeze()
            return l, l, l, l
        prev_rot = build_rotation(self.rot_cache[1])
        cur_rot = build_rotation(self.rot_cache[0])
        rel_rot = prev_rot @ cur_rot.transpose(1,2)
        cur_offset = self.means_cache[0][self.neighbours] - self.means_cache[0][:,None]
        last_offset = self.means_cache[1][self.neighbours] - self.means_cache[1][:,None]
        rot_offset = last_offset - cur_offset
        l_rigid = (torch.linalg.norm(rot_offset, dim=-1) * self.neighbour_weights).mean()

        l_rot = torch.sqrt(((rel_rot[:,None]-rel_rot[self.neighbours]) ** 2).sum(-1).sum(-1) * self.neighbour_weights + 1e-20).mean()
        curr_offset_mag = torch.linalg.norm(cur_offset, dim=-1)
        l_iso = (torch.abs(curr_offset_mag - self.neighbour_dists)*self.neighbour_weights).mean()
        ids = ~visibility_filter
        l_visible = self.means_def[ids].abs().sum() / (ids.sum()+1)
        return l_rigid, l_rot, l_iso, l_visible

    def get_new_params(self, shape):
        new_shape = shape[0]
        new_means = torch.zeros(new_shape, 3, device='cuda')
        new_rot = torch.zeros(new_shape, 4, device='cuda')
        return new_means, new_rot

    @torch.no_grad()
    def init_from_flow(self, deformation, weights):
        weight_sum = weights.sum(-1)
        deformation = (weights[..., None] * deformation).sum(1) / weight_sum[..., None]
        deformation[weight_sum < 0.1] = 0.0
        self.means_def += deformation.clamp(-0.01, 0.01)


class ExplicitSparseDeformation(ExplicitDeformation):
    def __init__(self, subsample: int=64):
        super().__init__()
        self.anchor_ids = None
        self.subsample = subsample
        self.control_pts = None

    def interpolate(self):
        """
            fast approximation to Gaussian Kernel Interpolation using pre-computed weights of k-most important
            control points
        """
        assert self.means_def.shape[0] > self.neighbours.max()
        means_def = (self.neighbour_weights[...,None] * self.means_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[...,None]
        rot_def = (self.neighbour_weights[...,None] * self.rot_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[...,None]
        return means_def, rot_def

    def forward(self, means, scales, rot, init=False):
        means_def, rot_def = self.interpolate()
        means_deformed = means + means_def
        rot_deformed = rot + rot_def
        if init:
            self.means_cache[1] = (means[self.anchor_ids] + self.means_def).detach()
            self.rot_cache[1] = (rot[self.anchor_ids] + self.rot_def).detach()
        self.means_cache[0] = means[self.anchor_ids] + self.means_def
        self.rot_cache[0] = rot[self.anchor_ids] + self.rot_def
        return means_deformed, scales, rot_deformed

    def get_mean_def(self, means):
        return self.interpolate()[0]

    def get_deformed_means(self, means):
        return means + self.interpolate()[0]

    def get_new_params(self, shape):
        new_shape = ceil(shape[0]/self.subsample)
        new_means = torch.zeros(new_shape, 3, device='cuda')
        new_rot = torch.zeros(new_shape, 4, device='cuda')
        return new_means, new_rot

    def add_gaussians(self, n_new_gaussians: int, means: torch.Tensor):
        means_np = means.detach().cpu().numpy()
        if self.neighbours is not None:
            anchor_ids = np.random.permutation(np.arange(self.neighbours.shape[0], means_np.shape[0]))[::self.subsample]
            self.anchor_ids = torch.cat((self.anchor_ids, torch.tensor(anchor_ids, device='cuda')))
        else:
            anchor_ids = np.random.permutation(np.arange(means_np.shape[0]))[::self.subsample]
            self.anchor_ids = torch.tensor(anchor_ids, device='cuda')

        self.means_def = torch.nn.Parameter(torch.cat((self.means_def.data, torch.zeros(self.anchor_ids.shape[0], 3, device='cuda')), dim=0))
        self.rot_def = torch.nn.Parameter(torch.cat((self.rot_def.data, torch.zeros(self.anchor_ids.shape[0], 4, device='cuda')), dim=0))
        self.update_topology(means, self.anchor_ids) if self.neighbours is not None else self.init_topology(means, self.anchor_ids)

    def replace(self, param_list, means, reinit):
        self.means_def = param_list[0]
        self.rot_def = param_list[1]
        if reinit or self.neighbours is None:
            anchor_ids = np.random.permutation(np.arange(means.shape[0]))[::self.subsample]
            self.anchor_ids = torch.tensor(anchor_ids, device='cuda')
            self.init_topology(means, self.anchor_ids)
        else:
            anchor_ids = np.random.permutation(np.arange(self.neighbours.shape[0], means.shape[0]))[::self.subsample]
            self.anchor_ids = torch.cat((self.anchor_ids, torch.tensor(anchor_ids, device='cuda')))
            self.update_topology(means, self.anchor_ids)

    @torch.no_grad()
    def init_topology(self, means, anchor_ids, classes=None, k=4, eps=0.0):
        """
            select a subset of points from the collection of Gaussians as control points
            init topology between control points
            init Gaussian kernel weights for all Gaussians for fast interpolation
        """
        means_np = means.detach().cpu().numpy()
        anchor_ids_np = anchor_ids.cpu().numpy()
        tree = KDTree(means_np[anchor_ids_np])
        neighbour_dists, neighbours = tree.query(means_np, k=k, eps=eps)

        self.control_pts = means[anchor_ids].detach()
        self.neighbour_dists = torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)
        self.neighbours = torch.tensor(neighbours, device="cuda")
        self.neighbour_weights = torch.exp(-4.5*(self.neighbour_dists))
        self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)
        self.means_cache, self.rot_cache = [None, None], [None, None]

    @torch.no_grad()
    def update_topology(self, means, anchor_ids, classes=None, k=4, eps=0.0):
        # assume that new points are added to the end of the tensor and none are removed
        means_np = means.detach().cpu().numpy()
        new_means = means_np[self.neighbours.shape[0]:]
        anchor_ids_np = anchor_ids.cpu().numpy()
        if new_means.shape[0] > 0:
            tree = KDTree(means_np[anchor_ids_np])
            neighbour_dists, neighbours = tree.query(new_means, k=k, eps=eps)
            self.control_pts = means[anchor_ids].detach()
            self.neighbour_dists = torch.cat((self.neighbour_dists, torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)), dim=0)
            self.neighbours = torch.cat((self.neighbours, torch.tensor(neighbours, device="cuda")), dim=0)
            self.neighbour_weights = torch.exp(-4.5*(self.neighbour_dists))
            self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)
            self.means_cache, self.rot_cache = [None, None], [None, None]

    def reg_loss(self, visibility_filter):
        if self.means_cache[1] is None:
            l = torch.zeros(1, device="cuda").squeeze()
            return l, l, l, l
        prev_rot = build_rotation(self.rot_cache[1])
        cur_rot = build_rotation(self.rot_cache[0])
        rel_rot = prev_rot @ cur_rot.transpose(1,2)
        cur_offset = self.means_cache[0][self.neighbours[self.anchor_ids]] - self.means_cache[0][:,None]
        last_offset = self.means_cache[1][self.neighbours[self.anchor_ids]] - self.means_cache[1][:,None]
        rot_offset = last_offset - cur_offset
        l_rigid = (torch.linalg.norm(rot_offset, dim=-1) * self.neighbour_weights[self.anchor_ids]).mean()

        l_rot = torch.sqrt(((rel_rot[:,None]-rel_rot[self.neighbours[self.anchor_ids]]) ** 2).sum(-1).sum(-1) * self.neighbour_weights[self.anchor_ids] + 1e-20).mean()

        curr_offset_mag = torch.linalg.norm(cur_offset, dim=-1)
        l_iso = (torch.abs(curr_offset_mag - self.neighbour_dists[self.anchor_ids])*self.neighbour_weights[self.anchor_ids]).mean()
        ids = ~visibility_filter[self.anchor_ids]
        l_visible = self.means_def[ids].abs().sum() / (ids.sum()+1) # avoid nan for mean() if empty slice
        return l_rigid, l_rot, l_iso, l_visible

    @torch.no_grad()
    def init_from_flow(self, deformation, weights):
        weight_sum = weights.sum(-1)
        deformation = (weights[..., None] * deformation).sum(1) / weight_sum[..., None]
        deformation[weight_sum < 0.1] = 0.0
        self.means_def += deformation[self.anchor_ids].clamp(-0.01, 0.01).float()


class MeshSparseDeformation(nn.Module):
    def __init__(self, vertices, visible_vertices=None, subsample: int=32, k: int=25):
        super().__init__()
        
        if visible_vertices is None:
            # Initialize control vertices
            n_vertices = vertices.shape[0]
            control_ids = np.random.permutation(np.arange(n_vertices))[::subsample]
        else:
            control_ids = visible_vertices[::subsample]
            self.visible_vertices = visible_vertices

        self.control_ids = torch.tensor(control_ids, device='cuda')
        
        # Initialize deformation parameters for control vertices
        self.control_def = torch.nn.Parameter(torch.zeros(len(self.control_ids), 3, device='cuda'))
        
        # Build KDTree from control vertices
        tree = KDTree(vertices[self.control_ids].detach().cpu().numpy())

        # Yields the k-nearest control points for each vertex, shape [n_vertices, k]
        neighbour_dists, neighbours = tree.query(vertices.detach().cpu().numpy(), k=k)

        # Store distances and weights
        self.neighbour_dists = torch.tensor(neighbour_dists, device="cuda", dtype=torch.float32)

        self.neighbours = torch.tensor(neighbours, device="cuda")
        self.neighbour_weights = torch.exp(-4.5*(self.neighbour_dists))

        # Yields a sum of shape [n_vertices, 1]
        self.neighbour_weights_sum = self.neighbour_weights.sum(dim=-1).clamp(1e-2)

        # Adding cache to store previous and current deformed vertices
        self.mean_cache = [None, None]

    def interpolate(self):
        """
        Fast approximation using pre-computed weights of k-most important neighbors
        """
        assert self.control_def.shape[0] > self.neighbours.max()
        # neighbors is [n_vertices, k_neighbors], indexing into control_def -> [n_vertices, k_neighbors, 3]
        # self.neighbour_weights is [n_vertices, k_neighbors], adding a dimension for broadcasting
        # Then summing over dim=1 -> [n_vertices, 1, 3]
        # Dividing by self.neighbour_weights_sum of shape [n_vertices, 1] with added last dim for broadcast
        # Gives final shape [n_vertices, 3]
        vertices_def = (self.neighbour_weights[...,None] * self.control_def[self.neighbours]).sum(dim=1) / self.neighbour_weights_sum[...,None]
        return vertices_def

    def forward(self, vertices):        
        vertices_def = self.interpolate()
        vertices_deformed = vertices + vertices_def
        if self.mean_cache[0] is not None:
            # Detach cache to avoid backprop through cache
            self.mean_cache[1] = self.mean_cache[0].detach()
        self.mean_cache[0] = vertices_deformed
        return vertices_deformed
    
    def compute_cotangent_weights(self, vertices, faces):
        """
        Compute cotangent weights for mesh edges.
        For each edge (i,j), weight_ij = (cot α + cot β)/2
        where α and β are the angles opposite to the edge in the two adjacent triangles.
        """
        
        # For each face, compute all three edges; shaped [n_faces, 3]
        e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        e2 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
        e3 = vertices[faces[:, 0]] - vertices[faces[:, 2]]
        
        # Compute angles using dot product; have to use minus because the vectors should both point outwards but the way we compute the edges
        #  means that one points inwards and the other outwards
        # Result is shaped [n_faces, 3]
        unnormalized_cos_angles = torch.stack([
            torch.sum(-e3 * e1, dim=1),
            torch.sum(-e1 * e2, dim=1),
            torch.sum(-e2 * e3, dim=1) 
        ], dim=1)
        
        # Compute all lengths
        len_e1 = torch.norm(e1, dim=1)
        len_e2 = torch.norm(e2, dim=1)
        len_e3 = torch.norm(e3, dim=1)

        # Have to normalize, since cos(angle) = dot product / (norm(e1) * norm(e2))
        cos_angles = torch.stack([
            unnormalized_cos_angles[:, 0] / (len_e3 * len_e1),
            unnormalized_cos_angles[:, 1] / (len_e1 * len_e2),
            unnormalized_cos_angles[:, 2] / (len_e2 * len_e3) 
        ], dim=1)

        # Clamping for numerical stability
        cos_angles = torch.clamp(cos_angles, -0.999999, 0.999999)

        # Computing cotangents, since cot(angle) = cos(angle) / sin(angle) = 1 / tan(angle) = 1 / tan(arccos(cos(angle)))
        cotangents = 1.0 / torch.tan(torch.acos(cos_angles))
        
        # Create matrix of weights
        n_vertices = vertices.shape[0]
        weights = torch.zeros((n_vertices, n_vertices), device=vertices.device)
        
        # Weights are 0.5 * cotangent_alpha + 0.5 * cotangent_beta where alpha and beta are the angles opposite to the edge
        for i in range(3):
            j = (i + 1) % 3
            # Get the vertex indices for the edge
            idx_i = faces[:, i]
            idx_j = faces[:, j]
            # Add the cotangent weights for the edge
            weights[idx_i, idx_j] += 0.5 * cotangents[:, (j - 1) % 3]
            weights[idx_j, idx_i] += 0.5 * cotangents[:, (j - 1) % 3]
            
        return weights

    def compute_rigid_loc_loss(self):
        if self.mean_cache[1] is None:
            return torch.tensor(0, device="cuda")

        cur_offset = self.mean_cache[0][self.neighbours] - self.mean_cache[0][:,None]
        last_offset = self.mean_cache[1][self.neighbours] - self.mean_cache[1][:,None]
        delta = last_offset - cur_offset
        l_rigid = (torch.linalg.norm(delta, dim=-1) * self.neighbour_weights).mean()
        return l_rigid
    
    def compute_rigid_rot_loss(self):
        # TODO: implement
        return torch.tensor(0, device="cuda")
    
    def compute_iso_loss(self):
        # if self.mean_cache[1] is None:
        #     return torch.tensor(0, device="cuda")
        
        # cur_offset = self.mean_cache[0][self.neighbours] - self.mean_cache[0][:,None]
        # curr_offset_mag = torch.linalg.norm(cur_offset, dim=-1)
        
        # # Normalize by mean neighbor distance to get comparable scale to rigid_loc_loss
        # mean_neighbor_dist = self.neighbour_dists.mean()
        # l_iso = (torch.abs(curr_offset_mag - self.neighbour_dists) / mean_neighbor_dist * self.neighbour_weights).mean()
        
        # return l_iso

        return torch.tensor(0, device="cuda")
        