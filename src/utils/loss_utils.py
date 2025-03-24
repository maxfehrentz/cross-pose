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
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import KDTree

def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mask is not None:
        ssim_map = ssim_map[mask]
        size_average = True
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# Taken from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        print(f"shape of input: {input.shape}")
        print(f"shape of target: {target.shape}")
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        print(f"shape of input after repeat: {input.shape}")
        print(f"shape of target after repeat: {target.shape}")
        mean = self.mean
        std = self.std
        input = (input-mean) / std
        target = (target-mean) / std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class GaussianAppearanceRegularizer:
    def __init__(self, gaussian_model, k_neighbors=250):
        self.gaussians = gaussian_model
        self.k = k_neighbors
        
        # Build KD-tree and compute initial distances
        self.neighbors = self._compute_neighbors()
        self.initial_distances = self._compute_pairwise_distances()
        
    def _compute_neighbors(self):
        """Get K nearest neighbors for each Gaussian using their 3D positions"""
        # Get Gaussian centers using get_xyz property from MeshAwareGaussianModel
        positions = self.gaussians.get_xyz
        
        # Use CPU for KD-tree building
        positions_np = positions.detach().cpu().numpy()
        tree = KDTree(positions_np)
        
        # Find k+1 nearest neighbors (first one is the point itself)
        _, indices = tree.query(positions_np, k=self.k + 1)
        
        # Remove self-reference (first index)
        return torch.from_numpy(indices[:, 1:]).to(positions.device)
    
    def _compute_pairwise_distances(self):
        """Compute pairwise distances between each Gaussian and its neighbors' appearance features"""
        # Get all appearance features from MeshAwareGaussianModel
        features_dc = self.gaussians._features_dc
        features_rest = self.gaussians._features_rest
        features_dc = features_dc.reshape(features_dc.shape[0], -1)
        features_rest = features_rest.reshape(features_rest.shape[0], -1)
        # all_features = torch.cat([features_dc, features_rest], dim=-1)  # [N, D]
        all_features = features_dc
        
        # Get features for all neighbors at once [N, K, D]
        neighbor_features = all_features[self.neighbors]
        
        # Compute distances for all Gaussians to their neighbors at once
        # [N, 1, D] - [N, K, D] -> [N, K]
        pairwise_distances = torch.norm(
            all_features.unsqueeze(1) - neighbor_features, 
            p=2, 
            dim=-1
        )
        
        return pairwise_distances
    
    def compute_regularization_loss(self):
        """Compute how much relative appearances have changed from initial state"""
        current_distances = self._compute_pairwise_distances()  # [N, K]
        
        # Normalize all distances at once
        init_dists_norm = F.normalize(self.initial_distances, p=2, dim=1)
        curr_dists_norm = F.normalize(current_distances, p=2, dim=1)
        
        # Compute loss for all Gaussians at once
        loss = F.l1_loss(curr_dists_norm, init_dists_norm)
        
        return loss

