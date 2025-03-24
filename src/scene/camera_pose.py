import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.graphics_utils import getProjectionMatrix, se3_to_SE3


# Taken from iComCa (https://github.com/YuanSun-XJTU/iComMa), inspired by iNeRF (https://github.com/salykovaa/inerf)
class CameraPose(nn.Module):
    def __init__(self,start_pose_w2c, FoVx, FoVy, image_width, image_height,
             trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0,
             ):
        super(CameraPose, self).__init__()

        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 3.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.cov_offset = 0
        
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        
        self.forward(start_pose_w2c)
    
    def forward(self, start_pose_w2c):
        deltaT=se3_to_SE3(self.w,self.v)
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        self.update()
    
    def current_campose_c2w(self):
        return self.pose_w2c.inverse().clone().cpu().detach().numpy()
    def update(self):
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

