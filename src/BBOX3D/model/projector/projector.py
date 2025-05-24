import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry_utils import project_pcd_to_img

class Projector(nn.Module):
    def __init__(
            self,
            fuse_with_xyz = True
    ):
        super().__init__()
        self.fuse_with_xyz = fuse_with_xyz

    def forward(self, points_3D, intrinsics, extrinsics, feats_2D):

        uv, valid_mask = project_pcd_to_img(points_3D, intrinsics, extrinsics)

        fused_feats = []

        for feat in feats_2D:
            print(f"Feature shape : {feat.shape}")
            B,C,H,W = feat.shape

            feat = feat[0]

            u_norm = (uv[:,0]/W) * 2 -1
            v_norm = (uv[:,1]/H) * 2 -1
            grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(1).unsqueeze(1)
            feat = feat.unsqueeze(0)  # (1, C, H, W)
            sampled = F.grid_sample(feat, grid, align_corners=True, mode='bilinear', padding_mode='zeros')  # (1, C, N, 1, 1)
            sampled = sampled.squeeze(-1).squeeze(-1).squeeze(0).T

            fused_feats.append(sampled)

        feat_cat = torch.cat(fused_feats, dim= 1)
        if self.fuse_with_xyz:
            feat_cat = torch.cat([points_3D, feat_cat], dim=1) 

        return feat_cat, valid_mask

