import torch
import torch.nn as nn
import math

class PositionalEncoding3D(nn.Module):
    def __init__(self, num_feats=32):
        super().__init__()
        self.num_feats = num_feats
        self.scale = 2 * math.pi

    def forward(self, coords):
        
        coords = coords * self.scale  # (N, 3)

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=coords.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_feats)  # (num_feats,)

        pos = []
        for i in range(3):  # x, y, z
            feat = coords[:, i].unsqueeze(1) / dim_t  # (N, num_feats)
            pos.append(torch.stack((feat.sin(), feat.cos()), dim=2).flatten(1))  # (N, 2*num_feats)

        pos_enc = torch.cat(pos, dim=1)  # (N, 3 * 2 * num_feats)
        return pos_enc
