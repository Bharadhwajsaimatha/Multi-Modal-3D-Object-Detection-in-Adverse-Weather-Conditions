import torch
import torch.nn as nn
from feats_2D.img_feat2D import ImageFeatureExtractor
from projector.projector import Projector
from engine.encoder import BBox3DEncoder
from engine.decoder import BBox3DDecoder
from engine.heads import PredictionHeads

class Detector3D(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 num_queries=100, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6):
        super().__init__()
        self.image_feature_extractor = ImageFeatureExtractor()
        self.projector = Projector(fuse_with_xyz=True)

        self.encoder = BBox3DEncoder(
            input_dim=3 + 256,  # Assuming FPN features total 256
            model_dim=d_model,
            num_layers=num_encoder_layers,
            nhead=8
        )

        self.decoder = BBox3DDecoder(
            num_queries=num_queries,
            d_model=d_model,
            num_layers=num_decoder_layers,
            nhead=8
        )

        self.prediction_heads = PredictionHeads(d_model=d_model)

    def forward(self, img, pcd, intrinsics, extrinsics):
        
        img_feats = self.image_feature_extractor(img)  # List[(1, C_i, H_i, W_i)]

        fused_feats, valid_mask = self.projector(pcd, intrinsics, extrinsics, img_feats)
        pcd_valid = pcd[valid_mask]

        encoded_feats = self.encoder(fused_feats[valid_mask], pcd_valid)  # (N', D)
        decoder_out = self.decoder(encoded_feats)  # (M, D)

        pred_3D_corners, pred_2D_box = self.prediction_heads(decoder_out)  # (M, 8, 3), (M, 4)

        return {
            'pred_3D': pred_3D_corners,
            'pred_2D': pred_2D_box
        }
