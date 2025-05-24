import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding3D

class BBox3DEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            model_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            use_positional_encoding=True
        ):
        super().__init__()
        self.use_pos = use_positional_encoding
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding3D(num_feats=model_dim // 6) if use_positional_encoding else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, fused_feats, coords):
        x = self.input_proj(fused_feats)  # (N, model_dim)

        if self.use_pos:
            pos_enc = self.pos_encoder(coords)  # (N, model_dim)
            x = x + pos_enc

        encoded = self.transformer_encoder(x.unsqueeze(0))  # (1, N, model_dim)
        return encoded.squeeze(0)
