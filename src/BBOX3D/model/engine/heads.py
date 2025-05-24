import torch
import torch.nn as nn

class PredictionHeads(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.box3D_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 24)  # 8 corners * 3 (x, y, z)
        )

        self.box2D_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)  # [x1, y1, x2, y2]
        )

    def forward(self, decoder_out):
        """
        Args:
            decoder_out: (M, D)

        Returns:
            box3D_pred: (M, 8, 3)
            box2D_pred: (M, 4)
        """
        pred_3D = self.box3D_head(decoder_out).reshape(-1, 8, 3)  # (M, 8, 3)
        pred_2D = self.box2D_head(decoder_out)  # (M, 4)
        return pred_3D, pred_2D
