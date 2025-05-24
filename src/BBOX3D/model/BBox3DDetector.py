import torch
import pytorch_lightning as pl
from detector3D import Detector3D
from loss.loss import hungarian_matching, compute_total_loss
from loss.metrics import eval_batch

class Lit3DDetector(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = Detector3D()
        self.lr = lr

    def forward(self, batch):
        img = batch['img'][0].unsqueeze(0)          # (1, 3, H, W)
        pcd = batch['pcd'][0]                       # (N, 3)
        intrinsics = batch['intrinsics'][0]
        extrinsics = batch['extrinsics'][0]
        return self.model(img, pcd, intrinsics, extrinsics)

    def training_step(self, batch, batch_idx):
        gt_corners = batch['bbox3D'][0]             # (N, 8, 3)
        gt_2D = batch['bbox2D'][0]                  # (N, 4)

        outputs = self(batch)
        pred_corners = outputs['pred_3D']           # (M, 8, 3)
        pred_2D = outputs['pred_2D']                # (M, 4)

        match_indices = hungarian_matching(pred_corners, gt_corners)
        loss_dict = compute_total_loss(pred_corners, pred_2D, gt_corners, gt_2D, match_indices)

        self.log('train/loss_total', loss_dict['loss_total'], on_step=True, on_epoch=True)
        self.log('train/loss_corner', loss_dict['loss_corner'], on_step=True, on_epoch=True)
        self.log('train/loss_2D', loss_dict['loss_2D'], on_step=True, on_epoch=True)

        return loss_dict['loss_total']

    def validation_step(self, batch, batch_idx):
        gt_corners = batch['bbox3D'][0]
        gt_2D = batch['bbox2D'][0]

        outputs = self(batch)
        pred_corners = outputs['pred_3D']
        pred_2D = outputs['pred_2D']

        match_indices = hungarian_matching(pred_corners, gt_corners)
        loss_dict = compute_total_loss(pred_corners, pred_2D, gt_corners, gt_2D, match_indices)
        metrics = eval_batch(pred_corners, pred_2D, gt_corners, gt_2D, match_indices)

        self.log('val/loss_total', loss_dict['loss_total'], on_step=False, on_epoch=True)
        self.log('val/loss_corner', loss_dict['loss_corner'], on_step=False, on_epoch=True)
        self.log('val/loss_2D', loss_dict['loss_2D'], on_step=False, on_epoch=True)
        self.log('val/IoU_3D', metrics['mean_iou_3d'], on_step=False, on_epoch=True)
        self.log('val/IoU_2D', metrics['mean_iou_2d'], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
