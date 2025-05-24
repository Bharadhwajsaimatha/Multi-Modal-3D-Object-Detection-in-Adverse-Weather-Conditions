import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

#ToDo: Implement the custom loss function
def custom_loss(pred_corners, mask):
    pass

def hungarian_matching(pred_corners, gt_corners):

    M, N = pred_corners.shape[0], gt_corners.shape[0]

    
    pred_flat = pred_corners.view(M, -1)
    gt_flat = gt_corners.view(N, -1)

    
    cost_matrix = torch.cdist(pred_flat, gt_flat, p=1).cpu().detach().numpy()

    # Hungarian matching
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)
    return list(zip(pred_idx, gt_idx))


def compute_total_loss(pred_corners, pred_2D, gt_corners, gt_2D, match_indices, lambda_2d=1.0):

    if len(match_indices) == 0:
        return {
            'loss_total': torch.tensor(0.0, device=pred_corners.device),
            'loss_corner': torch.tensor(0.0, device=pred_corners.device),
            'loss_2D': torch.tensor(0.0, device=pred_2D.device)
        }

    pred_matched_corners = torch.stack([pred_corners[i] for i, _ in match_indices], dim=0)  # (K, 8, 3)
    gt_matched_corners = torch.stack([gt_corners[j] for _, j in match_indices], dim=0)      # (K, 8, 3)

    pred_matched_2D = torch.stack([pred_2D[i] for i, _ in match_indices], dim=0)            # (K, 4)
    gt_matched_2D = torch.stack([gt_2D[j] for _, j in match_indices], dim=0)                # (K, 4)

    loss_corner = F.l1_loss(pred_matched_corners, gt_matched_corners, reduction='mean')
    loss_2D = F.l1_loss(pred_matched_2D, gt_matched_2D, reduction='mean')

    loss_total = loss_corner + lambda_2d * loss_2D

    return {
        'loss_total': loss_total,
        'loss_corner': loss_corner,
        'loss_2D': loss_2D
    }
