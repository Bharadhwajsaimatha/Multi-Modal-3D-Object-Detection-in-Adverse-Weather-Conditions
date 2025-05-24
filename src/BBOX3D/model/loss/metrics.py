import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def box3d_iou(pred_box, gt_box):
    try:
        pred_hull = ConvexHull(pred_box)
        gt_hull = ConvexHull(gt_box)
        inter = ConvexHull(np.concatenate([pred_box, gt_box])).volume
        union = pred_hull.volume + gt_hull.volume - inter
        iou = inter / union
    except:
        iou = 0.0
    return iou

def bbox2d_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def eval_batch(pred_corners, pred_2D, gt_corners, gt_2D, matches):
    iou3d_list = []
    iou2d_list = []
    for i, j in matches:
        pred_3d = pred_corners[i].cpu().numpy()
        gt_3d = gt_corners[j].cpu().numpy()
        pred_2d = pred_2D[i].cpu().numpy()
        gt_2d = gt_2D[j].cpu().numpy()
        iou3d = box3d_iou(pred_3d, gt_3d)
        iou2d = bbox2d_iou(pred_2d, gt_2d)
        iou3d_list.append(iou3d)
        iou2d_list.append(iou2d)
    return {
        'mean_iou_3d': float(np.mean(iou3d_list)) if iou3d_list else 0.0,
        'mean_iou_2d': float(np.mean(iou2d_list)) if iou2d_list else 0.0
    }
