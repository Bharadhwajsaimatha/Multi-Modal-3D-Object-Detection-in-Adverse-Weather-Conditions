import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from common_layers import FCNLayer,CRB2D

fasterRCNN = fasterrcnn_resnet50_fpn_v2(weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

class FeatureExtractor2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = fasterRCNN.transform
        self.backbone = fasterRCNN.backbone

    def forward(self,x):
        x, _ = self.transform(x)
        feats_2D = self.backbone(x)
        feats_2D = [feats_2D[0], feats_2D[1], feats_2D[2]]
        return feats_2D
    
