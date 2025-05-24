import torch.nn as nn
from feature2D import FeatureExtractor2D

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.featextractor = FeatureExtractor2D()
        self.featextractor.train(False)
        for name,param in self.featextractor.named_parameters():
            print(f"Disabling gradients for {name} parameter in 2D Feature Extractor")
            param.requires_grad = False

    def forward(self,x):
        return self.featextractor(x)