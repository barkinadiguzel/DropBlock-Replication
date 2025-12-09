import torch.nn as nn
import torch.nn.functional as F

class FeatureProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, target_size):
        resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]
        return resized
