import torch.nn as nn
from ..modules.backbone_resnet import ResNetBackbone
from ..layers.fpn_block import FPNBlock
from ..layers/output_seg_head import SegmentationHead

class DropBlockModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder: extracts features from input using ResNet-style backbone
        self.backbone = ResNetBackbone(use_dropblock=True)
        # Decoder: combines features like a simple FPN
        self.fpn = FPNBlock(512, 256)
        # Segmentation head: produces per-pixel class predictions
        self.seg_head = SegmentationHead(256, num_classes)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        # Combine features through FPN
        fpn_feat = self.fpn(features, features)  
        # Produce output with segmentation head
        out = self.seg_head(fpn_feat)
        return out
