import torch.nn as nn
import torch.nn.functional as F
from .conv_block import ConvBlock

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_conv = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        self.output_conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, lateral_feat):
        x = F.interpolate(x, size=lateral_feat.shape[2:], mode='nearest')
        x = x + self.lateral_conv(lateral_feat)
        x = self.output_conv(x)
        return x
