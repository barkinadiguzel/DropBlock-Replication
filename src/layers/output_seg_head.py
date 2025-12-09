import torch.nn as nn
from .conv_block import ConvBlock

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
