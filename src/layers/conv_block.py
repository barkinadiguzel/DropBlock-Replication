import torch.nn as nn
from .dropblock_layer import DropBlock2D

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_dropblock=False, block_size=3, keep_prob=0.9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropblock = use_dropblock
        if use_dropblock:
            self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x
