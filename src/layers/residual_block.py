import torch
import torch.nn as nn
from .conv_block import ConvBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropblock=False, block_size=3, keep_prob=0.9):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, use_dropblock=use_dropblock,
                               block_size=block_size, keep_prob=keep_prob)
        self.conv2 = ConvBlock(out_channels, out_channels, use_dropblock=use_dropblock,
                               block_size=block_size, keep_prob=keep_prob)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out
