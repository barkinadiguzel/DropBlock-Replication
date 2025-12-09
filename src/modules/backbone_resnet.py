import torch.nn as nn
from ..layers.residual_block import ResidualBlock

class ResNetBackbone(nn.Module):
    def __init__(self, layers=[2,2,2,2], use_dropblock=True, block_size=3, keep_prob=0.9):
        super().__init__()
        self.layer1 = self._make_layer(64, 64, layers[0], use_dropblock, block_size, keep_prob)
        self.layer2 = self._make_layer(64, 128, layers[1], use_dropblock, block_size, keep_prob)
        self.layer3 = self._make_layer(128, 256, layers[2], use_dropblock, block_size, keep_prob)
        self.layer4 = self._make_layer(256, 512, layers[3], use_dropblock, block_size, keep_prob)

    def _make_layer(self, in_channels, out_channels, num_blocks, use_dropblock, block_size, keep_prob):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, use_dropblock, block_size, keep_prob))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
