import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    def __init__(self, block_size, keep_prob=0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand_like(x) < gamma).float()
            block_mask = self._compute_block_mask(mask)
            # Apply mask
            out = x * block_mask
            # Normalize
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_gamma(self, x):
        feat_size = x.size(2)  
        return ((1 - self.keep_prob) / (self.block_size ** 2) *
                (feat_size ** 2) / ((feat_size - self.block_size + 1) ** 2))

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        block_mask = 1 - block_mask
        return block_mask
