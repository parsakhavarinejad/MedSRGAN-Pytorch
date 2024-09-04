import torch
from torch import nn


class DiscriminatorLossFunction(nn.Module):
    def __init__(self, device: str = 'cuda'):
        super().__init__()

    def forward(lr, hr, sr):
        loss = -1 * torch.log(hr) - torch.log(1 - sr)
        return loss
