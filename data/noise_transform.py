import torch


class NoiseTransform:
    def __init__(self, std=0.25, mode='bicubic'):
        self.mode = mode
        self.std = std

    def __call__(self, tensor):
        if self.mode == 'gaussian':
            return tensor + self.std * torch.randn(tensor.shape[1:])

        else:
            return F.interpolate(tensor.unsqueeze(0), scale_factor=1, mode=self.mode, align_corners=False).squeeze(0)