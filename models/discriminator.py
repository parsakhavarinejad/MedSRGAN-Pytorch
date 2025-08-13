import torch
import torch.nn as nn

class DBlock(nn.Module):
    """Discriminator Block: A conv block with optional BN and LeakyReLU."""
    def __init__(self, input_channels: int, output_channels: int, stride: int = 2, bn: bool = True, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=output_channels) if bn else None
        self.leakyr = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.leakyr(x)
        return x

class Discriminator(nn.Module):
    """Discriminator Network for the MEDSRGAN."""
    def __init__(self):
        super().__init__()
        self.block_1_sr = nn.Sequential(
            DBlock(1, 64, stride=1, bn=False),
            DBlock(64, 64, stride=2)
        )
        self.block_2_sr = nn.Sequential(
            DBlock(64, 128, stride=1),
            DBlock(128, 128, stride=2)
        )
        self.block_1_lr = nn.Sequential(
            DBlock(1, 64, stride=1, bn=False),
            DBlock(64, 128, stride=1, bn=True)
        )
        self.block_1 = nn.Sequential(
            DBlock(128, 256, stride=1),
            DBlock(256, 256, stride=2)
        )
        self.block_2 = nn.Sequential(
            DBlock(256, 512, stride=1),
            DBlock(512, 512, stride=2)
        )
        self.block_3 = nn.Sequential(
            DBlock(512, 1024, stride=1),
            DBlock(1024, 1024, stride=2)
        )
        self.final = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, lr_image: torch.Tensor, sr_or_hr_image: torch.Tensor) -> tuple:
        x_1_sr = self.block_1_sr(sr_or_hr_image)
        x_2_sr = self.block_2_sr(x_1_sr)
        y_1_lr = self.block_1_lr(lr_image)
        xy = torch.add(x_2_sr, y_1_lr)
        xy_1 = self.block_1(xy)
        xy_2 = self.block_2(xy_1)
        xy_3 = self.block_3(xy_2)
        xy_3_flat = xy_3.view(xy_3.size(0), -1)
        final_output_score = self.final(xy_3_flat)

        return (x_1_sr, x_2_sr, xy_1, xy_2, xy_3, final_output_score)