import torch
import torch.nn as nn

class RWMAB(nn.Module):
    """Residual Whole Map Attention Block (RWMAB)."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.conv_block(x)
        attention_weights = self.attention(x_out)
        x_out = torch.mul(x_out, attention_weights) + x
        return x_out

class ShortResidualConnection(nn.Module):
    """Short Residual Connection (SRC) block."""
    def __init__(self, channels: int = 64, num_rwmabs: int = 16):
        super().__init__()
        rwmabs = [RWMAB(channels) for _ in range(num_rwmabs)]
        self.src_block = nn.Sequential(
            *rwmabs,
            nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.src_block(x) + x

class Generator(nn.Module):
    """Generator Network for the MEDSRGAN."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_src_blocks: int = 4):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        src_blocks = [ShortResidualConnection(channels=64) for _ in range(num_src_blocks)]
        self.src_sequence = nn.Sequential(
            *src_blocks,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.upsampler = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_initial_features = self.initial_conv(x)
        x_src_output = self.src_sequence(x_initial_features)
        x_residual_output = x_initial_features + x_src_output
        x_upsampled = self.upsampler(x_residual_output)
        sr_image = self.final_conv(x_upsampled)
        return sr_image