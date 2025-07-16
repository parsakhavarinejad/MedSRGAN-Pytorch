# models/generator.py
import torch
import torch.nn as nn

class RWMAB(nn.Module):
    """
    Residual Whole Map Attention Block (RWMAB).
    This block is a building component for the Generator, inspired by RCAN.
    It applies two convolutional layers followed by a channel attention mechanism.
    """
    def __init__(self, channels: int = 64) -> None:
        """
        Initializes an RWMAB.

        Args:
            channels (int): Number of input/output channels for the convolutional layers. Defaults to 64.
        """
        super().__init__()
        # Two convolutional layers with ReLU activation
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), # Inplace operation saves memory
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        # Channel attention mechanism: 1x1 convolution followed by Sigmoid
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RWMAB.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map after convolution, attention, and residual connection.
        """
        # Apply convolutional block
        x_out = self.conv_block(x)
        # Calculate attention weights
        attention_weights = self.attention(x_out)
        # Apply attention and add residual connection
        # Element-wise multiplication of features with attention weights
        # Then add the original input (residual connection)
        x_out = torch.mul(x_out, attention_weights) + x
        return x_out

class ShortResidualConnection(nn.Module):
    """
    Short Residual Connection (SRC) block, consisting of multiple RWMABs
    followed by a 1x1 convolution and a global residual connection.
    """
    def __init__(self, channels: int = 64, num_rwmabs: int = 16) -> None:
        """
        Initializes a ShortResidualConnection block.

        Args:
            channels (int): Number of channels for the RWMABs and convolution. Defaults to 64.
            num_rwmabs (int): Number of RWMABs to stack within this block. Defaults to 16.
        """
        super().__init__()
        rwmabs = []
        for _ in range(num_rwmabs):
            rwmabs.append(RWMAB(channels))
        # Stack RWMABs and add a final 1x1 convolution
        self.src_block = nn.Sequential(
            *rwmabs,
            nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ShortResidualConnection block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map after processing through RWMABs,
                          1x1 conv, and a global residual connection.
        """
        x_processed = self.src_block(x)
        # Global residual connection: add the input to the processed output
        return x_processed + x

class Generator(nn.Module):
    """
    Generator Network for the MEDSRGAN.
    It takes a low-resolution (LR) image and upscales it to a high-resolution (HR) image.
    The architecture includes initial convolution, stacked ShortResidualConnection blocks,
    and pixel shuffle layers for upsampling.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_src_blocks: int = 4) -> None:
        """
        Initializes the Generator network.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images). Defaults to 3.
            out_channels (int): Number of output channels (e.g., 3 for RGB images). Defaults to 3.
            num_src_blocks (int): Number of ShortResidualConnection blocks to stack. Defaults to 4.
        """
        super().__init__()

        # Initial feature extraction convolution
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Stack ShortResidualConnection blocks
        src_blocks = []
        for _ in range(num_src_blocks):
            src_blocks.append(ShortResidualConnection(channels=64))
        # A final convolution after SRC blocks, before the global residual connection
        self.src_sequence = nn.Sequential(
            *src_blocks,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # Upsampling layers using PixelShuffle (sub-pixel convolution)
        # Two PixelShuffle layers for a total upscale factor of 4 (2*2)
        self.upsampler = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), # 64 * (upscale_factor^2) = 64 * 4 = 256
            nn.PixelShuffle(2), # Upscales by factor 2
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), # 64 * (upscale_factor^2) = 64 * 4 = 256
            nn.PixelShuffle(2)  # Upscales by factor 2 again
        )

        # Final convolution to map features back to image channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Generator.

        Args:
            x (torch.Tensor): Low-resolution input image tensor.

        Returns:
            torch.Tensor: Super-resolved output image tensor.
        """
        # Initial feature extraction
        x_initial_features = self.initial_conv(x)

        # Process through SRC blocks and apply global residual connection
        x_src_output = self.src_sequence(x_initial_features)
        # Global residual connection: add initial features to the output of SRC sequence
        x_residual_output = x_initial_features + x_src_output

        # Upsample the features
        x_upsampled = self.upsampler(x_residual_output)

        # Final convolution to get the super-resolved image
        sr_image = self.final_conv(x_upsampled)

        return sr_image
