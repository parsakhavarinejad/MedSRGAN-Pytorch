import torch
import torch.nn as nn
import torch.nn.functional as F

class DBlock(nn.Module):
    """
    Discriminator Block: A convolutional block with optional Batch Normalization and LeakyReLU activation.
    """
    def __init__(self, input_channels: int, output_channels: int, stride: int = 2, bn: bool = True, padding: int = 1) -> None:
        """
        Initializes a DBlock.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layer. Defaults to 2.
            bn (bool): Whether to apply Batch Normalization. Defaults to True.
            padding (int): Padding for the convolutional layer. Defaults to 1.
        """
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=output_channels) if bn else None
        self.leakyr = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, optional batch norm, and activation.
        """
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.leakyr(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator Network for the MEDSRGAN.
    It takes both low-resolution (LR) and high-resolution (HR) images as input
    and outputs a probability score indicating whether the HR image is real or fake,
    along with intermediate feature maps for feature matching loss.
    """
    def __init__(self):
        """
        Initializes the Discriminator network.
        """
        super().__init__()
        self.block_1_sr = nn.Sequential(
            DBlock(3, 64, stride=1, bn=False),  # Initial block, no BN on first layer
            DBlock(64, 64, stride=2)            # Downsampling
        )
        self.block_2_sr = nn.Sequential(
            DBlock(64, 128, stride=1),
            DBlock(128, 128, stride=2)          # Downsampling
        )

        self.block_1_lr = nn.Sequential(
            DBlock(3, 64, stride=1, bn=False),  # No BN on first layer
            DBlock(64, 128, stride=1, bn=True)  # No downsampling here, just feature extraction
        )

        # Combined blocks after concatenation/addition of SR/HR and LR features
        self.block_1 = nn.Sequential(
            DBlock(128, 256, stride=1),
            DBlock(256, 256, stride=2)          # Downsampling
        )
        self.block_2 = nn.Sequential(
            DBlock(256, 512, stride=1),
            DBlock(512, 512, stride=2)          # Downsampling
        )
        self.block_3 = nn.Sequential(
            DBlock(512, 1024, stride=1),
            DBlock(1024, 1024, stride=2)        # Downsampling
        )

        # Final fully connected layers for classification
        # The input size (1024*8*8) depends on the input image size (256x256)
        # and the number of downsampling steps.
        # 256 -> 128 (block_1_sr/block_2_sr) -> 64 (block_1) -> 32 (block_2) -> 16 (block_3) -> 8 (final stride 2 in block_3)
        self.final = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid() # Output a probability score between 0 and 1
        )

    def forward(self, lr_image: torch.Tensor, sr_or_hr_image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for the Discriminator.

        Args:
            lr_image (torch.Tensor): Low-resolution input image.
            sr_or_hr_image (torch.Tensor): Super-resolved or high-resolution image.

        Returns:
            tuple: A tuple containing intermediate feature maps and the final sigmoid output.
                   (x_1_sr, x_2_sr, xy_1, xy_2, xy_3, final_output_score)
        """
        # Process the super-resolved/high-resolution image (x)
        x_1_sr = self.block_1_sr(sr_or_hr_image)
        x_2_sr = self.block_2_sr(x_1_sr)

        # Process the low-resolution image (y)
        y_1_lr = self.block_1_lr(lr_image)

        # Combine features from SR/HR and LR paths. The original code uses torch.add,
        # implying feature maps should have compatible dimensions after processing.
        # Ensure that x_2_sr and y_1_lr have the same spatial dimensions and channel count (128)
        # for addition.
        xy = torch.add(x_2_sr, y_1_lr)

        # Pass combined features through subsequent blocks
        xy_1 = self.block_1(xy)
        xy_2 = self.block_2(xy_1)
        xy_3 = self.block_3(xy_2)

        # Flatten the output of the last convolutional block for the fully connected layers
        xy_3_flat = xy_3.view(xy_3.size(0), -1)

        # Get the final classification score
        final_output_score = self.final(xy_3_flat)

        # Return intermediate features and the final score for GeneratorLossFunction
        return (x_1_sr, x_2_sr, xy_1, xy_2, xy_3, final_output_score)