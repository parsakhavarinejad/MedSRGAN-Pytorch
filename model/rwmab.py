class RWMAB(nn.Module):
    """
    This class implements the Residual Whole Map Attention Network (RWMAN),
    a modification of RCAN for extracting features from low-resolution (LR) images
    and feeding them into a generator for image upscaling.
    """

    def __init__(self, input_shape: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_out = self.conv1(x)
        attention = self.attention(x_out)
        x_out = torch.mul(x_out, attention) + x
        return x_out