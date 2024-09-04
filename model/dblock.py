from torch import nn


class DBlock(nn.Module):

    def __init__(self, input_shape: int = 64,
                 output_shape: int = 64,
                 stride: int = 2,
                 bn: bool = True,
                 padding: int = 1) -> None:
        super().__init__()
        self.bntrue = bn
        self.conv_1 = nn.Conv2d(input_shape, output_shape, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=output_shape)
        self.leakyr = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.bn == True:
            return self.leakyr(self.bn(self.conv_1(x)))
        else:
            return self.leakyr(self.conv_1(x))