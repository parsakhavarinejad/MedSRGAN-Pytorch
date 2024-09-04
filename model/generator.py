class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.lrc = []
        for _ in range(8):
            self.lrc.append(ShortResidualConnection())
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lrc = nn.Sequential(*self.lrc, self.conv_2)

        upsample_1 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                                   nn.PixelShuffle(2),
                                   nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                                   nn.PixelShuffle(2))
        conv_3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.upscaler = nn.Sequential(upsample_1, conv_3)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.lrc(x_1)
        x_out = x_1 + x_2
        return self.upscaler(x_out)