class ShortResidualConnection(nn.Module):
    def __init__(self, input_shape: int = 64) -> None:
        super().__init__()
        RWMAN = []
        for _ in range(16):
            RWMAN.append(RWMAB())

        self.src = nn.Sequential(*RWMAN,
                                 nn.Conv2d(64, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x_1 = self.src(x)
        return x_1 + x