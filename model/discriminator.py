import torch
from torch import nn

from model.dblock import DBlock


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1_sr = nn.Sequential(DBlock(3, 64, stride=1, bn=False),
                                        DBlock(64, 64, stride=2))
        self.block_2_sr = nn.Sequential(DBlock(64, 128, stride=1),
                                        DBlock(128, 128, stride=2))
        self.block_1_lr = nn.Sequential(DBlock(3, 64, stride=1, bn=False),
                                        DBlock(64, 128, stride=1, bn=True))

        self.block_1 = nn.Sequential(DBlock(128, 256, stride=1),
                                     DBlock(256, 256, stride=2))
        self.block_2 = nn.Sequential(DBlock(256, 512, stride=1),
                                     DBlock(512, 512, stride=2))
        self.block_3 = nn.Sequential(DBlock(512, 1024, stride=1),
                                     DBlock(1024, 1024, stride=2))

        self.final = nn.Sequential(nn.Linear(1024 * 8 * 8, 100),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(100, 1),
                                   nn.Sigmoid())

    def forward(self, y, x):
        x_1 = self.block_1_sr(x)
        x_2 = self.block_2_sr(x_1)
        y_1 = self.block_1_lr(y)
        xy = torch.add(x_2, y_1)
        xy_1 = self.block_1(xy)
        xy_2 = self.block_2(xy_1)
        xy_3 = self.block_3(xy_2)
        xy_3 = xy_3.view(xy_3.size(0), -1)
        final = self.final(xy_3)

        return (x_1, x_2, xy_1, xy_2, xy_3, final)