import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, In, Out):
        super().__init__()
        self.conv1 = nn.Conv2d(In, Out, 3, 1, 1)
        self.conv2 = nn.Conv2d(Out, Out, 3, 1, 1)
        self.norm = nn.BatchNorm2d(Out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class Down(nn.Module):
    def __init__(self, In, Out):
        super(Down, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.double_conv = DoubleConv(In, Out)

    def forward(self, x):
        x = self.pool1(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, In, Out):
        super(Up, self).__init__()

        self.upsam1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(2 * Out, In)
        self.conv1 = nn.Conv2d(In, Out, 3, 1, 1)
        # self.norm = nn.BatchNorm2d(In)

    def forward(self, x1, x2):
        x1 = self.upsam1(x1)
        # x2 = self.norm(x2)
        x2 = self.conv1(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)

        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.double_conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(64, 128)
        self.up2 = Up(32,64)
        self.conv = nn.Conv2d(32, 1 ,3 ,1 ,1)
        # self.norm = nn.BatchNorm2d(1)

    def forward(self, x1):
        x1 = self.double_conv(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.norm(x)
        x= self.conv(x)
        return x


exp = np.random.rand(1, 1, 12, 20)
exp = torch.from_numpy(exp)
exp = exp.double()

net = Unet()
net = net.double()

print(net(exp).size())
