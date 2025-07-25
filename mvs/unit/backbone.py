import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mvs.unit.base import ConvBNReLU


class FeatureEX(nn.Module):
    def __init__(self,
                 out_chs: Tuple=(8, 16, 32, 64),
                 ) -> None:
        super(FeatureEX, self).__init__()
        c0, c1, c2, c3 = out_chs

        # 1/1
        self.conv01 = nn.Sequential(
            ConvBNReLU(3, c0, 3, 1, 1),
            ConvBNReLU(c0, c0, 3, 1, 1)
        )
        # 1/2
        self.conv12 = nn.Sequential(
            ConvBNReLU(c0, c1, 5, 2, 2),
            ConvBNReLU(c1, c1, 3, 1, 1),
            ConvBNReLU(c1, c1, 3, 1, 1)
        )
        # 1/4
        self.conv23 = nn.Sequential(
            ConvBNReLU(c1, c2, 5, 2, 2),
            ConvBNReLU(c2, c2, 3, 1, 1),
            ConvBNReLU(c2, c2, 3, 1, 1),
        )
        #1/8
        self.conv34 = nn.Sequential(
            ConvBNReLU(c2, c3, 5, 2, 2),
            ConvBNReLU(c3, c3, 3, 1, 1),
            ConvBNReLU(c3, c3, 3, 1, 1)
        )

        self.lat1 = nn.Conv2d(c0, c3, 1, bias=True)
        self.lat2 = nn.Conv2d(c1, c3, 1, bias=True)
        self.lat3 = nn.Conv2d(c2, c3, 1, bias=True)

        self.out1 = nn.Conv2d(c3, c0, 3, 1, 1, bias=False)
        self.out2 = nn.Conv2d(c3, c1, 3, 1, 1, bias=False)
        self.out3 = nn.Conv2d(c3, c2, 3, 1, 1, bias=False)
        self.out4 = nn.Conv2d(c3, c3, 1, bias=False)

        # print('{} parameters: {}'
        #       .format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self,
                x: torch.Tensor,
                ) :
        B, _, H, W = x.shape

        x1 = self.conv01(x)
        x2 = self.conv12(x1)
        x3 = self.conv23(x2)
        x4 = self.conv34(x3)

        y4 = self.out4(x4)
        x3 = F.interpolate(x4, scale_factor=2.0, mode="nearest") + self.lat3(x3)
        y3 = self.out3(x3)
        x2 = F.interpolate(x3, scale_factor=2.0, mode="nearest") + self.lat2(x2)
        y2 = self.out2(x2)
        x1 = F.interpolate(x2, scale_factor=2.0, mode="nearest") + self.lat1(x1)
        y1 = self.out1(x1)
        # del x3, x2, x1

        return y4, y3, y2, y1



