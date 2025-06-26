import torch.nn as nn
import torch.nn.functional as F

from mvs.unit.base import ConvBNReLU3D


class RegularNet(nn.Module):
    def __init__(self, in_chs, step_depth=False, base_chs=8):
        super(RegularNet, self).__init__()
        c0, c1, c2, c3 = base_chs, base_chs*2, base_chs*4, base_chs*8

        if step_depth:
            kernel, stride, padding, output_padding = 3, 2, 1, 1
        else:
            # kernel, stride, padding, output_padding = (1, 3, 3), (1, 2, 2), (0, 1, 1), (0, 1, 1)
            kernel, stride, padding, output_padding = (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1)


        self.conv01 = ConvBNReLU3D(in_chs, c0, kernel, padding=padding)

        self.conv12 = nn.Sequential(
            ConvBNReLU3D(c0, c1, kernel, stride, padding),
            ConvBNReLU3D(c1, c1, 3, 1, 1),
        )
        self.conv23 = nn.Sequential(
            ConvBNReLU3D(c1, c2, kernel, stride, padding),
            ConvBNReLU3D(c2, c2, 3, 1, 1),
        )
        self.conv343 = nn.Sequential(
            ConvBNReLU3D(c2, c3, kernel, stride, padding),
            ConvBNReLU3D(c3, c3, 3, 1, 1),
            nn.ConvTranspose3d(c3, c2, kernel, stride, padding, output_padding, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )
        self.trconv32 = nn.Sequential(
            nn.ConvTranspose3d(c2, c1, kernel, stride, padding, output_padding, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        self.trconv21 = nn.Sequential(
            nn.ConvTranspose3d(c1, c0, kernel, stride, padding, output_padding, bias=False),
            nn.BatchNorm3d(c0),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(c0, 1, 3, 1, 1, bias=False)

        # print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self, x):

        x1 = self.conv01(x)
        x2 = self.conv12(x1)
        x3 = self.conv23(x2)

        x3 = x3 + self.conv343(x3)
        x2 = x2 + self.trconv32(x3)
        x1 = x1 + self.trconv21(x2)
        x = self.prob(x1).squeeze(1)

        return F.softmax(x, dim=1)



if __name__=="__main__":
    pass

