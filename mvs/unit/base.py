import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self,
                 inchs: int,
                 outchs: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 ) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(inchs, outchs, kernel_size, stride, (kernel_size-1)//2, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm2d(outchs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class TrConvBNReLU(nn.Module):
    def __init__(self,
                 inchs: int,
                 outchs: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 output_padding: int=1,
                 groups: int = 1,
                 bias: bool = False,
                 ) -> None:
        super(TrConvBNReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(inchs, outchs, kernel_size, stride, padding, output_padding, groups, bias)
        self.bn = nn.InstanceNorm2d(outchs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU3D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups: int=1,
                 bias: bool=False,
                 ) -> None:
        super(ConvBNReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def homo_warping(src_fea, src_proj, ref_proj, depth_hypos):
    """

    @param src_fea: [B, C, H, W]
    @param src_proj: [B, 4, 4]
    @param ref_proj: [B, 4, 4]
    @param depth_hypos: [B, Ndepth, 1, 1] or [B,NDepths,H,W]
    @return: [B, C, Ndepth, H, W]
    """
    batch, ndepths, _, _ = depth_hypos.shape
    _, channels , height, width= src_fea.shape   #torch.Size([1, 32, 128, 160])
    proj_xy = get_proj_position(src_fea.shape, src_proj, ref_proj, depth_hypos)

    return F.grid_sample(src_fea, proj_xy, mode='bilinear', padding_mode='zeros', align_corners=True)\
        .view(batch, channels, ndepths, height, width)

@torch.no_grad()
def get_proj_position(fea_shape, src_proj, ref_proj, depth_hypos):
    """

    @param src_fea: [B, C, H, W]
    @param src_proj: [B, 4, 4]
    @param ref_proj: [B, 4, 4]
    @param depth_hypos: [B, Ndepth, 1, 1] or [B,NDepths,H,W]
    @return: [B, C, Ndepth, H, W]
    """
    batch, ndepths, H, W = depth_hypos.shape
    batch, channels ,height, width= fea_shape   #torch.Size([1, 32, 128, 160])

    proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # if error, use Python 3.8.10 & torch 2.0.0+cu117

    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_hypos.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth_hypos.device)])
    y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
    xyz = torch.unsqueeze(torch.stack((x, y, torch.ones_like(x))), 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    # del x, y

    proj_xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, ndepths, 1) * \
                    depth_hypos.view(batch, 1, ndepths, H * W) +\
                    trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

    # FIXME divide 0
    temp = proj_xyz[:, 2:3, :, :]
    temp[temp == 0] = 1e-9
    proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]

    proj_xy = torch.stack((proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1,
                           proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1), dim=3)  # [B, Ndepth, H*W, 2]

    return proj_xy.view(batch, ndepths * height, width, 2)


if __name__=="__main__":
    pass
