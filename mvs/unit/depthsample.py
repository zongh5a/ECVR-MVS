import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthSample(nn.Module):
    def __init__(self, inv=False):
        super(DepthSample, self).__init__()
        self.inv = inv

    @ torch.no_grad()
    def forward(self, scale_hw, depth_range, depth_hypos, prob_volume, ndepths):

        h, w = scale_hw
        itv = torch.arange(0, ndepths, device=depth_range.device).float()\
                  .reshape(1, -1, 1, 1)/ (ndepths - 1)

        if depth_hypos is None:
            # init & stage0
            depth_min, depth_max = depth_range[:, 0].float(), depth_range[:, 1].float()
            depth_min, depth_max = depth_min.view(-1, 1, 1, 1), depth_max.view(-1, 1, 1, 1)

            if self.inv:
                inv_depth_min, inv_depth_max = 1 / depth_min, 1 / depth_max
                depth_hypos = inv_depth_max + (inv_depth_min - inv_depth_max) * itv
                depth_hypos = torch.flip(1.0 / depth_hypos, dims=[1])
            else:
                depth_hypos = depth_min + (depth_max-depth_min) * itv
            depth_hypos = depth_hypos.repeat(1, 1, h, w)
        else:
            # 1. Window search
            n = (prob_volume.shape[1]+1)//2
            prob_volume_sumn = n * F.avg_pool3d(prob_volume.unsqueeze(1), (n, 1, 1), stride=1, padding=0).squeeze(1)
            _, index_min = prob_volume_sumn.max(1, keepdim=True)  # B 1 H W
            next_depth_min, next_depth_max = \
                torch.gather(depth_hypos, 1, index_min).squeeze(1), \
                    torch.gather(depth_hypos, 1, index_min + n - 1).squeeze(1)  # B H W

            # 2. deal extend range
            max_prob_index = prob_volume.max(1, keepdim=False)[1]  # B 1 H W
            mask_out_low, mask_out_high = max_prob_index == 0, max_prob_index == (prob_volume.shape[1] - 1)

            interval = (depth_hypos[:, 1] - depth_hypos[:, 0])
            next_depth_min[mask_out_low] -= interval[mask_out_low]
            next_depth_max[mask_out_low] -= interval[mask_out_low]
            next_depth_min[mask_out_high] += interval[mask_out_high]
            next_depth_max[mask_out_high] += interval[mask_out_high]

            # 3. generate depth hypos
            if self.inv:
                inv_depth_min, inv_depth_max = 1 / next_depth_min, 1 / next_depth_max
                depth_hypos = \
                    inv_depth_max[:, None, :, :] + (inv_depth_min - inv_depth_max)[:, None, :, :] * itv
                depth_hypos = torch.flip(1.0 / depth_hypos, dims=[1])
            else:
                depth_hypos = \
                    next_depth_min[:, None, :, :] + (next_depth_max - next_depth_min)[:, None, :, :] * itv

            # 4. upsample
            if next_depth_min.shape[-2:] != scale_hw:
                depth_hypos = F.interpolate(depth_hypos, scale_factor=2, mode='bilinear', align_corners=False)

        # depth_hypos[torch.isnan(depth_hypos)] = depth_range.mean()  # 1e-6    # e

        return depth_hypos

