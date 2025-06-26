import torch
import torch.nn as nn

from mvs.unit.base import ConvBNReLU3D, homo_warping


class CostAggLaplace(nn.Module):
    def __init__(self,
                 in_chs: int=8,
                 Lambda: float=3.0
                 ):
        super(CostAggLaplace, self).__init__()

        self.Lambda = Lambda

        # (B,s,D,H,W)->(B,1,D,H,W)
        self.weight_probs = nn.Sequential(
            ConvBNReLU3D(in_chs,16, 1, 1, 0),
            ConvBNReLU3D(16, 8, 1, 1, 0),
            nn.Conv3d(8, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

        # print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self, features, depth_hypos, ref_proj, src_projs):  # 1e-8

        # feas
        ref_feature, src_features = features[0], features[1:]  # (B,C,H,W), (nviews-1)*（B,C,H,W）
        ref_volume = ref_feature.unsqueeze(2)   # （B,C,1,H,W）

        cost_volume, weight_sum = 0.0, 0.0
        for src_fea, src_proj in zip(src_features, src_projs):
            # torch.cuda.empty_cache()
            src_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos) # (B,C,D,H,W）
            volume = torch.exp(-self.Lambda*torch.abs(1-src_volume/(ref_volume+1e-8)))
            weight = self.weight_probs(volume)

            cost_volume += volume * weight
            weight_sum += weight
            # del volume, weight

        return cost_volume/weight_sum

