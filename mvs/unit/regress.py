import torch
import torch.nn.functional as F


def depth_regression(depth_hypos, prob_volume):
    max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W

    return torch.gather(depth_hypos, 1, max_indices).squeeze(1)  # B H W


@torch.no_grad()
def confidence_regress(iter, prob_conf, prob_volume):

    n = prob_volume.shape[1]//3 if iter < 6 else 1  # 2
    with torch.no_grad():
        if n % 2 == 1:
            prob_volume_sum4 = n * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=[0, 0, 0, 0, n // 2, n // 2]),
                                                (n, 1, 1), stride=1, padding=0).squeeze(1)
        else:
            prob_volume_sum4 = n * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=[0, 0, 0, 0, n // 2 - 1, n // 2]),
                                                (n, 1, 1), stride=1, padding=0).squeeze(1)
        conf, _ = torch.max(prob_volume_sum4, 1)

    if prob_conf.shape[-2:] != conf.shape[-2:]:
        conf = F.interpolate(
            conf.unsqueeze(1), prob_conf.shape[-2:], mode='bilinear').squeeze(1)

    return prob_conf * conf


if __name__=="__main__":
    pass

