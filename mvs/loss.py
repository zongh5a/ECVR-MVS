import torch

from args.net import model_args


def dtu_loss(outputs, depth_gts, depth_range):

    loss = 1e-8
    depth_min, depth_max = depth_range[:, 0].view(-1, 1, 1), depth_range[:, 1].view(-1, 1, 1)

    for iter, (depth_hypos, prob_volume) in enumerate(outputs["loss_data"]):
        if iter in model_args.upsample_iters:
            stage = model_args.upsample_iters.index(iter)
            depth_gt = list(depth_gts.values())[stage]
            mask = depth_gt >= depth_min

        n = 2 if iter < 6 else 1
        loss += cross_entropy_loss(mask, depth_hypos, depth_gt, prob_volume, n)

    return loss, mask


def bld_loss(outputs, depth_gts, depth_range):

    return dtu_loss(outputs, depth_gts, depth_range)


def eth3d_loss(outputs, depth_gts, depth_range):

    return dtu_loss(outputs, depth_gts, depth_range)


def cross_entropy_loss(mask, depth_hypos, depth_gt, prob_volume, n=2):
    B, D, H, W = prob_volume.shape
    if depth_hypos.shape != prob_volume.shape:
        depth_hypos = depth_hypos.repeat(1, 1, H, W)

    with torch.no_grad():
        if n == 1:
            gt_index_image = torch.argmin(torch.abs(depth_hypos - depth_gt.unsqueeze(1)), dim=1)
            gt_index_image = torch.mul(mask, gt_index_image.type(torch.float))
            gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W
            gt_index_volume = torch.zeros(B, D, H, W).type(mask.type()).scatter_(1, gt_index_image, 1)
        if n == 2:
            # two-hot coding gt volume
            dist = torch.abs(depth_hypos - depth_gt.unsqueeze(1))
            sorted, indices = dist.sort(descending=False, dim=1)
            gt_index_image = indices[:, :2]  # B, 2, H, W
            gt_index_volume_mask = torch.zeros(depth_hypos.shape).type(depth_hypos.type()).scatter_(1, gt_index_image, 1)
            gt_index_volume = gt_index_volume_mask * dist
            gt_index_volume = (1 - gt_index_volume / torch.sum(gt_index_volume, dim=1, keepdim=True)) * gt_index_volume_mask

    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1)  # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    valid_pixel_num = torch.sum(mask, dim=[1, 2]) + 1e-6
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    return masked_cross_entropy