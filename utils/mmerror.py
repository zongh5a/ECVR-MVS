import os, torch, cv2

import config
from utils.data_io import read_pfm


# def _prepare_imgs(img, crop_base=64):
#     h, w = img.shape
#     crop_h, crop_w = h % crop_base, w % crop_base
#
#     img = img[crop_h // 2:h - crop_h // 2, crop_w // 2:w - crop_w // 2]
#
#
#     return img


def _prepare_imgs(img, crop_base=64):
    h, w = img.shape[:2]
    max_w, max_h = 1152, 864
    scale = 1.0 * max_h / h
    if scale * w > max_w:
        scale = 1.0 * max_w / w
    new_w, new_h = scale * w // crop_base * crop_base, scale * h // crop_base * crop_base
    img = cv2.resize(img, (int(new_w), int(new_h)))

    return img

def cal_mm_error(prob_path, gt_path, scans, nviews=49, depth_min=425.0):
    thres2mm_error, thres4mm_error, thres8mm_error, n = 0.0, 0.0, 0.0, 0
    for scan in scans:
        for view in range(nviews):

            depth_prob_path = os.path.join(prob_path, scan, "depth_est", "{:0>8d}.pfm".format(view))
            depth_gt_path = os.path.join(gt_path, scan, 'depth_map_{:0>4}.pfm'.format(view))

            depth_prob, _ = read_pfm(depth_prob_path)
            depth_gt, _ = read_pfm(depth_gt_path)

            # h, w = depth_prob.shape
            # depth_gt = depth_gt[:h, :w]
            depth_gt = _prepare_imgs(depth_gt)
            mask = depth_gt > depth_min


            depth_abs = torch.abs(torch.from_numpy(depth_prob[mask]).float() - torch.from_numpy(depth_gt[mask]).float())
            thres2mm_error += torch.mean((depth_abs <= 2.0).float()) # !!!:  <
            thres4mm_error += torch.mean((depth_abs <= 4.0).float())
            thres8mm_error += torch.mean((depth_abs <= 8.0).float())
            n +=1

            print(
                str(n), "\\", str(len(scans)*nviews), ":\t",
                "thres2mm_error: {:.4f}%".format(thres2mm_error / n),
                "thres4mm_error: {:.4f}%".format(thres4mm_error / n),
                "thres8mm_error: {:.4f}%".format(thres8mm_error / n),
            )

    print(
        "thres2mm_error: {:.4f}%".format(thres2mm_error/n),
        "thres4mm_error: {:.4f}%".format(thres4mm_error/n),
        "thres8mm_error: {:.4f}%".format(thres8mm_error/n),
    )


if __name__ == "__main__":
    load_args = config.DTUTest()
    label = load_args.scene_list

    scans = ["scan" + str(i) for i in label]
    prob_path = "\hy-tmp\outputs"
    gt_path = "D:\z_experiment_data\dtu_training\mvs_training\dtu\Depths_raw\Depths"

    cal_mm_error(prob_path, gt_path, scans)