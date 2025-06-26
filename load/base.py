import cv2, random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from args.paths import get_img_path, get_cam_path, get_depth_path
from utils.data_io import read_pfm

class LoadBase(torch.utils.data.Dataset):
    def __init__(self, data_args):
        super(LoadBase, self).__init__()
        self.dataset = data_args.dataset
        self.dataset_path = data_args.dataset_path
        self.nviews = data_args.nviews
        self.mode = data_args.mode
        self.robust_train = data_args.robust_train
        self.crop_base = data_args.crop_base    # 64

        self.all_samples = None # self._get_all_samples()

        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)


    def __getitem__(self, item):
        lighting = None
        if self.dataset.startswith("dtu"):
            scan, lighting, ref_view, src_views = self.all_samples[item]
        else:
            scan, ref_view, src_views = self.all_samples[item]

        if self.mode == "train" and self.robust_train:
            index = random.sample(range(len(src_views)), self.nviews - 1)
            index = sorted(index)
            rs_views = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)
        else:
            rs_views = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        imgs, ref_depths, extrinsics, intrinsics, projs = [], {}, [], [], []
        depth_min, depth_max, ref_depth_range = 0.0, 0.0, None
        for i, vid in enumerate(rs_views):
            img_filename = get_img_path(self.dataset_path, scan, vid, lighting, self.dataset)
            cam_filename = get_cam_path(self.dataset_path, scan, vid, self.dataset)

            img = self.read_img(img_filename)
            intrinsic, extrinsic, depth_min, depth_max, scale_factor = self._read_cam_file(cam_filename)

            depth_range = np.array([depth_min, depth_max], dtype=np.float32)
            img, intrinsic, extrinsic, depth_range = self._prepare_imgs(img, intrinsic, extrinsic, depth_range, scale)

            imgs.append(img)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

            if i == 0:  # reference view
                ref_depth_range = depth_range

                if self.mode in ["train", "val"]:
                    depth_filename = get_depth_path(self.dataset_path, scan, vid, self.dataset)
                    ref_depth = np.array(read_pfm(depth_filename)[0], dtype=np.float32) * scale_factor * scale

                    h, w = ref_depth.shape
                    ref_depths["3"] = cv2.resize(ref_depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
                    ref_depths["2"] = cv2.resize(ref_depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                    ref_depths["1"] = cv2.resize(ref_depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                    ref_depths["0"] = ref_depth

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        if self.mode in ["train", "val"]:
            return {"imgs": imgs,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "depth_range": ref_depth_range,
                    "ref_depths": ref_depths,
                    }
        else:
            return {"imgs": imgs,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "depth_range": ref_depth_range,
                    "filename": scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}",
                    }


    def __len__(self):
        return len(self.all_samples)


    def _get_all_samples(self):
        pass

    def _read_cam_file(self, cam_filename):
        pass

    def _prepare_imgs(self, img, intrinsic, extrinsic, depth_range, scale):
        h, w = img.shape[:2]
        crop_h, crop_w = h % self.crop_base, w % self.crop_base

        img = img[crop_h//2:h-crop_h//2, crop_w//2:w-crop_w//2, :]
        intrinsic[0, 2] = intrinsic[0, 2] - crop_w//2
        intrinsic[1, 2] = intrinsic[1, 2] - crop_h//2

        extrinsic[:3, 3] *= scale
        depth_range *= scale

        return img, intrinsic, extrinsic, depth_range

    def read_img(self, filename):
        img = Image.open(filename)

        if self.mode == 'train':
            img = self.color_augment(img)

        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


if __name__ == "__main__":
    pass