import os, cv2
import numpy as np

from load.base import LoadBase
from utils import data_io

class LoadDataset(LoadBase):
    def __init__(self, data_args):
        super(LoadDataset, self).__init__(data_args)
        self.scene_list = data_args.scene_list

        self.all_samples = self._get_all_samples()

        self._prepare_imgs = self._prepare_imgs_

    def _get_all_samples(self):
        all_samples = []
        for scan in self.scene_list:
            num_viewpoint, pair_data = \
                data_io.read_pairfile(os.path.join(self.dataset_path, scan, 'pair.txt'))    # "cams",
            for ref, srcs in pair_data:
                all_samples.append([scan, ref, srcs])

        return all_samples

    def _read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1]) #* 192 + depth_min

        scale_factor = 1.0

        return intrinsics, extrinsics, depth_min, depth_max, scale_factor


    def _prepare_imgs_(self, img, intrinsic, extrinsic, depth_range, scale, ):
        H, W = img.shape[:2]

        # resize
        new_W = 2048
        new_H = int(new_W * (H / W))
        img = cv2.resize(img, (new_W, new_H))

        intrinsic[0, :] *= new_W / W
        intrinsic[1, :] *= new_H / H

        # crop
        crop_h= new_H % self.crop_base
        img = img[crop_h // 2:new_H - crop_h // 2]

        if crop_h % 2 == 1:
            img = img[:-1]
        intrinsic[1, 2] -= crop_h // 2

        return img, intrinsic, extrinsic, depth_range


if __name__=="__main__":
    pass
