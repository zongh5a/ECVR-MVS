import cv2
import numpy as np

from load.base import LoadBase
from utils import data_io


class LoadDataset(LoadBase):
    def __init__(self, data_args):
        super(LoadDataset, self).__init__(data_args)
        self.pair_path = data_args.pair_path
        self.scene_list = data_args.scene_list
        self.lighting_label = data_args.lighting_label

        self.all_samples = self._get_all_samples()

        # resize to 1152x864
        if data_args.dataset == "dtutest":
            self._prepare_imgs = self._prepare_imgs_

    def _get_all_samples(self):
        num_viewpoint, pairs = data_io.read_pairfile(self.pair_path)
        all_samples = []
        for scene in self.scene_list:
            for r, s in pairs:
                for lighting in self.lighting_label:
                    all_samples.append(["scan{}".format(scene), lighting, r, s])

        return all_samples

    def _read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        #
        depth_min = 425.0   #float(lines[11].split()[0])
        depth_max = 935.0   #float(lines[11].split()[1])

        scale_factor = 1.0

        return intrinsics, extrinsics, depth_min, depth_max, scale_factor

    def _prepare_imgs_(self, img, intrinsic, extrinsic, depth_range, scale, ):
        h, w = img.shape[:2]
        max_w, max_h = 1152, 864
        scale = 1.0 * max_h / h
        if scale * w > max_w:
            scale = 1.0 * max_w / w
        new_w, new_h = scale * w // self.crop_base * self.crop_base, scale * h // self.crop_base * self.crop_base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsic[0, :] *= scale_w
        intrinsic[1, :] *= scale_h
        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsic, extrinsic, depth_range


if __name__=="__main__":
    pass
