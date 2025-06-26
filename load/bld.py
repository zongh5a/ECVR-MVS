import os
import numpy as np

from load.base import LoadBase


class LoadDataset(LoadBase):
    def __init__(self, data_args) -> None:
        super(LoadDataset, self).__init__(data_args)
        self.listfile = os.path.join(self.dataset_path, data_args.scene_list)    # "training_list.txt"
        self.all_samples = self._get_all_samples()

        self.scale_factors = {}

    def _get_all_samples(self):
        all_samples = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)

            with open(os.path.join(self.dataset_path, pair_file)) as f:
                num_viewpoint = int(f.readline())

                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            src_views += [src_views[0]] * (self.nviews - len(src_views))

                        all_samples.append((scan, ref_view, src_views))

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
        depth_max = float(lines[11].split()[3])

        scan = filename.split("/")[-3]
        if scan not in self.scale_factors:
            self.scale_factors[scan] = 100.0 / depth_min    # Scenarios share the same value
            assert np.isnan(self.scale_factors[scan]).sum() == 0 and np.isinf(self.scale_factors[scan]).sum() == 0

        scale_factor = self.scale_factors[scan]

        depth_min *= scale_factor
        depth_max *= scale_factor
        extrinsics[:3, 3] *= scale_factor

        return intrinsics, extrinsics, depth_min, depth_max, scale_factor


if __name__=="__main__":
    dataset = LoadDataset(datasetpath=".")

