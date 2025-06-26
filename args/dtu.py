import os

from args.base import Args


""" DTU """
class Train(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "dtutrain"
        self.dataset_path= os.path.join(self.root_dir, "dtu640x512")
        self.nviews = 5
        self.mode = "train"
        self.robust_train = True

        self.pair_path = os.path.join(self.dataset_path,"Cameras","pair.txt")
        self.lighting_label = [0, 1, 2, 3, 4, 5, 6]
        self.scene_list = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                      45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                      74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                      121, 122, 123, 124, 125, 126, 127, 128]

        # args
        self.start_epoch = 1
        self.max_epoch = 12
        self.batch_size = 2#4
        self.nworks = self.batch_size//2

        self.lr = 1e-3
        self.lrepochs = [7, 10]
        self.lr_gamma = 1/2

        self.parallel = True
        self.DEVICE = self.get_device(self.parallel)

        self.val = True
        self.fine_tune = False



class Val(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "dtuval"
        self.dataset_path = os.path.join(self.root_dir, "dtu640x512")
        self.nviews = 5
        self.mode = "val"
        self.robust_train = False

        self.pair_path = os.path.join(self.dataset_path, "Cameras", "pair.txt")
        self.lighting_label = [3]
        self.scene_list = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

        # args
        self.parallel = True
        self.DEVICE = self.get_device(self.parallel)
        self.batch_size = 4
        self.nworks = self.batch_size//2



class Test(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "dtutest"
        self.dataset_path = os.path.join(self.root_dir, "dtu1600x1200")
        self.nviews = 5
        self.mode = "test"
        self.robust_train = False

        self.pair_path = os.path.join(self.dataset_path, "pair.txt")
        self.lighting_label = [3]
        self.scene_list = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

        # args
        self.parallel = False
        self.DEVICE = self.get_device(self.parallel)
        self.batch_size = 1
        self.nworks = 1



if __name__=="__main__":
    dtutrain, dtuval, dtutest = Train(), Val(), Test()
    dtutrain.show_args()
    dtuval.show_args()
    dtutest.show_args()