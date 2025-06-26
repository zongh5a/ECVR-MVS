import os

from args.base import Args


""" BlendedMVS """
class Train(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "bldtrain"
        self.dataset_path = os.path.join(self.root_dir, "blendedmvs768x576")
        self.nviews = 7
        self.mode = "train"
        self.robust_train = True

        self.scene_list = "training_list.txt"

        # args
        self.start_epoch = 1
        self.max_epoch = 10
        self.batch_size = 2
        self.nworks = 2 #self.batch_size//2

        self.lr = 1e-4
        self.lrepochs = [7, 9]
        self.lr_gamma = 0.5
        self.parallel = True
        self.DEVICE = self.get_device(self.parallel)

        self.val = True
        self.fine_tune = True



class Val(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "bldval"
        self.dataset_path = os.path.join(self.root_dir, "blendedmvs768x576")
        self.nviews = 7
        self.mode = "val"
        self.robust_train = False

        self.scene_list = "validation_list.txt"

        # args
        self.parallel = True
        self.DEVICE = self.get_device(self.parallel)
        self.batch_size = 2
        self.nworks = 2  # self.batch_size//2



if __name__ == "__main__":
    bldtrain, bldval = Train(), Val()
    bldtrain.show_args()
    bldval.show_args()