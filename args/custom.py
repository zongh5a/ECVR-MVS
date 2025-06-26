import os

from args.base import Args


""" Custom """
class Test(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "custom"
        self.dataset_path = r"E:\3DGS\dataset\tandt_db\db"
        self.nviews = 11
        self.mode = "test"
        self.robust_train = False

        self.scene_list = ['playroom',]

        # args
        self.parallel = False
        self.DEVICE = self.get_device(self.parallel)
        self.batch_size = 1
        self.nworks = 1



if __name__ == "__main__":
    custom_args = Test()
    custom_args.show_args()