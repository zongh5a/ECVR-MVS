import os

from args.base import Args


""" Tanks """
class Test(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "tanks"
        self.dataset_path = os.path.join(self.root_dir, "TanksandTemples")
        self.nviews = 11
        self.mode = "test"
        self.robust_train = False

        """Intermediate set"""
        self.dataset_path = os.path.join(self.dataset_path, "intermediate")
        self.scene_list = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']

        """Advanced set"""
        # self.dataset_path = os.path.join(self.dataset_path, "advanced")
        # self.scene_list = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Temple', 'Palace']

        # args
        self.parallel = False
        self.DEVICE = self.get_device(self.parallel)
        self.batch_size = 1
        self.nworks = 1



if __name__ == "__main__":
    tanks_args = Test()
    tanks_args.show_args()