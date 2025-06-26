import os

from args.base import Args


"""Eth3d"""
class Test(Args):
    def __init__(self):
        super().__init__()
        # dataset
        self.dataset = "eth3d"
        self.dataset_path = os.path.join(self.root_dir, "eth3d")
        self.nviews = 11
        self.mode = "test"
        self.robust_train = False

        # self.scene_list = ['lakeside', "sand_box", "storage_room", "storage_room_2", "tunnel"]
        self.scene_list = ["botanical_garden", "boulders", "bridge", "courtyard", "delivery_area",
                          "door", "electro", "exhibition_hall", "facade", "kicker", "lecture_room",
                          "living_room", "lounge", "meadow", "observatory", "office", "old_computer",
                          "pipes", "playground", "relief", "relief_2", "statue", "terrace", "terrace_2", "terrains"]

        # args
        self.parallel = False
        self.DEVICE = self.get_device(self.parallel)
        self.batch_size = 1
        self.nworks = 1


if __name__ == "__main__":
    eth3d_args = Test()
    eth3d_args.show_args()

















