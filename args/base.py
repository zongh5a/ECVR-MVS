import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# apt-get install nvidia-modprobe

import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s: %(message)s")

import random, numpy, torch
seed_id = 0
random.seed(seed_id)
numpy.random.seed(seed_id)
torch.manual_seed(seed_id)


""" basic class: Args """
class Args():
    def __init__(self):
        self.crop_base = 64
        self.root_dir = "/hy-tmp"
        self.output_path = "/hy-tmp/outputs"
        self.pth_path = 'pth'  # pth file save path
        self.valid_cuda = os.environ['CUDA_VISIBLE_DEVICES']
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.pth_path, exist_ok=True)


    def show_args(self):
        print(self.__class__.__name__+":")
        for k, v in self.__dict__.items():
            print("\t"+k, ":", v)


    def get_device(self, parallel):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.deterministic = False
            if parallel:
                torch.cuda.manual_seed_all(seed_id)
                # os.environ["NCCL_DEBUG"] = "INFO"
                return os.environ['CUDA_VISIBLE_DEVICES']
            else:
                torch.cuda.manual_seed(seed_id)
                return torch.device("cuda:0")
        else:
            return torch.device("cpu")


if __name__ == "__main__":
    pass

