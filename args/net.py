import torch.nn as nn

from args.base import Args
from mvs.core import CoreNet
from mvs.unit import matrixscale, backbone
from mvs.unit.depthsample import DepthSample
from mvs.unit.volumebuild import CostAggLaplace
from mvs.unit.regular import RegularNet
from mvs.unit.regress import depth_regression, confidence_regress


# model args
class ModelArgs(Args):
    def __init__(self):
        self.ndepths = (9, 9, 9, 9, 9, 9, 7)
        self.upsample_iters = (0, 2, 4, 6)
        self.inv_depths = False
        self.volume_chs = (64, 64, 32, 32, 16, 16, 8)
        self.Lambda = 3.0

        self.show_args()

model_args = ModelArgs()

# scale matrix method
MatrixScale = matrixscale.scale_matrix
# Feature map extraction network
Backbone = backbone.FeatureEX()
# Depth sampling method
DepthSample = DepthSample(model_args.inv_depths)
# build cost volume
VolumeBuild = nn.ModuleList([
    CostAggLaplace(chs, model_args.Lambda) for chs in model_args.volume_chs
])
# Rerular volume
VolumeRegular = nn.ModuleList([
    RegularNet(in_chs) for in_chs in model_args.volume_chs
])
# predict
Regress = depth_regression, confidence_regress
# model
model = CoreNet(model_args.ndepths, model_args.upsample_iters, Backbone, DepthSample, MatrixScale, VolumeBuild, VolumeRegular, Regress)


if __name__=="__main__":
    MVSModel = model
