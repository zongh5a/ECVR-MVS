import torch


class CoreNet(torch.nn.Module):
    def __init__(self, ndepths, upsample_iters, Backbone, DepthSample, MatrixScale, VolumeBuild, VolumeRegular, Regress):
        super(CoreNet, self).__init__()
        self.ndepths = ndepths
        self.upsample_iters = upsample_iters
        self.Backbone = Backbone
        self.DepthSample = DepthSample
        self.MatrixScale = MatrixScale
        self.VolumeBuild = VolumeBuild
        self.VolumeRegular = VolumeRegular
        self.DepthRegress, self.ConfidenceRegress = Regress

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))


    def forward(self, origin_imgs, extrinsics, intrinsics, depth_range):
        """
        predict depth
        @param origin_imgs: （B,VIEW,3,H,W） view0 is ref img
        @param extrinsics: （B,VIEW,4,4）
        @param intrinsics: （B,VIEW,3,3）
        @param depth_range: (B, 2) B*(depth_min, depth_max) dtu: [425.0, 935.0] tanks: [-, -]
        @return:
        """
        B, V, C, H, W = origin_imgs.shape
        origin_imgs = torch.unbind(origin_imgs.float(), 1)  # VIEW*(B,C,H,W)

        # 0.0 feature extraction
        rs_feas = [self.Backbone(img) for img in origin_imgs]

        depth_hypos, prob_volume, loss_datas = None, None, []
        prob_conf = torch.ones([B, H, W], dtype=torch.float32, device=depth_range.device)
        for iter, ndepths in enumerate(self.ndepths):
            if iter in self.upsample_iters:
                stage = self.upsample_iters.index(iter)
                # 0.1 stage feas
                features = [fea[stage] for fea in rs_feas]
                # 1.0 scale matrix
                ref_proj, src_projs = self.MatrixScale(intrinsics, extrinsics, stage)

            # 2.0 get depth samples
            depth_hypos = \
                self.DepthSample(features[0].shape[-2:], depth_range, depth_hypos, prob_volume, ndepths)

            # 3.0 build cost volume
            cost_volume = self.VolumeBuild[iter](features, depth_hypos, ref_proj, src_projs)

            # 4.0
            prob_volume = self.VolumeRegular[iter](cost_volume)

            loss_datas.append([depth_hypos, prob_volume])

            # 5.0 conf map
            if not self.training:
                prob_conf = self.ConfidenceRegress(iter, prob_conf, prob_volume)

        # 5.1 depth map
        depth = self.DepthRegress(depth_hypos, prob_volume)

        if self.training:
            return {
                "depth": depth,
                "loss_data": loss_datas
            }

        return {
            "depth": depth,
            "confidence": prob_conf.float()
        }


if __name__=="__main__":
    pass

