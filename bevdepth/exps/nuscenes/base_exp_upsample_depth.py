from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 对预测的点云上采样进行监督
        self.downsample_factor = self.downsample_factor//4
        self.upsample_depth =  nn.UpsamplingBilinear2d(scale_factor=4)


    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)

        depth_preds  = self.upsample_depth(depth_preds)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

