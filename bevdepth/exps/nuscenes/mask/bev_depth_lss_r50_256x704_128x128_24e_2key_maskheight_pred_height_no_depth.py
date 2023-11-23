from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_mask_exp import \
    BEVDepthLightningModelPredHeight as BaseBEVDepthLightningModel
from bevdepth.models.base_bev_depth import MaskHeightBEVDepthPredHeight

import torch

class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = MaskHeightBEVDepthPredHeight(self.backbone_conf,
                                                  self.head_conf,
                                                  is_train_depth=True)
        
       
    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, masks_2d) = batch
        masks_2d = self.get_downsampled_masks_2d(masks_2d)
        masks_2d = torch.permute(masks_2d, (0, 1, 4, 2, 3)) # change channel 
        masks_pe = masks_2d[:, :, 1:3]
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            masks_pe = masks_pe.cuda()
        
        preds, depth_preds, mask_preds, height_preds = self(sweep_imgs, mats, masks_pe)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
        
        # 前景 mask
        gt_height = masks_2d[:, :, 0]
        B, N, H, W = gt_height.shape
        gt_height = gt_height.view(B*N, H, W)
        gt_mask = torch.where(gt_height>0, 1, 0)
        
        mask_loss = self.get_mask_loss(gt_mask, mask_preds)
        height_loss = self.get_heigth_loss(gt_height, gt_mask, height_preds)
        
        self.log('detection_loss', detection_loss, on_step=True, prog_bar=True)
        self.log('mask_loss', mask_loss, on_step=True, prog_bar=True)
        self.log('height_loss', height_loss, on_step=True, prog_bar=True)
        
        return detection_loss + mask_loss + height_loss



if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_2key_maskheight_pred_height_no_depth')