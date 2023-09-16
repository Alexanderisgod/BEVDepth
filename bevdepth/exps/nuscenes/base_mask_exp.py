from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.models.base_bev_depth import MaskHeightBEVDepth
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn

from functools import partial
import torch


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MaskHeightBEVDepth(self.backbone_conf,
                                        self.head_conf,
                                        is_train_depth=True)

    def forward(self, sweep_imgs, mats, masks_2d):
        return self.model(sweep_imgs, mats, masks_2d)
    
    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, masks_2d) = batch
        
        # import mmcv
        # import numpy as np
        # data = {
        #     'imgs':sweep_imgs.cpu().numpy(),
        #     'masks_2d':masks_2d.cpu().numpy()
        # }
        # path = '/home/yhzn/xiaohuahui/BEVDepth/vis'
        # mmcv.dump(data,f'{path}/data.pkl')
        # exit()

        masks_2d = self.get_downsampled_masks_2d(masks_2d)
        masks_2d = torch.permute(masks_2d, (0, 1, 4, 2, 3)) # change channel 
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            masks_2d = masks_2d.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, depth_preds = self(sweep_imgs, mats, masks_2d)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        self.log('detection_loss', detection_loss, on_step=True, prog_bar=True)
        self.log('depth_loss', depth_loss, on_step=True, prog_bar=True)
        return detection_loss + depth_loss
    
    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _, masks_2d) = batch
        
        masks_2d = self.get_downsampled_masks_2d(masks_2d)
        masks_2d = torch.permute(masks_2d, (0, 1, 4, 2, 3)) # change channel 
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            masks_2d = masks_2d.cuda()
        preds = self.model(sweep_imgs, mats, masks_2d)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results
    
    def get_downsampled_masks_2d(self, batch_masks_2d):
        start = self.downsample_factor//2
        return batch_masks_2d[:, :, start::self.downsample_factor, start::self.downsample_factor]

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.train_info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       use_fusion=self.use_fusion,
                                       imnormalize=True,
                                       return_mask2d=True,)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_mask2d=True,),
            sampler=None,
        )
        return train_loader
    
    def val_dataloader(self):
        val_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                     bda_aug_conf=self.bda_aug_conf,
                                     classes=self.class_names,
                                     data_root=self.data_root,
                                     info_paths=self.val_info_paths,
                                     is_train=False,
                                     img_conf=self.img_conf,
                                     num_sweeps=self.num_sweeps,
                                     sweep_idxes=self.sweep_idxes,
                                     key_idxes=self.key_idxes,
                                     return_depth=self.use_fusion,
                                     use_fusion=self.use_fusion,
                                     return_mask2d=True,)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion, is_return_mask2d=True,),
            num_workers=4,
            sampler=None,
        )
        return val_loader