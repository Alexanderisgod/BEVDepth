from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.models.base_bev_depth import (
    MaskHeightBEVDepth, MaskHeightBEVDepthPred, MaskHeightBEVDepthPredHeight
)
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn

from functools import partial
import torch
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast


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
                                       return_mask2d=True,
                                       foreground_mask_only=False,)

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
                                     return_mask2d=True,
                                     foreground_mask_only=False,)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion, is_return_mask2d=True,),
            num_workers=4,
            sampler=None,
        )
        return val_loader
    
class BEVDepthLightningModelPred(BaseBEVDepthLightningModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MaskHeightBEVDepthPred(self.backbone_conf,
                                            self.head_conf,
                                            is_train_depth=True)
        self.count=0 
    
    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, masks_2d) = batch
        masks_2d = self.get_downsampled_masks_2d(masks_2d)
        masks_2d = torch.permute(masks_2d, (0, 1, 4, 2, 3)) # change channel 
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            masks_2d = masks_2d.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        
        preds, depth_preds, mask_preds = self(sweep_imgs, mats)
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
        mask_loss = self.get_mask_loss(masks_2d, mask_preds)
        mask = torch.argmax(mask_preds, dim=1)

        self.log('detection_loss', detection_loss, on_step=True, prog_bar=True)
        self.log('depth_loss', depth_loss, on_step=True, prog_bar=True)
        self.log('mask_loss', mask_loss, on_step=True, prog_bar=True)
        
        return detection_loss + depth_loss + mask_loss

    def get_downsampled_masks_2d(self, batch_masks_2d):
        start = self.downsample_factor//2
        return batch_masks_2d[:, :, start::self.downsample_factor, start::self.downsample_factor]

    def get_mask_loss(self, gt_mask, pred_mask):
        gt_mask = gt_mask[:, :, 0]
        B, N, H, W = gt_mask.shape
        gt_mask = gt_mask.view(B*N, H, W).to(torch.int64)
        gt_mask = F.one_hot(gt_mask, num_classes=2).permute(0, 3, 1, 2)
        mask_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask.to(pred_mask))
        return mask_loss
    
    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _, depth_labels, masks_2d) = batch
        
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds, masks_pred = self.model(sweep_imgs, mats)
        
        import mmcv
        save_data = {
            "imgs": sweep_imgs.cpu().numpy(),
            "gt_mask": masks_2d.cpu().numpy(),
            "pred_mask": masks_pred.cpu().numpy()
        }
        mmcv.dump(save_data, f"vis/mask{self.count}.pkl")
        self.count += 1
        if self.count>=4:
            exit()
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
                                       return_mask2d=True,) # 只使用前景mask

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
    
    # vis
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
                                     # visulization
                                     imnormalize=True,
                                     return_mask2d=True,)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_mask2d=True,),
            num_workers=4,
            sampler=None,
        )
        return val_loader
    
class BEVDepthLightningModelPredHeight(BaseBEVDepthLightningModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MaskHeightBEVDepthPredHeight(self.backbone_conf,
                                                  self.head_conf,
                                                  is_train_depth=True)
        self.hbound=0.2
        self.height_bins=16
        self.count=0
    
    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, masks_2d) = batch
        masks_2d = self.get_downsampled_masks_2d(masks_2d)
        masks_2d = torch.permute(masks_2d, (0, 1, 4, 2, 3)) # change channel 
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        
        preds, depth_preds, mask_preds, height_preds = self(sweep_imgs, mats)
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
        
        # 前景 mask
        gt_height = masks_2d[:, :, 0]
        B, N, H, W = gt_height.shape
        gt_height = gt_height.view(B*N, H, W)
        gt_mask = torch.where(gt_height>0, 1, 0)
        
        mask_loss = self.get_mask_loss(gt_mask, mask_preds)
        height_loss = self.get_heigth_loss(gt_height, gt_mask, height_preds)
        
        self.log('detection_loss', detection_loss, on_step=True, prog_bar=True)
        self.log('depth_loss', depth_loss, on_step=True, prog_bar=True)
        self.log('mask_loss', mask_loss, on_step=True, prog_bar=True)
        self.log('height_loss', height_loss, on_step=True, prog_bar=True)
        
        return detection_loss + depth_loss + mask_loss + height_loss

    def get_downsampled_masks_2d(self, batch_masks_2d):
        start = self.downsample_factor//2
        return batch_masks_2d[:, :, start::self.downsample_factor, start::self.downsample_factor]

    def get_mask_loss(self, gt_mask, pred_mask):
        gt_mask = F.one_hot(gt_mask.to(torch.int64), num_classes=2).permute(0, 3, 1, 2)
        mask_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask.to(pred_mask))
        return mask_loss*2
    
    def get_heigth_loss(self, gt_height, gt_mask, pred_height):
        gt_height = gt_height//self.hbound # 高度离散化 B*N, H, W
        gt_height = gt_height * gt_mask# 去除背景的干扰 B*N, H, W
        gt_height = torch.clamp(gt_height, min=0, max=3.1)
        gt_height = F.one_hot(gt_height.to(torch.int64), num_classes=self.height_bins).permute(0, 3, 1, 2)
        height_loss = F.binary_cross_entropy_with_logits(pred_height, gt_height.float())
        # print(f"height_loss:{torch.isnan(height_loss).any()}")
        return height_loss*2
    
    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats)
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
                                       return_mask2d=True,
                                       foreground_mask_only=False,)

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

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _, depth_labels, masks_2d) = batch
        
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds, masks_pred, height = self.model(sweep_imgs, mats)
        
        import mmcv
        save_data = {
            "imgs": sweep_imgs.cpu().numpy(),
            "gt_mask": masks_2d.cpu().numpy(),
            "pred_mask": masks_pred.cpu().numpy(),
            "height": height.cpu().numpy(),
        }
        mmcv.dump(save_data, f"vis/height/mask{self.count}.pkl")
        self.count += 1
        if self.count>=4:
            exit()
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

    # vis
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
                                     # visulization
                                     imnormalize=True,
                                     return_mask2d=True,
                                     foreground_mask_only=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_mask2d=True,),
            num_workers=4,
            sampler=None,
        )
        return val_loader
    