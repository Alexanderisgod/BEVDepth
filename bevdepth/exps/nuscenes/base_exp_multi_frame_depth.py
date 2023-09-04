from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.datasets.multi_nusc_det_dataset import MultiNuscDataset

from functools import partial
import torch

from bevdepth.datasets.nusc_det_dataset import collate_fn


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sweep_idxes = [0, 1, 2]

    def train_dataloader(self):
        train_dataset = MultiNuscDataset(ida_aug_conf=self.ida_aug_conf,
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
                                         use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = MultiNuscDataset(ida_aug_conf=self.ida_aug_conf,
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
                                       use_fusion=self.use_fusion)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def predict_dataloader(self):
        predict_dataset = MultiNuscDataset(ida_aug_conf=self.ida_aug_conf,
                                           bda_aug_conf=self.bda_aug_conf,
                                           classes=self.class_names,
                                           data_root=self.data_root,
                                           info_paths=self.predict_info_paths,
                                           is_train=False,
                                           img_conf=self.img_conf,
                                           num_sweeps=self.num_sweeps,
                                           sweep_idxes=self.sweep_idxes,
                                           key_idxes=self.key_idxes,
                                           return_depth=self.use_fusion,
                                           use_fusion=self.use_fusion)
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return predict_loader
