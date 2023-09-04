import os

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from bevdepth.datasets.nusc_det_dataset import (
    NuscDetDataset, collate_fn,
    depth_transform,
    img_transform
)
from functools import partial
import matplotlib.pyplot as plt


H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]


def map_pointcloud_to_image(
    lidar_points,
    img,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


class MultiNuscDataset(NuscDetDataset):

    def get_lidar_depth_multiframe(self, lidar_points_list, img, cam_info):
        # transform points from sweep ego to key ego
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        multi_frame_point_cloud = np.concatenate(
            lidar_points_list, axis=0)
        # coordinates transform
        pts_img, depth = map_pointcloud_to_image(
            multi_frame_point_cloud.copy(), img, cam_calibrated_sensor, cam_ego_pose)

        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)

    def get_image(self, cam_infos, cams, lidar_infos=None, location=None,):
        """Given data and cam_names, return image data needed.
        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.
        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_extrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()

        # 需要点云数据时候-->深度信息/点云特征融合
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            # lidar_calibrated_sensor = lidar_infos[0]['LIDAR_TOP']['calibrated_sensor']
            for sweep_idx, lidar_info in enumerate(lidar_infos):
                lidar_path = lidar_info['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                    dtype=np.float32,
                    count=-1).reshape(-1, 5)[..., :4]

                lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
                lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
                # First step: transform the pointcloud to the ego vehicle
                lidar_points = LidarPointCloud(lidar_points.T)
                lidar_points.rotate(
                    Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
                lidar_points.translate(
                    np.array(lidar_calibrated_sensor['translation']))
                
                # Second step: transform from ego to the global frame.
                lidar_points.rotate(Quaternion(
                    lidar_ego_pose['rotation']).rotation_matrix)
                lidar_points.translate(np.array(lidar_ego_pose['translation']))
                lidar_points = (lidar_points.points[:4, :]).T

                sweep_lidar_points.append(lidar_points)

        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            extrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation()

            num_sweeps_per_sample  = len(self.sweeps_idx)+1
            # key frame only with no sweeps
            for sweep_idx, cam_info in enumerate(cam_infos[::num_sweeps_per_sample]):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # transform all coord from sweep to key_frame ego coord
                # sweep sensor to sweep ego
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()

                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])

                extrin_mat = sweepsensor2sweepego.inverse()

                # key frame lidar depth gt and ida_aug
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    # point_depth = self.get_lidar_depth(
                    #     sweep_lidar_points[sweep_idx], img,
                    #     lidar_infos[sweep_idx], cam_info[cam])
                    point_depth = self.get_lidar_depth_multiframe(
                        sweep_lidar_points[sweep_idx*num_sweeps_per_sample:(sweep_idx+1)*num_sweeps_per_sample],
                        img, cam_info[cam])

                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)

                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                           self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                extrin_mats.append(extrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])

            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_extrin_mats.append(torch.stack(extrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(
                sensor2sensor_mats))  # key sensor to cur sensor
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )
        # sweep_extrin_mats_tensor = torch.stack(
        #     sweep_extrin_mats).permute(1, 0, 2, 3)

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        return ret_list


def main():
    test_dataset = MultiNuscDataset(ida_aug_conf=ida_aug_conf,
                                    bda_aug_conf=bda_aug_conf,
                                    classes=CLASSES,
                                    data_root='/home/yhzn/nuScenes/nuScenes-mini/',
                                    info_paths='/home/yhzn/nuScenes/nuScenes-mini/nuscenes_infos_mini.pkl',
                                    is_train=False,
                                    use_cbgs=False,
                                    img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                                                  img_std=[
                                                      58.395, 57.12, 57.375],
                                                  to_rgb=True),
                                    num_sweeps=1,
                                    sweep_idxes=[0, 1, 2],
                                    key_idxes=list(),
                                    return_depth=True,  # return depth
                                    use_fusion=False)
    data = test_dataset.__getitem__(100)

    imgs = data[0].cpu().numpy()
    depth_maps = data[-1].cpu().numpy()
    img = imgs[0][4].transpose(1, 2, 0)

    depth = np.repeat(depth_maps[0][4][..., None], 3, axis=-1)

    blend_img = img*0.7/255.0+depth*0.3
    plt.axis('off')
    plt.imshow(blend_img)
    plt.savefig("multiframe_pc.png")
    print(depth.shape)


if __name__ == "__main__":
    main()
