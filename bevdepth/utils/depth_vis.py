import os
import refile
from tqdm import trange

def vis_depth_distribution(imgs, depth_preds, depth_gts, dbound, save_path, fb=False):
    """可视化depth pred和depth gt的统计分布

    Args:
        img (Torch.tensor) : [N, C, H, W]
        depth_pred () : [N, D, h, w]
        depth_gt () : [B*N, h, w, D]
    """
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # 预处理 end
    # reshape depth
    N, D, d_H, d_W = depth_preds.shape
    depth_gts = depth_gts.reshape(N, -1, D)
    depth_preds = depth_preds.transpose(0, 2, 3, 1).reshape(N, -1, D)
    for i in range(len(imgs)):
        # process single view
        img = imgs[i]
        depth_pred = depth_preds[i]
        depth_gt = depth_gts[i]
        C, H, W = img.shape
        plt.figure(figsize=(W * 2 / 100, H * 2 / 100))
        # 隐藏坐标轴
        ax = plt.axes()
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        # 处理子图
        for idx in trange(len(depth_pred), desc="[process points]"):
            plt.subplot(d_H, d_W, idx + 1)
            plt.axis("off")
            plt.tight_layout()
            # plt.ylim(0, 0.8)

            # plot depth gt
            depth_gt_point = depth_gt[idx]
            depth_max_index = np.argmax(depth_gt_point)
            x = np.linspace(dbound[0], dbound[1], D)
            x_ = x[depth_max_index]
            # 标记最大值
            plt.axvline(x=x_, ymin=0, ymax=1, color="r")

            # plot depth pred
            depth_pred_point = depth_pred[idx]
            plt.plot(x[:], depth_pred_point[:], color="g")
            depth_max_index = np.argmax(depth_pred_point)
            x_ = x[depth_max_index]
            plt.axvline(x=x_, ymin=0, ymax=1, color="b")
        plt.tight_layout()
        plt.savefig(f"{save_path}depth_distribution_{i}_frame_{i}_fb_{fb}.png", dpi=600)
        print(f"save as {save_path}depth_distribution_{i}_frame_{i}_fb_{fb}.png")
        # combine with img
        img = img.transpose(1, 2, 0)
        depth_curve_all = cv2.imread(f"{save_path}/depth_distribution_{i}_frame_{i}_fb_{fb}.png")
        depth_curve_all = cv2.resize(depth_curve_all, (1408 * 2, 512 * 2))
        img = cv2.resize(img, (1408 * 2, 512 * 2))
        black_depth_curve_all = (
            (depth_curve_all) * np.max((255 - depth_curve_all) // (255), axis=2, keepdims=True)
        ).astype(np.uint8)
        final_img = img.astype(np.int32) + black_depth_curve_all
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{save_path}depth_distribution_{i}_final_frame_{i}_fb_{fb}.png", final_img)
        # refile.smart_save_image(f"{save_path}/depth_distribution_/cam_{i}_final_fb_{fb}_frame_{i}.png", final_img[..., ::-1])
        print(f"save as {save_path}/depth_distribution_{i}_final_frame_{i}_fb_{fb}.png")


choose_index= 56
depth_path = f"/data/vis_imgs/depth2vis/{choose_index}.pkl"
scale_depth_path = f"/data/vis_imgs/depth2vis_scale_depth/{choose_index}.pkl"

save_path = "/data/vis_imgs/"

import mmcv
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

depth_dicts = mmcv.load(depth_path)
scale_depth_dicts = mmcv.load(scale_depth_path)

front_back_imgs = depth_dicts['front_back_imgs'].numpy()
side_imgs = depth_dicts['imgs'].numpy()
fb_depth_gt_map = depth_dicts["fb_depht_gt"].numpy()
side_depth_gt_map = depth_dicts["side_depth_gt"].numpy()
front_back_depths = depth_dicts["fb_depth_pred"].numpy()
left_right_depths = depth_dicts["side_depth_pred"].numpy()

scale_front_back_imgs = scale_depth_dicts['front_back_imgs'].numpy()
scale_side_imgs = scale_depth_dicts['imgs'].numpy()
scale_fb_depth_gt_map = scale_depth_dicts["fb_depht_gt"].numpy()
scale_side_depth_gt_map = scale_depth_dicts["side_depth_gt"].numpy()
scale_front_back_depths = scale_depth_dicts["fb_depth_pred"].numpy()
scale_left_right_depths = scale_depth_dicts["side_depth_pred"].numpy()


img = front_back_imgs[1][1]
scale_img = scale_front_back_imgs[1][1]

img = img.transpose(1, 2, 0).astype(np.int32)
scale_img = scale_img.transpose(1, 2, 0).astype(np.int32)

plt.subplot(211)
plt.axis("off")
plt.imshow(img)
plt.subplot(212)
plt.axis("off")
plt.imshow(scale_img)
plt.show()

fb_depth_gt_map.shape # (6, 16, 44, 160) (B*N, H, W, D)
front_back_depths.shape # (6, 160, 16, 44) (B*N, D, H, W)
B, N, C, H, W = front_back_imgs.shape # (2, 3, 3, 512, 1408)
front_back_imgs = front_back_imgs.reshape((-1, C, H, W))
side_imgs = side_imgs.reshape((-1, C, H, W))

scale_fb_depth_gt_map.shape # (6, 16, 44, 160) (B*N, H, W, D)
scale_front_back_depths.shape # (6, 160, 16, 44) (B*N, D, H, W)
B, N, C, H, W = scale_front_back_imgs.shape # (2, 3, 3, 512, 1408)
scale_front_back_imgs = scale_front_back_imgs.reshape((-1, C, H, W))
scale_side_imgs = scale_side_imgs.reshape((-1, C, H, W))

vis_depth_distribution(side_imgs[:4], left_right_depths[:4], side_depth_gt_map[:4], dbound, f"{save_path}/side/", fb=False)