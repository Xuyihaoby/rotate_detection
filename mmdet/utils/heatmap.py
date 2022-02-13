import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import torch
import time
import torch.nn.functional as F


def showHeatmap(feat, path='/home/xyh/rotate_detection/checkpoints/visual/heat/', img_metas=None, rescale=True):
    os.makedirs(path, exist_ok=True)
    if isinstance(feat, list):
        # size = [_feat.shape[-2] for _feat in feat]
        new_feat_list = [F.interpolate(_feat, size=256, mode='bilinear') for _feat in feat]
        new_feat = torch.cat(new_feat_list)
        new_feat, _ = new_feat.sum(1).max(0)
        new_feat = new_feat.detach().cpu().numpy()
        # flatten_feat = new_feat.flatten()
        max_item = new_feat.max()
        norm_img = None
        norm_img = cv.normalize(new_feat, norm_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        if rescale:
            norm_img = (norm_img * max_item / 255).astype(dtype=np.uint8)
        heat_img = cv.applyColorMap(norm_img, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
        # heat_img = heat_img * max_item / 255
        resize_heatimg = cv.resize(heat_img, (512, 512), interpolation=cv.INTER_AREA)

        ori_img = cv.imread(img_metas[0]['filename'])
        resize_ori_img = cv.resize(ori_img, (512, 512))
        fuse_img = 0.5 * resize_heatimg + 0.5 * resize_ori_img

        cv.imwrite(path + osp.basename(img_metas[0]['filename']), fuse_img)
    else:
        if feat.dim() == 4:
            new_feat = feat.sum(1).squeeze()
            # new_feat = new_feat.detach().cpu().numpy()
            new_feat = new_feat.detach().cpu().numpy()
            norm_img = None
            norm_img = cv.normalize(new_feat, norm_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            heat_img = cv.applyColorMap(norm_img, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
            resize_heatimg = cv.resize(heat_img, (1024, 1024), interpolation=cv.INTER_AREA)
            cv.imwrite(path + str(time.time()) + '.jpg', resize_heatimg)
