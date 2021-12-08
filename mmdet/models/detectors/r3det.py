import torch.nn as nn
import torch
import numpy as np

from mmdet.core import rbbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.ops.fr import FeatureRefineModule
import mmcv
import warnings
from mmdet.core.visualization import imshow_det_rbboxes


@DETECTORS.register_module()
class R3Det(BaseDetector):
    """
    Rotated Refinement RetinaNet
    """

    def __init__(self,
                 num_refine_stages,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 frm_cfgs=None,
                 refine_heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(R3Det, self).__init__()
        self.num_refine_stages = num_refine_stages
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            bbox_head.update(train_cfg=train_cfg['s0'])
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.feat_refine_module = nn.ModuleList()
        self.refine_head = nn.ModuleList()
        for i, (frm_cfg, refine_head) in enumerate(zip(frm_cfgs, refine_heads)):
            self.feat_refine_module.append(FeatureRefineModule(**frm_cfg))
            if train_cfg is not None:
                refine_head.update(train_cfg=train_cfg['sr'][i])
            refine_head.update(test_cfg=test_cfg)
            self.refine_head.append(build_head(refine_head))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(R3Det, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        for i in range(self.num_refine_stages):
            self.feat_refine_module[i].init_weights()
            self.refine_head[i].init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        losses = dict()
        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        train_cfg = self.train_cfg['s0']
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses['s0.{}'.format(name)] = value

        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]
            train_cfg = self.train_cfg['sr'][i]

            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = self.refine_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses['sr{}.{}'.format(i, name)] = (
                    [v * lw for v in value] if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        return losses

    def simple_test(self,
                    img,
                    img_meta,
                    rescale=False):
        if 'tile_offset' in img_meta[0]:
            # using tile-cropped TTA. force using aug_test instead of simple_test
            return self.aug_test(imgs=[img], img_metas=[img_meta], rescale=True)

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.refine_head[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError
    #     AUG_BS = 8
    #     assert rescale, '''while r3det uses overlapped cropping augmentation by default,
    #     the result should be rescaled to input images sizes to simplify the test pipeline'''
    #     if 'tile_offset' in img_metas[0][0]:
    #         assert imgs[0].size(0) == 1, '''when using cropped tiles augmentation,
    #         image batch size must be set to 1'''
    #         aug_det_bboxes, aug_det_labels = [], []
    #         num_augs = len(imgs)
    #         for idx in range(0, num_augs, AUG_BS):
    #             img = imgs[idx:idx + AUG_BS]
    #             img_meta = img_metas[idx:idx + AUG_BS]
    #             act_num_augs = len(img_meta)
    #             img = torch.cat(img, dim=0)
    #             img_meta = sum(img_meta, [])
    #             # for img, img_meta in zip(imgs, img_metas):
    #             x = self.extract_feat(img)
    #             outs = self.bbox_head(x)
    #             rois = self.bbox_head.filter_bboxes(*outs)
    #             # rois: list(indexed by images) of list(indexed by levels)
    #             det_bbox_bs = [[] for _ in range(act_num_augs)]
    #             det_label_bs = [[] for _ in range(act_num_augs)]
    #             for i in range(self.num_refine_stages):
    #                 x_refine = self.feat_refine_module[i](x, rois)
    #                 outs = self.refine_head[i](x_refine)
    #                 if i + 1 in range(self.num_refine_stages):
    #                     rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
    #
    #                 bbox_inputs = outs + (img_meta, self.test_cfg, False)
    #                 bbox_bs = self.refine_head[i].get_bboxes(*bbox_inputs, rois=rois)
    #                 # [(rbbox_aug0, class_aug0), (rbbox_aug1, class_aug1), (rbbox_aug2, class_aug2), ...]
    #                 for j in range(act_num_augs):
    #                     det_bbox_bs[j].append(bbox_bs[j][0])
    #                     det_label_bs[j].append(bbox_bs[j][1])
    #
    #             for j in range(act_num_augs):
    #                 det_bbox_bs[j] = torch.cat(det_bbox_bs[j])
    #                 det_label_bs[j] = torch.cat(det_label_bs[j])
    #
    #             aug_det_bboxes += det_bbox_bs
    #             aug_det_labels += det_label_bs
    #
    #         aug_det_bboxes, aug_det_labels = merge_tiles_aug_rbboxes(
    #             aug_det_bboxes,
    #             aug_det_labels,
    #             img_metas,
    #             self.test_cfg.merge_cfg,
    #             self.CLASSES)
    #
    #         return rbbox2result(aug_det_bboxes, aug_det_labels, self.refine_head[-1].num_classes)
    #
    #     else:
    #         raise NotImplementedError

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
