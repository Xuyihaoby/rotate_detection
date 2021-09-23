import torch
import torch.nn as nn
import warnings
import mmcv
import numpy as np
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .rfaster_rcnn import RFasterRCNN

from mmdet.core.visualization import imshow_det_rbboxes, imshow_det_bboxes


@DETECTORS.register_module()
class OientedRCNN(RFasterRCNN):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 obb=False,
                 submission=False):
        super(OientedRCNN, self).__init__(backbone=backbone,
                                          neck=neck,
                                          rpn_head=rpn_head,
                                          roi_head=roi_head,
                                          train_cfg=train_cfg,
                                          test_cfg=test_cfg,
                                          pretrained=pretrained,
                                          obb=obb,
                                          submission=submission)

    def forward_train(self,
                      img,
                      img_metas,
                      hor_gt_bboxes,
                      gt_labels,
                      hor_gt_bboxes_ignore=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                hor_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 hor_gt_bboxes, gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
