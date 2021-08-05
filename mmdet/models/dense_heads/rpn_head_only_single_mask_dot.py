import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin

from mmcv.cnn import (ConvModule, bias_init_with_prob)
from mmdet.core import (multi_apply, images_to_levels)
import matplotlib.pyplot as plt
import numpy as np


@HEADS.register_module()
class RPNHeadOnlySingleMaskDot(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, dilations=[1, 6, 12, 18],
                 loss_mask=dict(type='FocalLoss'),
                 return_lvls_mask=False,
                 **kwargs):
        self.dilations = dilations
        super(RPNHeadOnlySingleMaskDot, self).__init__(1, in_channels, **kwargs)
        self.return_lvls_mask = return_lvls_mask
        self.loss_mask = build_loss(loss_mask)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.mask_head_convs = nn.ModuleList()

        # 添加的ASPP模块
        for dilation in self.dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            in_channels = (self.in_channels
                           if dilation == 0 else self.feat_channels)
            mask_conv = ConvModule(
                in_channels,
                self.feat_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.mask_head_convs.append(mask_conv)

        self.aspp_out = ConvModule(
            self.feat_channels * len(self.dilations),  # 256*4
            self.feat_channels,
            kernel_size=1,
            bias=True)

        self.mask_pred_conv = nn.Conv2d(self.feat_channels, 1, kernel_size=1)  ##

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

        bias_cls = bias_init_with_prob(0.01)  # 并不能完全理解为什么要通过给定的可能值去设定偏差
        # normal_init(self.seg_fea_conv, std=0.01)
        normal_init(self.mask_pred_conv, std=0.01, bias=bias_cls)

    def forward_single(self, x, mask_):
        """Forward feature map of a single scale level."""
        x = x.mul(mask_.sigmoid())
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        # P2-style, stride = 4
        mask_x = feats[0] + F.interpolate(feats[1], scale_factor=2, mode='bilinear', align_corners=True) + \
                 F.interpolate(feats[2], scale_factor=4, mode='bilinear', align_corners=True) + \
                 F.interpolate(feats[3], scale_factor=8, mode='bilinear', align_corners=True) + \
                 F.interpolate(feats[4], scale_factor=16, mode='bilinear', align_corners=True)  ## [B, C, H, W], P2-style, stride=4
        # 将5个输出通道的feature map进行整合成p2

        aspp_feas = []
        for conv in self.mask_head_convs:
            aspp_feas.append(conv(mask_x))

        x = torch.cat(aspp_feas, dim=1)  # [B, C*4, H, W]
        #TOD : 进行调试验证
        x = self.aspp_out(x)  # [B, C, H, W]
        mask_pred = self.mask_pred_conv(x)  # [B, 1, H, W], P2-style, stride=4
        # seg_fea = self.seg_fea_conv(x)  # [B, C, H, W], P2-style, stride=4
        mask_lvls = [mask_pred]  ## 4
        mask_lvls.append(F.max_pool2d(mask_lvls[-1], 1, stride=2))  # 8
        mask_lvls.append(F.max_pool2d(mask_lvls[-1], 1, stride=2))  # 16
        mask_lvls.append(F.max_pool2d(mask_lvls[-1], 1, stride=2))  # 32
        mask_lvls.append(F.max_pool2d(mask_lvls[-1], 1, stride=2))  # 64

        rpn_cls_score, rpn_bbox_pred = multi_apply(self.forward_single, feats, mask_lvls)
        # 得到的结果是list

        if self.return_lvls_mask:
            return rpn_cls_score, rpn_bbox_pred, mask_pred, mask_lvls[:4]  ## P2~P5
        else:
            return rpn_cls_score, rpn_bbox_pred, mask_pred, None

    def loss_mask_func(self, mask_pred, mask_target):
        mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='bilinear', align_corners=True)
        # mask_pred shape([B, 1, 1024, 1024])
        # mask_target shape([B, 1, 1024, 1024])
        #
        # plt.imshow(mask_pred[0][0].cpu().detach().numpy(), cmap='binary')
        # plt.savefig("4.png")
        mask_pred = mask_pred.permute(0, 2, 3, 1).reshape(-1)  ## [B*H*W]
        mask_target = mask_target.permute(0, 2, 3, 1).reshape(-1).type(torch.long)  ## [B*H*W]
        # mask_weight = mask_weight.permute(0, 2, 3, 1).reshape(-1)  ## [B*H*W]
        # assert mask_pred.numel() == mask_target.numel() == mask_weight.numel()
        assert mask_pred.numel() == mask_target.numel()
        # ind = torch.nonzero(mask_target > -1)
        # mask_pred = mask_pred[ind] ## [X]
        # mask_target = mask_target[ind] ## [X]
        num_ = mask_target.numel()


        # TODO:进行调试
        loss_mask = self.loss_mask(mask_pred.unsqueeze_(1),  # [X,1]
                                   mask_target,  # [X], (0,1)
                                   avg_factor=num_)
        return loss_mask

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             mask_preds,
             gt_bboxes,
             gt_masks,
             img_metas,
             gt_labels=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # cls：channel = num_classes * anchors; list[] five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        # box_pred: channel = anchors * 4; list[] five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # both len list is batch_size;
        # and anchor_list doesn't have sth. with images each [Num_anchors, 4]
        # etc:anchor_list[0]=[[Num_anchors, 4],[],..numlvls]
        # TODO : valid
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # self.sampling主要根据sampling loss_cls的方式来决定
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # mask_targets = gt_masks.float()
        # TODO 这里缺少mask weight，但我个人目前认为影响或许不是很大
        mask_targets = torch.stack([1-gt_masks[i].to_tensor(torch.float32, device) for i in range(len(gt_masks))], dim=0)
        # 原文这里使用的是get mask
        # gt_masks是list

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        # concat_anchor_list [[num_anchors,4],[num_anchors,4],...(batch_size)]
        # all_anchor_list [[batchsize, all_anchors_in_a_lvl, 4],[]...(numlvls)]
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        loss_mask = self.loss_mask_func(mask_pred=mask_preds,  # [B, 1, H, W]
                                          mask_target=mask_targets)  # [B, 1, H, W])
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox, loss_rpn_mask=loss_mask)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        # cls_score_list  [[num_base_achors, H, W],..(nun_lvls)]
        # bbox_pred_list  [[4*num_base_achors, H, W],..(nun_lvls)]
        # cfg指的是proposal_cfg
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)  # [num_base_achors, H, W] --> [H, W, num_base_achors]
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # [4*num_base_achors, H, W] --> [H, W, 4*num_base_achors] --> [H*W*num_base_achors, 4]
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                # 我现在越来越能理解dense是什么意思了
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)  # [[2000],...(num_lvls)] --> dim:1
        anchors = torch.cat(mlvl_valid_anchors)  # [[2000, 4], ..(num_lvls)] --> [num, 4]
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)  # [num, 4]
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        # dets[n, 5(x1, y1, x2, y2, scores)]
        # keep index
        return dets[:cfg.nms_post]

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_masks=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        rpn_cls_score, rpn_bbox_pred, rpn_mask_pred, mask_lvls = self(x)  # 自己调用自己
        # outs --> tuple([[batchsize, C, H, W],..(num_lvls)],..(outputs_num_branch))
        if gt_labels is None:
            loss_inputs = (rpn_cls_score, rpn_bbox_pred, rpn_mask_pred) + (gt_bboxes, gt_masks, img_metas)
        else:
            loss_inputs = (rpn_cls_score, rpn_bbox_pred, rpn_mask_pred) + (gt_bboxes, gt_masks, img_metas, gt_labels)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        outs = (rpn_cls_score, rpn_bbox_pred)
        if proposal_cfg is None:
            return losses, mask_lvls
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            # [[n, 5],...(num_images)]
            return losses, proposal_list, mask_lvls

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image.
        """
        rpn_cls_score, rpn_bbox_pred, _, _ = self(x)
        rpn_outs = (rpn_cls_score, rpn_bbox_pred)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        return proposal_list




