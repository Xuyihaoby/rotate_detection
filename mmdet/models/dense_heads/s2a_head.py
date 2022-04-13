# MODIFIED FROM https://github.com/csuhan/s2anet
import torch
import torch.nn as nn

from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms_r, unmap)
from mmdet.ops.orn import ORConv2d, RotationInvariantPooling
from ..builder import HEADS, build_loss
import numpy as np


@HEADS.register_module()
class S2ANetHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_conv=2,
                 with_ornconv=True,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='RAnchorGenerator',
                     scales=[4],
                     ratios=[1.0],
                     strides=[8, 16, 32, 64, 128],
                     angles=[0.]),
                 bbox_coder=dict(
                     type='DeltaRXYWHThetaBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0, 0.),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_fam_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_fam_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_odm_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_odm_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(S2ANetHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_conv = stacked_conv
        self.with_orconv = with_ornconv
        self.anchor_strides = anchor_generator['strides']

        self.use_sigmoid_cls = loss_odm_cls.get('use_sigmoid', False)
        self.odm_sampling = loss_odm_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]

        self.fam_sampling = loss_fam_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]

        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_fam_cls = build_loss(loss_fam_cls)
        self.loss_fam_bbox = build_loss(loss_fam_bbox)
        self.loss_odm_cls = build_loss(loss_odm_cls)
        self.loss_odm_bbox = build_loss(loss_odm_bbox)

        self.train_fam_cfg = train_cfg.fam_cfg if hasattr(train_cfg, 'fam_cfg') else None
        self.train_odm_cfg = train_cfg.odm_cfg if hasattr(train_cfg, 'odm_cfg') else None
        self.test_cfg = test_cfg

        if self.train_fam_cfg:
            self.fam_assigner = build_assigner(self.train_fam_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.fam_sampling and hasattr(self.train_fam_cfg, 'sampler'):
                sampler_fam_cfg = self.train_fam_cfg.sampler
            else:
                sampler_fam_cfg = dict(type='PseudoSampler')
            self.sampler_fam = build_sampler(sampler_fam_cfg, context=self)

        if self.train_odm_cfg:
            self.odm_assigner = build_assigner(self.train_odm_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.odm_sampling and hasattr(self.train_odm_cfg, 'sampler'):
                sampler_odm_cfg = self.train_odm_cfg.sampler
            else:
                sampler_odm_cfg = dict(type='PseudoSampler')
            self.sampler_odm = build_sampler(sampler_odm_cfg, context=self)
        self.fp16_enabled = False

        self.training = True

        self.anchor_generator = build_anchor_generator(anchor_generator)

        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]

        self.base_anchors = dict()
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.fam_reg_convs = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        for i in range(self.stacked_conv):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.fam_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.align_conv = AlignConv(
            self.feat_channels, self.feat_channels, kernel_size=3)

        if self.with_orconv:
            self.or_conv = ORConv2d(self.feat_channels, int(
                self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        self.or_pool = RotationInvariantPooling(256, 8)

        self.odm_reg_convs = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        for i in range(self.stacked_conv):
            chn = int(self.feat_channels /
                      8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1
                )
            )
            self.odm_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1
                )
            )
        self.fam_reg = nn.Conv2d(self.feat_channels, 5, 1)
        self.fam_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.odm_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.odm_reg = nn.Conv2d(self.in_channels, 5, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for fam_reg_module in self.fam_reg_convs:
            normal_init(fam_reg_module.conv, std=0.01)
        for fam_cls_module in self.fam_cls_convs:
            normal_init(fam_cls_module.conv, std=0.01)

        for odm_reg_module in self.odm_reg_convs:
            normal_init(odm_reg_module.conv, std=0.01)
        for odm_cls_module in self.odm_cls_convs:
            normal_init(odm_cls_module.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fam_cls, std=0.01, bias=bias_cls)
        normal_init(self.fam_reg, std=0.01)
        normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        normal_init(self.odm_reg, std=0.01)

        self.align_conv.init_weights()

    def forward_single(self, x, stride):
        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat)
        fam_reg_pred = self.fam_reg(fam_reg_feat)

        if self.training:
            fam_cls_feat = x
            for fam_cls_conv in self.fam_cls_convs:
                fam_cls_feat = fam_cls_conv(fam_cls_feat)
            fam_cls_pred = self.fam_cls(fam_cls_feat)
        else:
            fam_cls_pred = None

        # fpn特征图分别经过4个stacked conv预测前后种类和回归的参数
        num_level = self.anchor_strides.index(stride)
        featmap_size = fam_reg_pred.shape[-2:]  # (batch_size, 5, h, w)
        if (num_level, featmap_size) in self.base_anchors:
            init_anchors = self.base_anchors[(num_level, featmap_size)]
        else:
            device = fam_reg_pred.device
            init_anchors = self.anchor_generator.single_level_grid_anchors(
                self.anchor_generator.base_anchors[num_level].to(device),
                featmap_size,
                self.anchor_generator.strides[num_level],
                device=device)
            self.base_anchors[(num_level, featmap_size)] = init_anchors
            # 这里其实可以优化一下

        refine_anchor = self.bbox_decode(fam_reg_pred.detach(),
                                         init_anchors)
        # 将预测的结果和anchor进行解码操作

        align_feat = self.align_conv(x, refine_anchor.clone(), stride)
        # 重新进行采样之后，进行形变卷积

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat

        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)

        odm_cls_score = self.odm_cls(odm_cls_feat)
        odm_reg_pred = self.odm_reg(odm_reg_feat)
        return fam_cls_pred, fam_reg_pred, refine_anchor, odm_cls_score, odm_reg_pred

    def bbox_decode(
            self,
            bbox_preds,
            anchors):
        num_imgs, _, H, W = bbox_preds.shape
        bboxes_list = []
        for img_id in range(num_imgs):
            bbox_pred = bbox_preds[img_id]
            # bbox_pred.shape=[5,H,W]
            bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            bboxes = self.bbox_coder.decode(
                anchors, bbox_delta, wh_ratio_clip=1e-6)
            bboxes = bboxes.reshape(H, W, 5)
            bboxes_list.append(bboxes)
        return torch.stack(bboxes_list, dim=0)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.anchor_strides)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)  # get anchors in images according to feature point
        # list[5] each tensor [G, 4]
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True,
                            kind='fam'):
        if kind == 'fam':
            inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                               img_meta['img_shape'][:2],
                                               self.train_fam_cfg.allowed_border)
        elif kind == 'odm':
            inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                               img_meta['img_shape'][:2],
                                               self.train_odm_cfg.allowed_border)
        # check whether the anchor inside the border
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.fam_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.fam_sampling else gt_labels)
        sampling_result = self.sampler_fam.sample(assign_result, anchors,
                                                  gt_bboxes)
        # TODO: godeeper
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if kind == 'fam':
                if self.train_fam_cfg.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.train_fam_cfg.pos_weight
            elif kind == 'odm':
                if self.train_odm_cfg.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.train_odm_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        # 由于产生的一些不合格的anchor在该方法的开头已经被去掉，为了保证anchor的总数所以要在对应的结果后加上0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False,
                    kind='fam'):
        # list(list(len = scales)[num, 5])
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        #  [[[],[],[],..(num_lvls)],[[],[],..(numlvls)],..(batch_size)] --> [[],[],..(batch_size)]
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
            kind=kind)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        # [[[], [],..(batch_size)], [[], [],..(batch_size)], ..(numlvls)]
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_fam_single(self, fam_cls_score, fam_bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)  # [batch_size,num_anchors_in_a_lvl] --> [batch_size*num_anchors_in_a_lvl]
        label_weights = label_weights.reshape(-1)

        cls_score = fam_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        # [batchszie, channel, H, W] --> [batchszie, H, W, channel] --> [., c]
        loss_fam_cls = self.loss_fam_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)  # [batchsize, num_bboxes, 4]
        bbox_weights = bbox_weights.reshape(-1, 5)  # [batchsize, num_bboxes, 4]
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        # [batch_size, 4*num_base_anchors, H, W] --> [batch_size, H, W, 4*num_base_anchors] --> [N, 4]
        # TODO: reshape, self.reg_decoded_bbox
        # 每一层有多少anchors，同样会预测多少bbox
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 5)
            fam_bbox_pred = self.bbox_coder.decode(anchors, fam_bbox_pred)
        loss_fam_bbox = self.loss_fam_bbox(
            fam_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_fam_cls, loss_fam_bbox

    def loss_odm_single(self, odm_cls_score, odm_bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)  # [batch_size,num_anchors_in_a_lvl] --> [batch_size*num_anchors_in_a_lvl]
        label_weights = label_weights.reshape(-1)

        cls_score = odm_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        # [batchszie, channel, H, W] --> [batchszie, H, W, channel] --> [., c]
        loss_odm_cls = self.loss_odm_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)  # [batchsize, num_bboxes, 5]
        bbox_weights = bbox_weights.reshape(-1, 5)  # [batchsize, num_bboxes, 5]
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        # [batch_size, 5*num_base_anchors, H, W] --> [batch_size, H, W, 5*num_base_anchors] --> [N, 5]
        # TODO: reshape, self.reg_decoded_bbox
        # 每一层有多少anchors，同样会预测多少bbox
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 5)
            odm_bbox_pred = self.bbox_coder.decode(anchors, odm_bbox_pred)
        loss_odm_bbox = self.loss_odm_bbox(
            odm_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_odm_cls, loss_odm_bbox

    def get_refine_anchors(self,
                           featmap_sizes,
                           refine_anchors,
                           img_metas,
                           is_train=True,
                           device='cuda'):
        # len(refine_anchor) == 5
        num_levels = len(featmap_sizes)

        refine_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
                mlvl_refine_anchors.append(refine_anchor)
            refine_anchors_list.append(mlvl_refine_anchors)
        # len(refine_anchors_list) == batch
        # len(refine_anchors_list[0]) == num_lvls

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = self.anchor_generator.valid_flags(
                    featmap_sizes, img_meta['pad_shape'], device)
                valid_flag_list.append(multi_level_flags)
        return refine_anchors_list, valid_flag_list

    @force_fp32(apply_to=('fam_cls_pred', 'fam_reg_pred', 'odm_cls_score', 'odm_reg_pred'))
    def loss(self,
             fam_cls_pred,
             fam_reg_pred,
             refine_anchor,
             odm_cls_score,
             odm_reg_pred,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # cls：channel = num_classes * anchors; list[] five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        # box_pred: channel = anchors * 4;five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_score]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = odm_cls_score[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # both len list is batch_size;
        # and anchor_list doesn't have sth. with images each [Num_anchors, 4]
        # etc:anchor_list[0]=[[Num_anchors, 5],[],..numlvls]

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            kind='fam')
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # self.sampling主要根据sampling loss_cls的方式来决定
        num_total_samples = (
            num_total_pos + num_total_neg if self.fam_sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        # concat_anchor_list [[num_anchors,5],[num_anchors,5],...(batch_size)]
        # all_anchor_list [[batchsize, all_anchors_in_a_lvl, 5],[]...(numlvls)]
        losses_fam_cls, losses_fam_bbox = multi_apply(
            self.loss_fam_single,
            fam_cls_pred,
            fam_reg_pred,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        # 一阶段和原本铺的anchor进行分配之后预测进行loss计算
        # 二阶段将预测的结果和anchor进行重新refine 之后第二次检测

        # ***********************stage2 ************************************#
        # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchor, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            kind='odm')
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # self.sampling主要根据sampling loss_cls的方式来决定
        num_total_samples = (
            num_total_pos + num_total_neg if self.odm_sampling else num_total_pos)

        num_level_anchors = [anchors.size(0)
                             for anchors in refine_anchors_list[0]]

        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(torch.cat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_odm_cls, losses_odm_bbox = multi_apply(
            self.loss_odm_single,
            odm_cls_score,
            odm_reg_pred,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_fam_cls=losses_fam_cls, loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls, loss_odm_bbox=losses_odm_bbox)

    @force_fp32(apply_to=(
            'fam_cls_scores',
            'fam_bbox_preds',
            'odm_cls_scores',
            'odm_bbox_preds'))
    def get_bboxes(self,
                   fam_cls_pred,
                   fam_reg_pred,
                   refine_anchor,
                   odm_cls_score,
                   odm_reg_pred,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(odm_cls_score) == len(odm_reg_pred)
        num_levels = len(odm_cls_score)

        device = odm_cls_score[0].device
        featmap_sizes = [odm_cls_score[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        refine_anchors = self.get_refine_anchors(
            featmap_sizes, refine_anchor, img_metas, is_train=False, device=device)

        # [[anchors_num, 5],..(num_lvls)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                odm_cls_score[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                odm_reg_pred[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    refine_anchors[0][0], img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    refine_anchors[0][0], img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            # proposals [n, 5]
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        # 这里的cfg时proposal cfg
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_r(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def aug_test(self, feats, img_metas, rescale=False):
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        # outs --> tuple([[batchsize, C, H, W],..(num_lvls)],..(outputs_num_branch))
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            # [[n, 5],...(num_images)]
            return losses, proposal_list


class AlignConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2,
                                        deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        # 用于得到形变卷积所需要的偏移量
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        # 得到的应该是卷积核核每个位置相对于卷积核中心点的偏移增量

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy
        # xc, xx均是一维的，利用python的广播机制进行计算得到对应点的横坐标
        # 应该有更简单的方式得到横纵坐标

        # get sampling locations of anchors
        # 与ga不同的地方在于他们是自己算的
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        # 这里计算的是kernel的宽高和anchor的宽高的比， 之后再利用该宽高计算旋转边框的偏移
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # theta为负才可以在左手系的坐标中为逆时针

        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        # (18, feat_h, feat_w)
        return offset

    def forward(self, x, anchors, stride):
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor))
        return x
