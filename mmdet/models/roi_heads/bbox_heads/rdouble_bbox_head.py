import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init
from mmdet.core import multi_apply, multiclass_nms, multiclass_nms_r
from mmdet.models.backbones.resnet import Bottleneck
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from .bbox_head import BBoxHead
from mmcv.runner import auto_fp16, force_fp32
import torch.nn.functional as F

from mmdet.core import build_bbox_coder


class BasicResBlock(nn.Module):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module()
class RDoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

    .. code-block:: none

                                          /-> cls
                      /-> shared convs ->
                                          \-> reg
        roi features
                                          /-> cls
                      \-> shared fc    ->
                                          \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 bbox_coder_r=dict(
                     type='DeltaXYWHBThetaBoxCoder',
                     target_means=[0., 0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2, 0.2]),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(RDoubleConvFCBBoxHead, self).__init__(**kwargs)
        self.bbox_coder_r = build_bbox_coder(bbox_coder_r)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()

        # rotate
        out_dim_reg = 5 if self.reg_class_agnostic else 5 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)

        # hor
        out_dim_reg_h = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg_h = nn.Linear(self.conv_out_channels, out_dim_reg_h)

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.fc_cls_h = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers."""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

        normal_init(self.fc_cls_h, std=0.01)
        normal_init(self.fc_reg_h, std=0.001)

        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)
        bbox_pred_h = self.fc_reg_h(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        cls_score = self.fc_cls(x_fc)
        cls_score_h = self.fc_cls_h(x_fc)

        return cls_score_h, bbox_pred_h, cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_hor_gt_bboxes,
                           pos_gt_labels, pos_gt_bboxes, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        # horizon
        hor_bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        hor_bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        # rotate
        bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            if not self.reg_decoded_bbox:
                pos_hor_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_hor_gt_bboxes)
                pos_bbox_targets = self.bbox_coder_r.encode(pos_bboxes, pos_gt_bboxes)
                # [N1, 5]
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_hor_bbox_targets = pos_hor_gt_bboxes
            hor_bbox_targets[:num_pos, :] = pos_hor_bbox_targets
            hor_bbox_weights[:num_pos, :] = 1
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, hor_bbox_targets, hor_bbox_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        # 为了便于理解将下列列表每个元素的维度表上
        # pos_gt_bboxes[0].size() >>> torch.Size([22, 5])
        # pos_bboxes_list[0].size() >>> torch.Size([22, 4])
        # gt_bboxes[0].size() >>> torch.Size([16, 5])
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 4]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 4]
        pos_hor_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 4]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_bboxes = [gt_bboxes[i][res.pos_assigned_gt_inds, :] for i, res in enumerate(sampling_results)]
        # list( len = batchsize ) each is [N, 5]

        labels, label_weights, hor_bbox_targets, hor_bbox_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_hor_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_bboxes,
            cfg=rcnn_train_cfg)
        # labels >>> [[num_samples],..(batchsize)]
        if concat:
            # 将list里的tensor合并
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            # horizon
            hor_bbox_targets = torch.cat(hor_bbox_targets, 0)
            hor_bbox_weights = torch.cat(hor_bbox_weights, 0)
            # rotate
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, hor_bbox_targets, hor_bbox_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'cls_score_h', 'bbox_pred_h'))
    def loss(self,
             cls_score,
             bbox_pred,
             cls_score_h,
             bbox_pred_h,
             rois,
             labels,
             label_weights,
             hor_bbox_targets,
             hor_bbox_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score_h is not None and cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # rotate
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
            # horizon
            if cls_score_h.numel() > 0:
                losses['loss_cls_h'] = self.loss_cls(
                    cls_score_h,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc_h'] = accuracy(cls_score_h, labels)

        if bbox_pred_h is not None and bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred_h = self.bbox_coder.decode(rois[:, 1:], bbox_pred_h)
                if self.reg_class_agnostic:
                    pos_bbox_pred_h = bbox_pred_h.view(
                        bbox_pred_h.size(0), 4)[pos_inds.type(torch.bool)]
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred_h.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    # bbox_pred torch.Size([1024, 75])
                    # bbox_pred_h torch.Size([1024, 60])
                    # horizon
                    pos_bbox_pred_h = bbox_pred_h.view(
                        bbox_pred_h.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    # rotate
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox_h'] = self.loss_bbox(
                    pos_bbox_pred_h,
                    hor_bbox_targets[pos_inds.type(torch.bool)],
                    hor_bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=hor_bbox_targets.size(0),
                    reduction_override=reduction_override)
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox_h'] = bbox_pred_h[pos_inds].sum()
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'cls_score_h', 'bbox_pred_h'))
    def get_bboxes(self,
                   rois,
                   cls_score_h,
                   bbox_pred_h,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if isinstance(cls_score_h, list):
            cls_score_h = sum(cls_score_h) / float(len(cls_score_h))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        scores_h = F.softmax(cls_score_h, dim=1) if cls_score_h is not None else None

        # rois --- [batchinds, x1, y1, x2, y2]
        # rotate
        if bbox_pred is not None:
            bboxes = self.bbox_coder_r.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes_h = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes_h[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes_h[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor  # TODO do some improvement
            else:
                scale_factor_r = bboxes.new_tensor(scale_factor)  # array ---> tensor
                scale_factor_r = torch.cat([scale_factor_r, scale_factor_r.new_ones(1)])
                bboxes = (bboxes.view(bboxes.size(0), -1, 5) /
                          scale_factor_r).view(bboxes.size()[0], -1)
                # [1000, 75] --> [1000, 15, 5] 把scale_factor除完之后再送回去 --> [1000, 75]

        # horizon
        if bbox_pred_h is not None:
            bboxes_h = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred_h, max_shape=img_shape)
        else:
            bboxes_h = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes_h[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes_h[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes_h.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes_h /= scale_factor
            else:
                scale_factor = bboxes_h.new_tensor(scale_factor)  # array ---> tensor
                bboxes_h = (bboxes_h.view(bboxes_h.size(0), -1, 4) /
                            scale_factor).view(bboxes_h.size()[0], -1)
        if cfg is None:
            return bboxes_h, scores_h, bboxes, scores
        else:
            det_bboxes_h, det_labels_h = multiclass_nms(bboxes_h, scores_h,
                                                        cfg.score_thr, cfg.nms_h,
                                                        cfg.max_per_img_h)

            det_bboxes, det_labels = multiclass_nms_r(bboxes, scores,
                                                      cfg.score_thr, cfg.nms_r,
                                                      cfg.max_per_img)

            # det_bboxes is [n, 6]

            return det_bboxes_h, det_labels_h, det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        # 这里其实将正负样本都进行了相应的解码处理
        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]  # 正样本中有些是gt

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_h2rclass(self, rois, label, bbox_pred, img_meta):
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        # 这里其实将正负样本都进行了相应的解码处理
        if rois.size(1) == 4:
            new_rois = self.bbox_coder_r.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder_r.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    @force_fp32(apply_to=('bbox_preds',))
    def refine_rbboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]  # 正样本中有些是gt

            bboxes = self.regress_by_h2rclass(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list


@HEADS.register_module()
class RDoubleOrient2BBoxHead(RDoubleConvFCBBoxHead):
    def __init__(self, **kwargs):
        super(RDoubleOrient2BBoxHead, self).__init__(**kwargs)

    def _get_target_single(self, pos_hor_bboxes, neg_hor_bboxes, pos_bboxes, neg_bboxes, pos_hor_gt_bboxes,
                           pos_gt_labels, pos_gt_bboxes, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        # horizon
        hor_bbox_targets = pos_hor_bboxes.new_zeros(num_samples, 4)
        hor_bbox_weights = pos_hor_bboxes.new_zeros(num_samples, 4)
        # rotate
        bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
        # import pdb
        # pdb.set_trace()
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_hor_bbox_targets = self.bbox_coder.encode(
                    pos_hor_bboxes, pos_hor_gt_bboxes)
                pos_bbox_targets = self.bbox_coder_r.encode(pos_bboxes, pos_gt_bboxes)
                # [N1, 5]
            else:
                pos_hor_bbox_targets = pos_hor_gt_bboxes
            hor_bbox_targets[:num_pos, :] = pos_hor_bbox_targets
            hor_bbox_weights[:num_pos, :] = 1
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, hor_bbox_targets, hor_bbox_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_hor_bboxes_list = [res.pos_hor_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 4]
        neg_hor_bboxes_list = [res.neg_hor_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 4]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 5]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 5]
        pos_hor_gt_bboxes_list = [res.pos_hor_gt_bboxes for res in sampling_results]
        # list( len = batchsize ) each is [N, 4]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_bboxes_list = [gt_bboxes[i][res.pos_assigned_gt_inds, :] for i, res in enumerate(sampling_results)]
        # list( len = batchsize ) each is [N, 5]
        labels, label_weights, hor_bbox_targets, hor_bbox_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_hor_bboxes_list,
            neg_hor_bboxes_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_hor_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_bboxes_list,
            cfg=rcnn_train_cfg)
        # labels >>> [[num_samples],..(batchsize)]
        if concat:
            # 将list里的tensor合并
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            # horizon
            hor_bbox_targets = torch.cat(hor_bbox_targets, 0)
            hor_bbox_weights = torch.cat(hor_bbox_weights, 0)
            # rotate
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, hor_bbox_targets, hor_bbox_weights, bbox_targets, bbox_weights
