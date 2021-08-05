import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.core import multi_apply, multiclass_nms, multiclass_nms_r
import torch.nn.functional as F

from mmdet.models.losses import accuracy
from mmcv.runner import force_fp32

from mmdet.core import build_bbox_coder

@HEADS.register_module()
class ConvFCMultiLvlsWithOriginalImageSingleMaskBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 bbox_coder_r=dict(
                     type='DeltaXYWHBThetaBoxCoder',
                     target_means=[0., 0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2, 0.2]),
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCMultiLvlsWithOriginalImageSingleMaskBBoxHead, self).__init__(*args, **kwargs)
        self.bbox_coder_r = build_bbox_coder(bbox_coder_r)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            # rotate
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
            # horizon
            self.fc_cls_h = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            # horizon
            out_dim_reg_h = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            # [dx, dy, dw, dh]
            self.fc_reg_h = nn.Linear(self.reg_last_dim, out_dim_reg_h)

            # rotate
            out_dim_reg = (5 if self.reg_class_agnostic else 5 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
            # [dx, dy, dw, dh, dtheta]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        # super(RConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.with_cls:
            nn.init.normal_(self.fc_cls_h.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls_h.bias, 0)
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg_h.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg_h.bias, 0)
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            # [batch_size * rpn_out_c, bbox_head_in_channel, 7, 7] -->
            # [batch_size * rpn_out_c, bbox_head_in_channel*7*7]
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # horizon
        cls_score_h = self.fc_cls_h(x_cls) if self.with_cls else None  # [N, 16]
        bbox_pred_h = self.fc_reg_h(x_reg) if self.with_reg else None  # [N, 15*4]
        # rotate
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score_h, bbox_pred_h, cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_hor_gt_bboxes,
                           pos_gt_labels, pos_gt_bboxes, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
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
                bboxes /= scale_factor   # TODO
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
                                                    cfg.max_per_img)


            det_bboxes, det_labels = multiclass_nms_r(bboxes, scores,
                                                    cfg.score_thr, cfg.nms_r,
                                                    cfg.max_per_img)
            # det_bboxes is [n, 6]

            return det_bboxes_h, det_labels_h, det_bboxes, det_labels




@HEADS.register_module()
class MultiLvlsWithOriginalImageSingleMaskShared2FCBBoxHead(ConvFCMultiLvlsWithOriginalImageSingleMaskBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(MultiLvlsWithOriginalImageSingleMaskShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class MultiLvlsWithOriginalImageSingleMaskShared4Conv1FCBBoxHead(ConvFCMultiLvlsWithOriginalImageSingleMaskBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(MultiLvlsWithOriginalImageSingleMaskShared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
