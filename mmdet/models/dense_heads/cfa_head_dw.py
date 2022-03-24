import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d
from mmdet.ops import minaerarect

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, levels_to_images,
                        multi_apply, multiclass_nms, multiclass_nms_r, unmap, obb2poly, poly2obb)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.ops.iou import convex_overlaps

from mmdet.utils.heatmap import showHeatmap


@HEADS.register_module()
class CFAHeaddw(AnchorFreeHead):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='RotatedIoULoss', loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='RotatedIoULoss', loss_weight=1.0),
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01,
                 use_cfa=True,
                 topk=6,
                 anti_factor=0.75,
                 **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.use_grid_points = use_grid_points
        self.center_init = center_init

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        # each point relative offset to center

        self.topk = topk
        self.anti_factor = anti_factor
        self.use_cfa = use_cfa
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(
                self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.iou_init = loss_bbox_init['type'] in ['ConvexGIoULoss']
        self.iou_refine = loss_bbox_init['type'] in ['ConvexGIoULoss']

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """

        if self.transform_method == 'direct':
            pts_trans = pts.view(pts.shape[0], -1, *pts.shape[2:]).permute(0, 2, 3, 1)

        elif self.transform_method == 'partial_minmax':
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                              ...]
            pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                              ...]
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                              ...]
            pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                              ...]
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        else:
            raise NotImplementedError
        return pts_trans

    def gen_grid_from_reg(self, reg, previous_boxes):
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        # tensor([[[[0.0000]],
        #          [[0.5000]],
        #          [[1.0000]]]])
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            raise NotImplementedError
            # scale = self.point_base_scale / 2
            # points_init = dcn_base_offset / dcn_base_offset.max() * scale
            # bbox_init = x.new_tensor([-scale, -scale, scale,
            #                           scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            raise NotImplementedError
            # pts_out_init, bbox_out_init = self.gen_grid_from_reg(
            #     pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        temp_clsfeat = self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset))
        cls_out = self.reppoints_cls_out(temp_clsfeat)
        temp_point = self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset))
        pts_out_refine = self.reppoints_pts_refine_out(temp_point)
        if self.use_grid_points:
            raise NotImplementedError
            # pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
            #     pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        if self.training:
            return cls_out, pts_out_init, pts_out_refine
        else:
            return cls_out, pts_out_init, pts_out_refine, cls_feat

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i], device)
            # return point coordinate (x, y, stride) logits of program col first
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        # return list each lvl points coordinate and stride and valid flag
        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points.

        Only used in :class:`MaxIoUAssigner`.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
            # img & lvl channel exchange
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             label_channels=1,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore,
                                        None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)
        gt_inds = assign_result.gt_inds

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, gt_bboxes.shape[-1]])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, gt_bboxes.shape[-1]])
        labels = proposals.new_full((num_valid_proposals,),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            gt_inds = unmap(gt_inds, num_total_proposals,
                            inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, gt_inds)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list, gt_inds_list) = multi_apply(
            self._point_target_single,
            proposals_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            stage=stage,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        pos_inds = []
        pos_gt_index = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = single_labels < self.num_classes
            pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))
            pos_gt_index.append(gt_inds_list[i][pos_mask.nonzero(as_tuple=False).view(-1)])
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg, pos_inds, pos_gt_index)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine, stride,
                    num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine+1)
        # points loss
        # init point
        bbox_gt_init = bbox_gt_init.reshape(-1, 8)

        # bbox_weights_init = bbox_weights_init.reshape(-1, 8)
        bbox_weights_init = bbox_weights_init.mean(dim=2).reshape(-1)
        pos_ind_init = (bbox_weights_init > 0).nonzero(as_tuple=False).reshape(-1)
        if self.iou_init:
            bbox_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        else:
            bbox_pred_init = self.points2bbox(
                pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        pts_pred_init_norm = bbox_pred_init[pos_ind_init]
        bbox_gt_init_norm = bbox_gt_init[pos_ind_init]
        bbox_weights_init_norm = bbox_weights_init[pos_ind_init]

        # refine point
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 8)
        # bbox_weights_refine = bbox_weights_refine.reshape(-1, 8)
        bbox_weights_refine = bbox_weights_refine.mean(dim=2).reshape(-1)
        pos_ind_refine = (bbox_weights_refine > 0).nonzero(as_tuple=False).reshape(-1)
        if self.iou_refine:
            bbox_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        else:
            bbox_pred_refine = self.points2bbox(
                pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        pts_pred_refine_norm = bbox_pred_refine[pos_ind_refine]
        bbox_gt_refine_norm = bbox_gt_refine[pos_ind_refine]
        bbox_weights_refine_norm = bbox_weights_refine[pos_ind_refine]

        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            pts_pred_init_norm / normalize_term,
            bbox_gt_init_norm / normalize_term,
            bbox_weights_init_norm,
            avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(
            pts_pred_refine_norm / normalize_term,
            bbox_gt_refine_norm / normalize_term,
            bbox_weights_refine_norm,
            avg_factor=num_total_samples_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def get_pos_loss(self, cls_scores, pts_coordinate_preds_init, pts_coordinate_preds_refine,
                     labels, label_weights, bbox_gt_refine, bbox_weights_refine, pos_inds_refine):
        pos_scores = cls_scores[pos_inds_refine]
        pos_pts_pred = pts_coordinate_preds_init[pos_inds_refine]
        pos_bbox_gt = bbox_gt_refine[pos_inds_refine]
        pos_label = labels[pos_inds_refine]
        pos_label_weights = label_weights[pos_inds_refine]
        pos_bbox_weights = bbox_weights_refine[pos_inds_refine]
        try:
            loss_cls = self.loss_cls(
                pos_scores,
                pos_label,
                pos_label_weights,
                reduction_override='none'
            )
        except:
            import pdb
            pdb.set_trace()
        pos_bbox_weights = pos_bbox_weights.mean(dim=-1).reshape(-1)
        loss_bbox = self.loss_bbox_refine(
            pos_pts_pred,
            pos_bbox_gt,
            pos_bbox_weights,
            reduction_override='none'
        )

        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_cls + loss_bbox
        # _loss_cls = (loss_cls-loss_cls.min()) / (loss_cls.max() - loss_cls.min())
        # _loss_bbox = (loss_bbox-loss_bbox.min()) / (loss_bbox.max() - loss_bbox.min())
        # import matplotlib.pyplot as plt
        # x = range(_loss_cls.shape[0])
        # plt.plot(x, _loss_cls.cpu().numpy(), label='cls')
        # plt.plot(x, _loss_bbox.cpu().numpy(), label='bbox')
        # # plt.plot(x, (_loss_bbox.cpu().numpy() + _loss_cls.cpu().numpy())*0.5, label='mean')
        # plt.savefig('1.png')
        # pos_loss = loss_bbox + loss_cls
        return pos_loss, loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        if self.transform_method in ['direct']:
            gt_bboxes = [obb2poly(gt_bboxes_i, version='v1') for gt_bboxes_i in gt_bboxes]
            gt_bboxes_ignore = [gt_bboxes_ignore_i.new_zeros(0, 8) for gt_bboxes_ignore_i in gt_bboxes_ignore]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        device = cls_scores[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)
        # 这里得到的中心的坐标是已经映射回原图的pioint坐标
        # center list feature map coordinate
        # center list: list(list(tensor))  batch(num_lvl(tensor))
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        # 将预测的偏差改为点的坐标
        # list num_lvl each shape (bs, num, 18)
        if self.train_cfg.init.assigner['type'] == 'RPointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            candidate_list = []
            for i_img, center in enumerate(center_list):
                bbox = []
                for i_lvl in range(len(pts_preds_init)):
                    pts_trans_init = self.points2bbox(
                        pts_preds_init[i_lvl].detach())
                    pts_shift = pts_trans_init * self.point_strides[i_lvl]
                    bbox_center = center[i_lvl][:, :2].repeat(1, pts_shift.shape[-1] // 2)
                    bbox.append(bbox_center + pts_shift[i_img].reshape(-1, 2 * self.num_points))
                candidate_list.append(bbox)
        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, pos_inds, pos_gt_index) = cls_reg_targets_init

        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        # 将预测的偏差转换成点的坐标
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                # 这里可以思索一下使用何种方式将点转换成bbox
                pts_trans_init = self.points2bbox(
                    pts_preds_init[i_lvl].detach())
                # 这里得到的偏差只是相对偏差，并没有得到绝对坐标的位置关系
                pts_shift = pts_trans_init * self.point_strides[i_lvl]
                bbox_center = center[i_lvl][:, :2].repeat(1, pts_shift.shape[-1] // 2)
                bbox.append(bbox_center + pts_shift[i_img].reshape(-1, 2 * self.num_points))
            bbox_list.append(bbox)

        cls_reg_targets_refine = self.get_targets(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='refine',
            label_channels=label_channels)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine, pos_inds_refine, pos_gt_index_refine) = cls_reg_targets_refine
        # 首先上述几个变量都已经被映射成lvl batch
        # label_list 就是标签 [bs, num]
        # label_weights 标签权重
        # bbox_gt_list_refine 对应featurepoint上的gt
        # candidate_list_refine 预测得到的bbox
        # pos inds 代表的正样本的序列号
        # pos gt index 代表的是正样本所指代的gt标号

        cls_scores_image = levels_to_images(cls_scores)
        pts_coordinate_preds_init_image = levels_to_images(pts_coordinate_preds_init)
        pts_coordinate_preds_refine_image = levels_to_images(pts_coordinate_preds_refine)
        labels_list_image = levels_to_images(labels_list)
        label_weights_list_image = levels_to_images(label_weights_list)
        bbox_gt_list_refine_image = levels_to_images(bbox_gt_list_refine)
        bbox_weights_list_refine_image = levels_to_images(bbox_weights_list_refine)

        num_proposals_each_lvl = [(featmap.size(-1) * featmap.size(-2))
                                  for featmap in cls_scores]
        num_lvls = len(featmap_sizes)
        assert num_lvls == len(pts_coordinate_preds_init)
        with torch.no_grad():
            pos_losses_list, loss_cls, loss_bbox = multi_apply(self.get_pos_loss, cls_scores_image, pts_coordinate_preds_init_image,
                                           pts_coordinate_preds_refine_image,
                                           labels_list_image, label_weights_list_image, bbox_gt_list_refine_image,
                                           bbox_weights_list_refine_image, pos_inds_refine)
            labels_list, label_weights_list, bbox_weights_list_refine, num_pos, pos_normalize_term = \
                multi_apply(self.reassign, pos_losses_list, loss_cls, loss_bbox, labels_list_image, label_weights_list_image,
                            pts_coordinate_preds_init_image, pts_coordinate_preds_refine_image,
                            bbox_weights_list_refine_image, gt_bboxes,
                            pos_inds_refine, pos_gt_index_refine, num_proposals_lvls=num_proposals_each_lvl,
                            num_lvls=num_lvls)
            num_pos = sum(num_pos)
            # assert num_pos!=0

        labels_list = images_to_levels(labels_list, num_proposals_each_lvl)
        label_weights_list = images_to_levels(label_weights_list, num_proposals_each_lvl)
        bbox_weights_list_refine = images_to_levels(bbox_weights_list_refine, num_proposals_each_lvl)
        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_pos)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all

    def reassign(self, pos_losses, loss_cls, loss_bbox, label, label_weight, pts_pred_init, pts_pred_refine, bbox_weight, gt_bbox, pos_inds,
                 pos_gt_inds,
                 num_proposals_lvls=None, num_lvls=None):
        if len(pos_inds) == 0:
            return label, label_weight, bbox_weight, 0, torch.tensor([]).type_as(bbox_weight)
        num_gt = pos_gt_inds.max()
        num_proposals_each_level_ = num_proposals_lvls.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_lvls):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        overlaps_matrix = convex_overlaps(gt_bbox, pts_pred_init)
        pos_inds_after_assign = []
        cls_pos_inds_after_assign = []
        bbox_pos_inds_after_assign = []
        ignore_inds_after_assign = []
        reassign_weights_after_assign = []
        reassign_weights_after_assign_cls = []
        reassign_weights_after_assign_bbox = []
        for gt_inds in range(num_gt):
            pos_inds_cfa = []
            pos_loss_cfa = []
            cls_loss_assign = []
            cls_inds_assign = []
            bbox_loss_assign = []
            bbox_inds_assign = []
            pos_overlaps_init_cfa = []
            gt_mask = (pos_gt_inds == gt_inds + 1)
            for level in range(num_lvls):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                # 属于某一个GT的损失项
                # value, topk_inds = pos_losses[level_gt_mask].topk(
                #     min(level_gt_mask.sum(), self.topk), largest=False)
                #
                # # 额外增加了种类和bbox的前k个索引和值
                # cls_value, cls_topk_inds = loss_cls[level_gt_mask].topk(
                #     min(level_gt_mask.sum(), self.topk), largest=False)
                #
                # bbox_value, box_topk_inds = loss_bbox[level_gt_mask].topk(
                #     min(level_gt_mask.sum(), self.topk), largest=False
                # )

                # 这里已经做过排序了，所以下面的排序会显得有些多余
                _value, _inds = torch.stack((pos_losses, loss_cls, loss_bbox), dim=0)[:,
                                level_gt_mask].topk(min(level_gt_mask.sum(), self.topk), dim=1, largest=False)

                value, topk_inds = _value[0], _inds[0]
                cls_value, cls_topk_inds = _value[1], _inds[1]
                bbox_value, box_topk_inds = _value[2], _inds[2]

                # pytorch 没想到居然还能这样用
                pos_inds_cfa.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_cfa.append(value)

                # 将loss和bbox的索引和值存入到列表中
                cls_inds_assign.append(pos_inds[level_gt_mask][cls_topk_inds])
                cls_loss_assign.append(cls_value)

                bbox_inds_assign.append((pos_inds[level_gt_mask][box_topk_inds]))
                bbox_loss_assign.append(bbox_value)

                pos_overlaps_init_cfa.append(overlaps_matrix[:, pos_inds[level_gt_mask][topk_inds]])
            pos_inds_cfa = torch.cat(pos_inds_cfa)
            pos_loss_cfa = torch.cat(pos_loss_cfa)
            pos_overlaps_init_cfa = torch.cat(pos_overlaps_init_cfa, 1)
            # 将cls和bbox进行拼接
            cls_inds_assign = torch.cat(cls_inds_assign)
            cls_loss_assign = torch.cat(cls_loss_assign)

            bbox_inds_assign = torch.cat(bbox_inds_assign)
            bbox_loss_assign = torch.cat(bbox_loss_assign)

            if len(pos_inds_cfa) < 2:
                pos_inds_after_assign.append(pos_inds_cfa)
                # cls_pos_inds_after_assign.append(pos_inds_cfa)
                # bbox_pos_inds_after_assign.append(pos_inds_cfa)
                ignore_inds_after_assign.append(pos_inds_cfa.new_tensor([]))
                reassign_weights_after_assign.append(pos_loss_cfa.new_ones([len(pos_inds_cfa)]))
                reassign_weights_after_assign_bbox.append(pos_loss_cfa.new_ones([len(pos_inds_cfa)]))
                reassign_weights_after_assign_cls.append(pos_loss_cfa.new_ones([len(pos_inds_cfa)]))
            else:
                pos_loss_cfa, sort_inds = pos_loss_cfa.sort()
                pos_inds_cfa = pos_inds_cfa[sort_inds]

                # cls
                # cls_loss_assign, cls_sort_inds = cls_loss_assign.sort()
                # cls_inds_assign = cls_inds_assign[cls_sort_inds]
                _cls_loss_assign = cls_loss_assign[sort_inds]
                _cls_inds_assign = cls_inds_assign[sort_inds]

                # bbox
                # bbox_loss_assign, bbox_sort_inds = bbox_loss_assign.sort()
                # bbox_inds_assign = bbox_inds_assign[bbox_sort_inds]
                _bbox_loss_assign = bbox_loss_assign[sort_inds]
                _bbox_inds_assign = bbox_inds_assign[sort_inds]

                pos_overlaps_init_cfa = pos_overlaps_init_cfa[:, sort_inds].reshape(-1, len(pos_inds_cfa))
                pos_loss_cfa = pos_loss_cfa.reshape(-1)
                loss_mean = pos_loss_cfa.mean()
                loss_var = pos_loss_cfa.var()

                # cls
                # cls_loss_assign = cls_loss_assign.reshape(-1)
                # cls_loss_assign_mean = cls_loss_assign.mean()
                # cls_loss_assign_var = cls_loss_assign.var()

                # bbox
                # bbox_loss_assign = bbox_loss_assign.reshape(-1)
                # bbox_loss_assign_mean = bbox_loss_assign.mean()
                # bbox_loss_assign_var = bbox_loss_assign.var()

                guass_prob_density = (-(pos_loss_cfa - loss_mean) ** 2 / loss_var).exp() / loss_var.sqrt()
                index_inverted, _ = torch.arange(len(guass_prob_density)).sort(descending=True)
                guass_prob_inverted = torch.cumsum(guass_prob_density[index_inverted], 0)
                guass_prob = guass_prob_inverted[index_inverted]
                guass_prob_norm = (guass_prob - guass_prob.min()) / (guass_prob.max() - guass_prob.min())

                # # cls
                # cls_guass_prob_density = (-(cls_loss_assign - cls_loss_assign_mean) ** 2 /
                #                           cls_loss_assign_var).exp() / cls_loss_assign_var.sqrt()
                # cls_index_inverted, _ = torch.arange(len(cls_guass_prob_density)).sort(descending=True)
                # cls_guass_prob_inverted = torch.cumsum(cls_guass_prob_density[cls_index_inverted], 0)
                # cls_guass_prob = cls_guass_prob_inverted[cls_index_inverted]
                # cls_guass_prob_norm = (cls_guass_prob-cls_guass_prob.min())/(cls_guass_prob.max() - cls_guass_prob.min())
                #
                # # bbox
                # bbox_guass_prob_density = (-(bbox_loss_assign - bbox_loss_assign_mean) ** 2 /
                #                           bbox_loss_assign_var).exp() / bbox_loss_assign_var.sqrt()
                # bbox_index_inverted, _ = torch.arange(len(bbox_guass_prob_density)).sort(descending=True)
                # bbox_guass_prob_inverted = torch.cumsum(bbox_guass_prob_density[bbox_index_inverted], 0)
                # bbox_guass_prob = bbox_guass_prob_inverted[bbox_index_inverted]
                # bbox_guass_prob_norm = (bbox_guass_prob - bbox_guass_prob.min()) / (
                #             bbox_guass_prob.max() - bbox_guass_prob.min())


                # splitting by gradient consistency
                # 这里就相当于论文中的置信度乘loss 将高斯分布的均值移到正半轴上
                loss_curve = guass_prob_norm * pos_loss_cfa
                _, max_thr = loss_curve.topk(1)
                reweights = guass_prob_norm[:max_thr + 1]


                # # cls
                # cls_loss_curve = cls_guass_prob_norm * cls_loss_assign
                # _, cls_max_thr = cls_loss_curve.topk(1)
                #
                # # bbox
                # bbox_loss_curve = bbox_guass_prob_norm * bbox_loss_assign
                # _, bbox_max_thr = bbox_loss_curve.topk(1)

                # feature anti-aliasing coefficient
                pos_overlaps_init_cfa = pos_overlaps_init_cfa[:, :max_thr + 1]
                overlaps_level = pos_overlaps_init_cfa[gt_inds] / (pos_overlaps_init_cfa.sum(0) + 1e-6)

                # reweights = self.anti_factor * overlaps_level * reweights + 1e-6
                reweights_cls = torch.abs((1 + (-_cls_loss_assign[:max_thr + 1]).exp()-_bbox_loss_assign[:max_thr + 1]))**2
                reweights_bbox = torch.abs((1 - (-_cls_loss_assign[:max_thr + 1]).exp()+_bbox_loss_assign[:max_thr + 1]))**2
                # reweights = reweights * reweights_
                reassign_weights = reweights.reshape(-1) / reweights.sum() * torch.ones(len(reweights)).type_as(
                    guass_prob_norm).sum()
                pos_inds_temp = pos_inds_cfa[:max_thr + 1]
                ignore_inds_temp = pos_inds_cfa.new_tensor([])

                # cls
                # cls_inds_assign_temp = cls_inds_assign[:cls_max_thr + 1]
                #
                # # bbox
                # bbox_inds_assign_temp = bbox_inds_assign[:bbox_max_thr + 1]

                # # 取交集以及余项
                # combined_all = torch.cat((cls_inds_assign_temp, bbox_inds_assign_temp))
                # uniques, counts = combined_all.unique(return_counts=True)
                # difference = uniques[counts < 2]
                # intersection = uniques[counts == 2]

                # if cls_loss_assign.max()-cls_loss_assign.min():

                pos_inds_temp = pos_inds_temp
                # 将bbox weight与cls weight分开得权重

                # cls_bbox_union = torch.cat((cls_inds_assign_temp, bbox_inds_assign_temp))
                # cls_bbox_union_uniques, cls_bbox_union_counts = cls_bbox_union.unique(return_counts=True)
                # cls_bbox_difference = cls_bbox_union_uniques[counts < 2]
                # cls_bbox_intersection = cls_bbox_union_uniques[counts == 2]


                pos_inds_after_assign.append(pos_inds_temp)
                # cls_pos_inds_after_assign.append(cls_inds_assign_temp)
                # bbox_pos_inds_after_assign.append(bbox_inds_assign_temp)
                ignore_inds_after_assign.append(ignore_inds_temp)
                reassign_weights_after_assign.append(reassign_weights)

                # cls bbox
                reassign_weights_after_assign_cls.append(reweights_cls)
                reassign_weights_after_assign_bbox.append(reweights_bbox)

        pos_inds_after_cfa = torch.cat(pos_inds_after_assign)
        # cls_pos_inds_after_assign = torch.cat(cls_pos_inds_after_assign)
        # bbox_pos_inds_after_assign = torch.cat(bbox_pos_inds_after_assign)

        ignore_inds_after_cfa = torch.cat(ignore_inds_after_assign)
        reassign_weights_after_cfa = torch.cat(reassign_weights_after_assign)


        # cls bbox
        reassign_weights_after_assign_bbox_ =torch.cat(reassign_weights_after_assign_bbox)
        reassign_weights_after_assign_cls_ = torch.cat(reassign_weights_after_assign_cls)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_cfa).all(1)

        # cls和bbox
        # cls_reassign_mask = (pos_inds.unsqueeze(1) != cls_pos_inds_after_assign).all(1)
        # bbox_reassign_mask = (pos_inds.unsqueeze(1) != bbox_pos_inds_after_assign).all(1)
        reassign_ids = pos_inds[reassign_mask]

        # cls bbox inds
        # cls_reassign_ids = pos_inds[cls_reassign_mask]
        # bbox_reassign_ids = pos_inds[bbox_reassign_mask]

        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_cfa] = 0

        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_cfa)

        # import matplotlib.pyplot as plt
        # _assign_cls = loss_cls[~reassign_mask].cpu().numpy()
        # _assign_bbox = loss_bbox[~reassign_mask].cpu().numpy()
        # x_range = range(_assign_cls.shape[0])
        # plt.plot(x_range, (_assign_cls-_assign_cls.min())/(_assign_cls.max()-_assign_cls.min()), 'o')
        # plt.plot(x_range, (_assign_bbox-_assign_bbox.min())/(_assign_bbox.max()-_assign_bbox.min()),'o')
        # plt.savefig('1.png')
        # plt.cla()
        reassign_weights_mask = (pos_inds.unsqueeze(1) == pos_inds_after_cfa).any(1)
        reweight_ids = pos_inds[reassign_weights_mask]
        label_weight[reweight_ids] = reassign_weights_after_assign_cls_
        bbox_weight[reweight_ids, :] = reassign_weights_after_assign_bbox_.unsqueeze(1)

        pos_level_mask_after_cfa = []
        for i in range(num_lvls):
            mask = (pos_inds_after_cfa >= inds_level_interval[i]) & (
                    pos_inds_after_cfa < inds_level_interval[i + 1])
            pos_level_mask_after_cfa.append(mask)
        pos_level_mask_after_cfa = torch.stack(pos_level_mask_after_cfa, 0).type_as(label)
        pos_normalize_term = pos_level_mask_after_cfa * (
                self.point_base_scale *
                torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(bbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_cfa)

        return label, label_weight, bbox_weight, num_pos, pos_normalize_term

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   cls_feat,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        # showHeatmap(cls_feat, img_metas=img_metas, rescale=True)
        # cls score second stage
        # pts_preds_init first stage
        # pts_preds_refine second stage
        assert len(cls_scores) == len(pts_preds_refine)
        device = cls_scores[0].device
        bbox_preds_refine = [
            # self.points2bbox(pts_pred_refine)
            pts_pred_refine
            for pts_pred_refine in pts_preds_refine
        ]
        bbox_preds_init = [
            # self.points2bbox(pts_pred_refine)
            pts_pred_init
            for pts_pred_init in pts_preds_init
        ]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i], device)
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            bbox_pred_list_init = [
                bbox_preds_init[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, bbox_pred_list_init,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms, img_metas=img_metas)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           bbox_preds_init,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           img_metas=None,
                           show_point=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        sift_pts = []
        sift_center = []
        for i_lvl, (cls_score, bbox_pred, bbox_pred_init, points) in enumerate(
                zip(cls_scores, bbox_preds, bbox_preds_init, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            bbox_pred_init = bbox_pred_init.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                bbox_pred_init = bbox_pred_init[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pred = bbox_pred.reshape(-1, self.num_points, 2)
            pts_pred_init = bbox_pred_init.reshape(-1, self.num_points, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]
            pts_pred_offsetx = pts_pred[:, :, 1::2]
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety], dim=2).reshape(-1, 2 * self.num_points)

            pts_pred_offsety_init = pts_pred_init[:, :, 0::2]
            pts_pred_offsetx_init = pts_pred_init[:, :, 1::2]
            pts_pred_init = torch.cat([pts_pred_offsetx_init, pts_pred_offsety_init], dim=2).reshape(-1,
                                                                                                     2 * self.num_points)

            if show_point:
                pts = pts_pred_init * self.point_strides[i_lvl]
                sift_pts.append(pts.clone())
                sift_center.append(points.clone())

            bbox_pred = minaerarect(pts_pred)
            bbox_pos_center = points[:, :2].repeat(1, 4)
            # bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            bboxes = poly2obb(bboxes)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes[:, :-1] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if with_nms:
            return_inds = True
            det_bboxes, det_labels, keep = multiclass_nms_r(mlvl_bboxes, mlvl_scores,
                                                            cfg.score_thr, cfg.nms,
                                                            cfg.max_per_img, return_inds=return_inds,
                                                            class_agnostic=True)
            if show_point:
                sift_pts = torch.cat(sift_pts)
                sift_center = torch.cat(sift_center)
                if return_inds:
                    inds = mlvl_scores[:, :-1].reshape(-1) > cfg.score_thr
                    sift_pts = sift_pts[:, None].expand(-1, 15, sift_pts.size(-1))
                    sift_pts = sift_pts.reshape(-1, sift_pts.size(-1))
                    sift_center = sift_center[:, None].expand(-1, 15, sift_center.size(-1))
                    sift_center = sift_center.reshape(-1, sift_center.size(-1))
                    # inds = valid_mask.sum(1).squeeze().type(torch.bool)
                    sift_pts = sift_pts[inds]
                    sift_pts = sift_pts[keep]
                    sift_center = sift_center[inds]
                    sift_center = sift_center[keep]
                from mmdet.utils.heatmap import showPoints
                showPoints(sift_pts, sift_center, img_metas=img_metas, bboxes=det_bboxes)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores
