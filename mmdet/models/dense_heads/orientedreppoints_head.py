import torch

from .rreppoints_head import RRepPointsHead
from mmdet.core.bbox import obb2poly
from mmdet.core import multi_apply, unmap
from ..builder import HEADS, build_loss
from mmdet.ops import minaerarect, ChamferDistance2D
from mmdet.core.anchor import levels_to_images, images_to_levels
import numpy as np


# modified from https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA
@HEADS.register_module()
class OrientedReppointsHead(RRepPointsHead):
    def __init__(self, top_ratio=0.4,
                 loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
                 **kwargs):
        super(OrientedReppointsHead, self).__init__(**kwargs)
        self.top_ratio = top_ratio
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)

    def quality_assessment(self,
                           cls_score,
                           pts_pred_init,
                           pts_pred_refine,
                           labels,
                           gt_bbox,
                           label_weights,
                           bbox_weight):
        # label weight
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        # bbox weight
        pos_inds = (bbox_weight.mean(dim=1).reshape(-1) > 0).nonzero(as_tuple=False).reshape(-1)
        pos_pts_pred_init = pts_pred_init.reshape(-1, 18)[pos_inds]
        pos_pts_pred_refine = pts_pred_refine.reshape(-1, 18)[pos_inds]
        gt_bbox_ = gt_bbox.reshape(-1, 8)[pos_inds]
        bbox_weight_ = bbox_weight.reshape(-1, 8)[pos_inds]
        bbox_weight_ = bbox_weight_.mean(dim=1).reshape(-1)

        # qua_cls
        qua_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=1.0,
            reduction_override='none')

        qua_loc_init = self.loss_bbox_init(
            pos_pts_pred_init,
            gt_bbox_,
            bbox_weight_,
            avg_factor=1.0,
            reduction_override='none')

        qua_loc_refine = self.loss_bbox_refine(  # Q_loc_loss_refine: torch.size(gt_label_num*topk)
            pos_pts_pred_refine,
            gt_bbox_,
            bbox_weight_,
            avg_factor=1.0,
            reduction_override='none')

        corners_pred_init = minaerarect(pos_pts_pred_init)
        corners_pred_refine = minaerarect(pos_pts_pred_refine)

        # sample points
        sampling_pts_pred_init = self.sampling_points(corners_pred_init, 10)
        sampling_pts_pred_refine = self.sampling_points(corners_pred_refine, 10)
        corners_pts_gt = self.sampling_points(gt_bbox_, 10)

        #  Chamfer distance2D
        qua_ori_init = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_refine)
        # TODO 这里的样本cls可以继续想办法打改进
        qua_cls = qua_cls.sum(-1)[pos_inds]
        # weight inti-stage and refine-stage
        qua = qua_cls + 0.2 * (qua_loc_init + 0.3 * qua_ori_init) + 0.8 * (
                qua_loc_refine + 0.3 * qua_ori_refine)
        return qua, qua_cls

    def sampling_points(self, corners, points_num):
        """
        Args:
            corners(tensor) : torch.size(n, 8), 四边形的四个点的位置
            points_num(int) : 每条边上的采样点数量
        return:
            all_points(tensor) : torch.size(n, 4*points_num, 2) ，四边形的采样点集的绝对坐标
        """
        device = corners.device
        corners_xs, corners_ys = corners[:, 0::2], corners[:, 1::2]
        first_edge_x_points = corners_xs[:, 0:2]  # 第一条边取x坐标 (n, 2)
        first_edge_y_points = corners_ys[:, 0:2]  # 第一条边取y坐标 (n, 2)
        sec_edge_x_points = corners_xs[:, 1:3]
        sec_edge_y_points = corners_ys[:, 1:3]
        third_edge_x_points = corners_xs[:, 2:4]
        third_edge_y_points = corners_ys[:, 2:4]
        four_edge_x_points_s = corners_xs[:, 3]
        four_edge_y_points_s = corners_ys[:, 3]
        four_edge_x_points_e = corners_xs[:, 0]
        four_edge_y_points_e = corners_ys[:, 0]

        edge_ratio = torch.linspace(0, 1, points_num).to(device).repeat(corners.shape[0],
                                                                        1)  # 0-1采样points_num个ratio，并重复n次  torch.size(n, points_num)
        all_1_edge_x_points = edge_ratio * first_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_x_points[:, 0:1]  # (n, points_num)  开始间隔采样，得到真实坐标
        all_1_edge_y_points = edge_ratio * first_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_y_points[:, 0:1]

        all_2_edge_x_points = edge_ratio * sec_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_x_points[:, 0:1]
        all_2_edge_y_points = edge_ratio * sec_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_y_points[:, 0:1]

        all_3_edge_x_points = edge_ratio * third_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_x_points[:, 0:1]
        all_3_edge_y_points = edge_ratio * third_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_y_points[:, 0:1]

        all_4_edge_x_points = edge_ratio * four_edge_x_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_x_points_s.unsqueeze(1)
        all_4_edge_y_points = edge_ratio * four_edge_y_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_y_points_s.unsqueeze(1)

        all_x_points = torch.cat([all_1_edge_x_points, all_2_edge_x_points,  # (n, 4*points_num, 1)
                                  all_3_edge_x_points, all_4_edge_x_points], dim=1).unsqueeze(dim=2)

        all_y_points = torch.cat([all_1_edge_y_points, all_2_edge_y_points,  # (n, 4*points_num, 1)
                                  all_3_edge_y_points, all_4_edge_y_points], dim=1).unsqueeze(dim=2)

        all_points = torch.cat([all_x_points, all_y_points], dim=2)  # (n, 4*points_num, 2)
        return all_points

    def point_samples_selection(self, qua_assess, labels, label_weight, bbox_weight, pos_gt_index,
                                num_proposals_each_lvl):
        pos_inds = (bbox_weight.mean(dim=1).reshape(-1) > 0).nonzero(as_tuple=False).reshape(-1)
        if len(pos_inds) == 0:
            return labels, label_weight, bbox_weight, 0, torch.tensor([]).to(bbox_weight)
        num_gt = pos_gt_index.max()
        num_proposals_each_lvl_ = num_proposals_each_lvl.copy()
        num_lvl = len(num_proposals_each_lvl)
        num_proposals_each_lvl_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_lvl_)

        pos_level_mask = []
        for i in range(num_lvl):
            # set index in each feature map lvl
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        pos_inds_after_select = []
        ignore_inds_after_select = []
        assign_list = []
        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = (pos_gt_index == gt_ind + 1)
            assign_list.append(gt_mask.sum())
            for level in range(num_lvl):
                level_mask = pos_level_mask[level]
                # find point which correspond to positive and especially gt
                level_gt_mask = level_mask & gt_mask
                # find topk qua
                value, topk_inds = qua_assess[level_gt_mask].topk(
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))
            else:
                # small to large
                pos_loss_select, sort_inds = pos_loss_select.sort()
                pos_inds_select = pos_inds_select[sort_inds]
                topk = int(np.ceil(pos_loss_select.shape[0] * self.top_ratio))
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(
                    pos_inds_select_topk)  # List[num_gt*tensor] tensor.size(n) n为质量评估后的topk正样本数量
                ignore_inds_after_select.append(pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)

        # find ignore positive samples
        # ignore can be deleted
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]
        labels[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        pos_level_mask_after_select = []
        for i in range(num_lvl):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                    pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select, 0).type_as(labels)

        pos_normalize_term = pos_level_mask_after_select * (
                self.point_base_scale *
                torch.as_tensor(self.point_strides).type_as(labels)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(bbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return labels, label_weight, bbox_weight, num_pos, pos_normalize_term  # pos_normalize_term： torch.size(num_pos) 重采样后正样本的对应的尺度

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
            gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)

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
         all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds) = multi_apply(
            self._point_target_single,
            proposals_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            stage=stage,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # add pos_gt_inds
        pos_inds = []
        pos_gt_index = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = single_labels < self.num_classes
            pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))
            pos_gt_index.append(all_gt_inds[i][pos_mask.nonzero(as_tuple=False).view(-1)])

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
                proposal_weights_list, num_total_pos, num_total_neg, pos_gt_index)

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
            avg_factor=num_total_samples_refine)
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

        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm / normalize_term,
            bbox_gt_init_norm / normalize_term,
            bbox_weights_init_norm
        )
        loss_border_refine = self.loss_spatial_init(
            pts_pred_refine_norm / normalize_term,
            bbox_gt_refine_norm / normalize_term,
            bbox_weights_refine_norm
        )
        return loss_cls, loss_pts_init, loss_pts_refine, loss_border_init, loss_border_refine

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
            # transform center list to bbox list and
            #   assign target for bbox list
            # bbox_list = self.centers_to_bboxes(center_list)
            # candidate_list = bbox_list
            raise NotImplementedError
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
         num_total_pos_init, num_total_neg_init, pos_gt_index_init) = cls_reg_targets_init

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
                # bbox_center = torch.cat(
                #     [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
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
         num_total_neg_refine, pos_gt_index_refine) = cls_reg_targets_refine
        # 首先上述几个变量都已经被映射成lvl batch
        # label_list 就是标签 [bs, num]
        # label_weights 标签权重
        # bbox_gt_list_refine 对应featurepoint上的gt
        # candidate_list_refine 预测得到的bbox

        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        with torch.no_grad():
            # lvl --> images
            cls_scores_ = levels_to_images(cls_scores)
            pts_preds_init_ = levels_to_images(pts_preds_init)
            pts_preds_refine_ = levels_to_images(pts_preds_refine)
            labels_list_ = levels_to_images(labels_list)
            bbox_gt_list_refine_ = levels_to_images(bbox_gt_list_refine)
            label_weights_list_ = levels_to_images(label_weights_list)
            bbox_weights_list_refine_ = levels_to_images(bbox_weights_list_refine)
            num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))  # size(5) f_h*f_w
                                        for featmap in cls_scores]

            qua_assess, _ = multi_apply(self.quality_assessment,
                                        cls_scores_,
                                        pts_preds_init_,
                                        pts_preds_refine_,
                                        labels_list_,
                                        bbox_gt_list_refine_,
                                        label_weights_list_,
                                        bbox_weights_list_refine_)

            labels_list, label_weights_list, bbox_weights_list_refine, num_pos, _ = multi_apply(
                self.point_samples_selection,
                qua_assess,
                labels_list_,
                label_weights_list_,
                bbox_weights_list_refine_,
                pos_gt_index_refine,
                num_proposals_each_lvl=num_proposals_each_level
            )
            num_pos = sum(num_pos)

        labels_list = images_to_levels(labels_list, num_proposals_each_level)
        label_weights_list = images_to_levels(label_weights_list, num_proposals_each_level)
        bbox_weights_list_refine = images_to_levels(bbox_weights_list_refine, num_proposals_each_level)

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine, loss_border_init, loss_border_refine = multi_apply(
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
            'loss_pts_refine': losses_pts_refine,
            'loss_border_init': loss_border_init,
            'loss_border_refine': loss_border_refine
        }
        return loss_dict_all
