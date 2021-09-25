import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import batched_rnms
from mmdet.core import anchor_inside_flags, unmap
from mmcv.ops import batched_nms

from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin
from mmdet.core import multi_apply, images_to_levels
from mmcv.runner import force_fp32

from mmdet.core.bbox.rtransforms import CV_L2LT_RB_TORCH


@HEADS.register_module()
class ORPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(ORPNHead, self).__init__(1, in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 5, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)  # [batch_size,num_anchors_in_a_lvl] --> [batch_size*num_anchors_in_a_lvl]
        label_weights = label_weights.reshape(-1)

        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        # [batchszie, channel, H, W] --> [batchszie, H, W, channel] --> [., c]
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)  # [batchsize*num_bboxes, 5]
        bbox_weights = bbox_weights.reshape(-1, 5)  # [batchsize*num_bboxes, 5]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        # [batch_size, 5*num_base_anchors, H, W] --> [batch_size, H, W, 5*num_base_anchors] --> [N, 5]
        # TODO: reshape, self.reg_decoded_bbox
        # 每一层有多少anchors，同样会预测多少bbox
        if self.reg_decoded_bbox:
            raise NotImplementedError
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             hor_gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # cls：channel = num_classes * anchors; list[] five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        # box_pred: channel = anchors * 5;five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # both len list is batch_size;
        # and anchor_list doesn't have sth. with images each [Num_anchors, 4]
        # etc:anchor_list[0]=[[Num_anchors, 4],[],..numlvls]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            hor_gt_bboxes,
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
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            hor_gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        # check whether the anchor inside the border
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assert gt_bboxes_ignore.size(0) == 0, 'Not Implement for ignore use'
        assign_result = self.assigner.assign(
            anchors, hor_gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              hor_gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros(num_valid_anchors, 5)
        bbox_weights = anchors.new_zeros(num_valid_anchors, 5)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, gt_bboxes[sampling_result.pos_assigned_gt_inds])
            else:
                raise NotImplementedError
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
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
                    hor_gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
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
            hor_gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
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
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        # cls_score_list  [[num_base_achors, H, W],..(nun_lvls)]
        # bbox_pred_list  [[5*num_base_achors, H, W],..(nun_lvls)]
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
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            # [5*num_base_achors, H, W] --> [H, W, 5*num_base_achors] --> [H*W*num_base_achors, 5]
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
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)  # [[2000],...(num_lvls)] --> dim:1
        anchors = torch.cat(mlvl_valid_anchors)  # [[2000, 5], ..(num_lvls)] --> [num, 4]
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)  # [num, 5]
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2]
            h = proposals[:, 3]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]
        nms_cfg = dict(type='rnms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_rnms(proposals, scores, ids, nms_cfg)
        # dets[n, 5(x, y, w, h, theta, scores)]
        # keep index
        return dets[:cfg.nms_post]
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      hor_gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)  # 自己调用自己
        # outs --> tuple([[batchsize, C, H, W],..(num_lvls)],..(outputs_num_branch))
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, hor_gt_bboxes, None, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, hor_gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            # [[n, 5],...(num_images)]
            return losses, proposal_list

