import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms, rbbox2result, rbbox2roi, CV_L_Rad2LE_DEF_TORCH)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin, RBBoxTestMixin


@HEADS.register_module()
class RoItranshead(BaseRoIHead, RBBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(RoItranshead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'],)
        return outs

    def _bbox_forward(self, stage, x, rois, roi_format='CV_LEFT'):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)

        # do not support caffe_c4 model anymore
        cls_score_h, bbox_pred_h, cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, cls_score_h=cls_score_h, bbox_pred_h=bbox_pred_h, \
            bbox_feats=bbox_feats)
        return bbox_results

    def _rbbox_forward(self, stage, x, rois, roi_format='CV_LEFT'):
        """Box head forward function used in both training and testing."""
        _rois = rois.clone()
        if roi_format == 'CV_LEFT':
            _rois[:, 1:] = CV_L_Rad2LE_DEF_TORCH(_rois[:, 1:])
            _rois[:, -1] = -_rois[:, -1]
            _rois[:, 3] = _rois[:, 3] * 1.2
            _rois[:, 4] = _rois[:, 4] * 1.4
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        _rois)

        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, \
            bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)

        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'], bbox_results['bbox_pred'],
                                               bbox_results['cls_score_h'], bbox_results['bbox_pred_h'],
                                               rois, *bbox_targets)
        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _rbbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._rbbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)

        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'], bbox_results['bbox_pred'],
                                               rois, *bbox_targets)
        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      hor_gt_bboxes,
                      gt_bboxes,
                      gt_labels,
                      hor_gt_bboxes_ignore=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        losses = dict()
        rcnn_train_cfg = self.train_cfg[0]
        lw = self.stage_loss_weights[0]


        if self.with_bbox:
            sampling_results = []
            bbox_assigner = self.bbox_assigner[0]
            bbox_sampler = self.bbox_sampler[0]
            num_imgs = len(img_metas)

            if hor_gt_bboxes_ignore is None:
                hor_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[j], hor_gt_bboxes[j], hor_gt_bboxes_ignore[j],
                    gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    hor_gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        bbox_results = self._bbox_forward_train(0, x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                rcnn_train_cfg)

        for name, value in bbox_results['loss_bbox'].items():
            losses[f's0.{name}'] = (
                value * lw if 'loss' in name else value)

        # bbox head first stage bbox refine
        # and horizon proposal --> rotate proposal
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # bbox_targets is a tuple
        roi_labels = bbox_results['bbox_targets'][0]
        with torch.no_grad():
            roi_labels = torch.where(
                roi_labels == self.bbox_head[0].num_classes,
                bbox_results['cls_score'][:, :-1].argmax(1),
                roi_labels)
            proposal_list = self.bbox_head[0].refine_rbboxes(
                bbox_results['rois'], roi_labels,
                bbox_results['bbox_pred'], pos_is_gts, img_metas)

        # the second stage
        rcnn_train_cfg = self.train_cfg[1]
        lw = self.stage_loss_weights[1]

        if self.with_bbox:
            sampling_results = []
            bbox_assigner = self.bbox_assigner[1]
            bbox_sampler = self.bbox_sampler[1]
            num_imgs = len(img_metas)

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                    gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        bbox_results = self._rbbox_forward_train(1, x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                rcnn_train_cfg)

        for name, value in bbox_results['loss_bbox'].items():
            losses[f's1.{name}'] = (
                value * lw if 'loss' in name else value)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False, obb=True, submission=True):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # 只有计算置信度的时候是需要三个阶段的预测头一起进行的
        ms_scores_r = []
        rcnn_test_cfg = self.test_cfg

        # the first stage is horizon bbox
        rois = bbox2roi(proposal_list)
        bbox_results = self._bbox_forward(0, x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        num_proposals_per_img = tuple(
            len(proposals) for proposals in proposal_list)

        rois = rois.split(num_proposals_per_img, 0)
        # rotate
        cls_score = cls_score.split(num_proposals_per_img, 0)

        if isinstance(bbox_pred, torch.Tensor):
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        else:
            bbox_pred = self.bbox_head[0].bbox_pred_split(
                bbox_pred, num_proposals_per_img)

        ms_scores_r.append(cls_score)

        # 得到旋转预测之后，再将其转换成旋转框
        bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
        rois = torch.cat([
            self.bbox_head[0].regress_by_h2rclass(rois[j], bbox_label[j],
                                               bbox_pred[j],
                                               img_metas[j])
            for j in range(num_imgs)
        ])

        bbox_results = self._rbbox_forward(1, x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        num_proposals_per_img = tuple(
            len(proposals) for proposals in proposal_list)
        # split proposal
        rois = rois.split(num_proposals_per_img, 0)

        # rotate
        cls_score = cls_score.split(num_proposals_per_img, 0)

        if isinstance(bbox_pred, torch.Tensor):
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        else:
            bbox_pred = self.bbox_head[1].bbox_pred_split(
                bbox_pred, num_proposals_per_img)
        ms_scores_r.append(cls_score)
        import pdb
        pdb.set_trace()
        # rotate
        cls_score_r = [
            sum([score_r[i] for score_r in ms_scores_r]) / float(len(ms_scores_r))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score_r[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)

            # 返回bbox
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if submission == True:
            ensenmble_results = {}
            # ensenmble_results['hbb'] = [
            #     bbox2result(det_bboxes_h[i], det_labels_h[i],
            #                 self.bbox_head[-1].num_classes)
            #     for i in range(num_imgs)
            # ]
            ensenmble_results['rbb'] = [
                rbbox2result(det_bboxes[i], det_labels[i],
                             self.bbox_head[-1].num_classes)
                for i in range(num_imgs)
            ]
        else:
            if obb:
                ensenmble_results = [
                    rbbox2result(det_bboxes[i], det_labels[i],
                                 self.bbox_head[-1].num_classes)
                    for i in range(num_imgs)
                ]
            else:
                ensenmble_results = [
                    rbbox2result(det_bboxes[i], det_labels[i],
                                 self.bbox_head[-1].num_classes)
                    for i in range(num_imgs)
                ]

        # 暂时不考虑实例分割
        if self.with_mask:
            # results = list(
            #     zip(ms_bbox_result_h['ensemble'], ms_segm_result_h['ensemble']))
            assert not self.with_mask, 'not yet complete'
        # else:
        #     results = ms_bbox_result['ensemble']

        return ensenmble_results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        raise NotImplementedError
