import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin, RBBoxTestMixin
from mmdet.core import rbbox2result, rbbox2roi, CV_L_Rad2LT_RB_TORCH, CV_L_Rad2LE_DEF_TORCH


@HEADS.register_module()
class OrientedRoIHead(BaseRoIHead, RBBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, num_bboxtype=1, **kwargs):
        super(OrientedRoIHead, self).__init__(**kwargs)
        self.num_bboxtype = num_bboxtype

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

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

        # assign gts and sample proposals
        # x = [[batch_size,RPNout_channel ,H, W],...(num_lvls)]
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assert gt_bboxes_ignore[i].size(0)==0, 'ignore not implement'
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # TODO check the rightness
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # 如果正样本不足的话，那也只能将所有正样本带上剩下的都是负样本
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    hor_gt_bboxes)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, roi_format='CV_LEFT'):
        _rois = rois.clone()
        if roi_format == 'CV_LEFT':
            _rois[:, 1:] = CV_L_Rad2LE_DEF_TORCH(_rois[:, 1:])
            _rois[:, -1] = -_rois[:, -1]
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], _rois)

        # [batchsize * rpn_out_channel, bbox_head_in_channels, 7, 7]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # TODO add horizon braunch
        if self.num_bboxtype == 1:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred,
                bbox_feats=bbox_feats)
        elif self.num_bboxtype == 2:
            cls_score_h, bbox_pred_h, cls_score, bbox_pred = self.bbox_head(bbox_feats)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, cls_score_h=cls_score_h, bbox_pred_h=bbox_pred_h,
                bbox_feats=bbox_feats)

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            hor_gt_bboxes):
        """Run forward function and calculate loss for box head in training."""
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        # rois是res sampling之后的结果etc.[512, 5] -->[n, 6], [batch_ind, x, y, w, h, theta]
        bbox_results = self._bbox_forward(x, rois)

        if self.num_bboxtype == 1:
            bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                      gt_labels, self.train_cfg)

            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'],
                                            rois, *bbox_targets)
        elif self.num_bboxtype == 2:
            for i in range(len(sampling_results)):
                sampling_results[i].neg_hor_bboxes = CV_L_Rad2LT_RB_TORCH(sampling_results[i].neg_bboxes)
                sampling_results[i].pos_hor_bboxes = CV_L_Rad2LT_RB_TORCH(sampling_results[i].pos_bboxes)
                sampling_results[i].pos_hor_gt_bboxes = hor_gt_bboxes[i][sampling_results[i].pos_assigned_gt_inds.tolist(), :]
            bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                      gt_labels, self.train_cfg)

            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'],
                                            bbox_results['cls_score_h'], bbox_results['bbox_pred_h'],
                                            rois, *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = rbbox2roi(proposals)
        # rois [1000, 5] [x1, y1, x2, y2, scores] ---> [batchind, x1, y1, x2, y2]
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        if self.num_bboxtype == 2:
            cls_score_h = bbox_results['cls_score_h']
            bbox_pred_h = bbox_results['bbox_pred_h']
            cls_score_h = cls_score_h.split(num_proposals_per_img, 0)
            if bbox_pred_h is not None:
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred_h, torch.Tensor):
                    bbox_pred_h = bbox_pred_h.split(num_proposals_per_img, 0)  # tuple(per_img_bbox_pred)
                else:
                    bbox_pred_h = self.bbox_head_h.bbox_pred_split(
                        bbox_pred_h, num_proposals_per_img)
            else:
                bbox_pred_h = (None,) * len(proposals)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)
        if self.num_bboxtype == 1:
            # apply bbox post-processing to each image individually
            det_bboxes = []
            det_labels = []
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            return det_bboxes, det_labels
        elif self.num_bboxtype == 2:
            det_bboxes_h = []
            det_labels_h = []
            det_bboxes = []
            det_labels = []
            for i in range(len(proposals)):
                det_bbox_h, det_label_h, det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score_h[i],
                    bbox_pred_h[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes_h.append(det_bbox_h)
                det_labels_h.append(det_label_h)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            return det_bboxes_h, det_labels_h, det_bboxes, det_labels

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    obb=False,
                    submission=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # proposal_list [n,5]----[x1,y1,x2,y2,score]
        if self.num_bboxtype == 1:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        # det_bboxes is list len=test_batchsize [n, 5] ---[x1, y1, x2, y2, score]
        # det_bboxes is list len=test_batchsize [n]   n is nms = dict(max_per_img)
        elif self.num_bboxtype == 2:
            det_bboxes_h, det_labels_h, det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)


        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        if submission == False:
            if obb:
                bbox_results = [
                    rbbox2result(det_bboxes[i], det_labels[i],
                                 self.bbox_head.num_classes)
                    for i in range(len(det_bboxes))
                ]
            # else:
            #     bbox_results = [
            #         bbox2result(det_bboxes_h[i], det_labels_h[i],
            #                      self.bbox_head.num_classes)
            #         for i in range(len(det_bboxes))
            #     ]
        else:
            bbox_results = {}
            bbox_results['hbb'] = [
                    bbox2result(det_bboxes_h[i], det_labels_h[i],
                                 self.bbox_head.num_classes)
                    for i in range(len(det_bboxes))
                ]
            bbox_results['rbb'] = [
                rbbox2result(det_bboxes[i], det_labels[i],
                             self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]
        # 返回的是一个(list)[(list)[[bbox]...(num.class)],...,(batch)]

        # (list)[(list)[该list内一共有num_class个array，每个array对应的该类里的[bbox(4个), score]],.., (test_batchsize)]
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
