import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin, RBBoxTestMixin
from mmdet.core import rbbox2result


@HEADS.register_module()
class HSPRoIHead(BaseRoIHead, RBBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

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
                      img,
                      seg_fea,
                      mask_lvls,
                      hor_gt_bboxes_ignore=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        # x = [[batch_size,RPNout_channel ,H, W],...(num_lvls)]
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                hor_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], hor_gt_bboxes[i], hor_gt_bboxes_ignore[i],
                    gt_labels[i])  #
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    hor_gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # 如果正样本不足的话，那也只能将所有正样本带上剩下的都是负样本
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, img, seg_fea, mask_lvls)

            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, img, seg_fea, mask_lvls):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois, img=img, seg_fea=seg_fea, mask_lvls=mask_lvls)
        # len(bbox_feats) = 6 [P2-P5, seg, img]
        # [batchsize * rpn_out_channel, bbox_head_in_channels, 7, 7]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score_h, bbox_pred_h, cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, cls_score_h=cls_score_h, bbox_pred_h=bbox_pred_h,
            bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, img, seg_fea, mask_lvls):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # sampling result list len=batchsize
        # rois是res sampling之后的结果etc.[512, 4] -->[n, 5], [batch_ind, x1, y1, x2, y2]
        bbox_results = self._bbox_forward(x, rois, img, seg_fea, mask_lvls)

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

    def simple_test(self,
                    x,
                    img,
                    proposal_list,
                    seg_fea,
                    mask_lvls,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    obb=False,
                    submission=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # proposal_list [n,5]----[x1,y1,x2,y2,score]

        det_bboxes_h, det_labels_h, det_bboxes, det_labels = self.simple_test_bboxes(
            x, img, img_metas, proposal_list, seg_fea, mask_lvls, self.test_cfg, rescale=rescale)
        # det_bboxes_h is list len=test_batchsize [n, 5] ---[x1, y1, x2, y2, score]
        # det_labels_h is list len=test_batchsize [n]   n is nms = dict(max_per_img)

        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes_h, det_labels_h, det_bboxes, det_labels

        if submission == False:
            if obb:
                bbox_results = [
                    rbbox2result(det_bboxes[i], det_labels[i],
                                 self.bbox_head.num_classes)
                    for i in range(len(det_bboxes))
                ]
            else:
                bbox_results = [
                    bbox2result(det_bboxes_h[i], det_labels_h[i],
                                self.bbox_head.num_classes)
                    for i in range(len(det_bboxes))
                ]
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

    def simple_test_bboxes(self,
                           x,
                           img,
                           img_metas,
                           proposals,
                           seg_fea,
                           mask_lvls,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        # rois [1000, 5] [x1, y1, x2, y2, scores] ---> [batchind, x1, y1, x2, y2]
        bbox_results = self._bbox_forward(x, rois, img, seg_fea, mask_lvls)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']  # size [num_proposals, num_class+1]
        cls_score_h = bbox_results['cls_score_h']
        bbox_pred = bbox_results['bbox_pred']  # size [num_proposals, 5]
        bbox_pred_h = bbox_results['bbox_pred_h']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)  # tuple(per_img_bbox_pred)
        cls_score = cls_score.split(num_proposals_per_img, 0)  # tuple(per_img_bbox_pred_cls)
        cls_score_h = cls_score_h.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        # rotate
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)  # tuple(per_img_bbox_pred)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # horizon
        if bbox_pred_h is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred_h, torch.Tensor):
                bbox_pred_h = bbox_pred_h.split(num_proposals_per_img, 0)  # tuple(per_img_bbox_pred)
            else:
                bbox_pred_h = self.bbox_head_h.bbox_pred_split(
                    bbox_pred_h, num_proposals_per_img)
        else:
            bbox_pred_h = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
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
        # return det_bboxes, det_labels
