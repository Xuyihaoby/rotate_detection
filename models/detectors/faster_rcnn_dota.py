import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, MaskTestMixin, BBoxTestMixinDOTA
from .. import builder
# from ..registry import DETECTORS
from ..builder import DETECTORS
from mmdet.core import bbox2roi, build_assigner, build_sampler, bbox2result_rotate


@DETECTORS.register_module
class FasterRCNNDOTA(BaseDetector, RPNTestMixin, BBoxTestMixinDOTA,
					   MaskTestMixin):

	def __init__(self,
				 backbone,
				 rpn_head=None,
				 bbox_roi_extractor=None,
				 bbox_head=None,
				 mask_roi_extractor=None,
				 mask_head=None,
				 train_cfg=None,
				 test_cfg=None,
				 neck=None,
				 shared_head=None,
				 pretrained=None):
		super(FasterRCNNDOTA, self).__init__()
		self.backbone = builder.build_backbone(backbone)

		if neck is not None:
			self.neck = builder.build_neck(neck)

		if shared_head is not None:
			self.shared_head = builder.build_shared_head(shared_head)

		if rpn_head is not None:
			self.rpn_head = builder.build_head(rpn_head)

		if bbox_head is not None:
			self.bbox_roi_extractor = builder.build_roi_extractor(
				bbox_roi_extractor)
			self.bbox_head = builder.build_head(bbox_head)

		if mask_head is not None:
			self.mask_roi_extractor = builder.build_roi_extractor(
				mask_roi_extractor)
			self.mask_head = builder.build_head(mask_head)

		self.train_cfg = train_cfg
		self.test_cfg = test_cfg

		self.init_weights(pretrained=pretrained)

	@property
	def with_rpn(self):
		return hasattr(self, 'rpn_head') and self.rpn_head is not None

	def init_weights(self, pretrained=None):
		super(FasterRCNNDOTA, self).init_weights(pretrained)
		self.backbone.init_weights(pretrained=pretrained)
		if self.with_neck:
			if isinstance(self.neck, nn.Sequential):
				for m in self.neck:
					m.init_weights()
			else:
				self.neck.init_weights()
		if self.with_rpn:
			self.rpn_head.init_weights()
		if self.with_shared_head:
			self.shared_head.init_weights(pretrained=pretrained)
		if self.with_bbox:
			self.bbox_roi_extractor.init_weights()
			self.bbox_head.init_weights()
		if self.with_mask:
			self.mask_roi_extractor.init_weights()
			self.mask_head.init_weights()

	def extract_feat(self, img):
		x = self.backbone(img) ## [C2, C3, C4, C5]
		if self.with_neck:
			x = self.neck(x) ## [P2, P3, P4, P5, P6]
		return x

	def forward_train(self,
					  img,
					  img_meta,
					  gt_bboxes, ## list, each is [G, 5(x_ctr,y_ctr, w,h, theta)]
					  hor_gt_boxes, ## list, each is [G, 4(x1,y1, x2,y2)]
					  gt_bboxes_ignore, ## list, each is [?, 4]
					  gt_labels, ## list, each is [G]
					  gt_masks=None,
					  proposals=None):
		x = self.extract_feat(img) ## [P2, P3, P4, P5, P6]

		losses = dict()
		# import pdb
		# pdb.set_trace()
		# print('gt_bboxes', gt_bboxes)
		# print('hor_gt_boxes', hor_gt_boxes)
		# print('gt_labels', gt_labels)

		# RPN forward and loss
		if self.with_rpn:
			rpn_outs = self.rpn_head(x) ## rpn_cls_score[P2,P3,..], rpn_bbox_pred[P2,P3,..]
			rpn_loss_inputs = rpn_outs + (hor_gt_boxes, img_meta,
										  self.train_cfg.rpn)
			rpn_losses = self.rpn_head.loss(*rpn_loss_inputs) ## sample 256 label and <128 bbox to compute loss.
			losses.update(rpn_losses)

			proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
			proposal_list = self.rpn_head.get_bboxes(*proposal_inputs) ## list, each is [2000, 5(x1,y1, x2,y2, score)]
			# ls_cl = losses['loss_cls'][0] + losses['loss_cls'][1] + losses['loss_cls'][2] + losses['loss_cls'][3] + losses['loss_cls'][4]
			# ls_re = losses['loss_reg'][0] + losses['loss_reg'][1] + losses['loss_reg'][2] + losses['loss_reg'][3] + losses['loss_reg'][4]
			# if ls_cl > 100:
			#	 import pdb
			#	 pdb.set_trace()
			# else:
			#	 print('class = {}, reg = {}.'.format(ls_cl, ls_re))
		else:
			proposal_list = proposals

		# assign gts and sample proposals
		if self.with_bbox or self.with_mask:
			bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
			bbox_sampler = build_sampler(
				self.train_cfg.rcnn.sampler, context=self)
			num_imgs = img.size(0)
			sampling_results = []
			for i in range(num_imgs):
				assign_result = bbox_assigner.assign(
					proposal_list[i], hor_gt_boxes[i], gt_bboxes_ignore[i],
					gt_labels[i])
				sampling_result = bbox_sampler.sample(
					assign_result,
					proposal_list[i],
					hor_gt_boxes[i],
					gt_labels[i],
					feats=[lvl_feat[i][None] for lvl_feat in x]) ## sample 512 bbox from 2000 proposals to forward.
				sampling_results.append(sampling_result)

		# bbox head forward and loss
		if self.with_bbox:
			rois = bbox2roi([res.bboxes for res in sampling_results]) ## [batch_ind, x1, y1, x2, y2]
			# TODO: a more flexible way to decide which feature maps to use
			# import pdb
			# pdb.set_trace()
			bbox_feats = self.bbox_roi_extractor(
				x[:self.bbox_roi_extractor.num_inputs], rois) ## [N*512, 256, 7, 7]
			if self.with_shared_head:
				bbox_feats = self.shared_head(bbox_feats)
			## [N*512, 16], [N*512, 16*4], [N*512, 16], [N*512, 16*5]
			cls_score_h, bbox_pred_h, cls_score_r, bbox_pred_r = self.bbox_head(bbox_feats)

			bbox_targets = self.bbox_head.get_target(
				sampling_results, gt_bboxes, self.train_cfg.rcnn)
			loss_bbox = self.bbox_head.loss(cls_score_h, bbox_pred_h, cls_score_r, bbox_pred_r,
											*bbox_targets)
			losses.update(loss_bbox)

		return losses

	def simple_test(self, img, img_meta, proposals=None, rescale=False):
		"""Test without augmentation."""
		assert self.with_bbox, "Bbox head must be implemented."

		x = self.extract_feat(img) ## [P2, P3, P4, P5, P6]

		## list, each is [2000, 5(x1,y1, x2,y2, score)]
		proposal_list = self.simple_test_rpn(
						x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

		## [K, 5(x1,y1, x2,y2, score)]
		## [K(0-start)]
		## [K, 6(x_ctr,y_ctr, w,h, theta, score)]
		## [K(0-start)]
		det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r = self.simple_test_bboxes(
						x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

		## dict(horizontal = results_h, rotate = results_r), label is 1-start
		## convert from 0-start to 1-start, just for use merge_func
		bbox_results = bbox2result_rotate(det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r)

		if not self.with_mask:
			return bbox_results
		else:
			segm_results = self.simple_test_mask(
				x, img_meta, det_bboxes, det_labels, rescale=rescale)
			return bbox_results, segm_results

	def aug_test(self, imgs, img_metas, rescale=False):
		"""Test with augmentations.

		If rescale is False, then returned bboxes and masks will fit the scale
		of imgs[0].
		"""
		# recompute feats to save memory
		proposal_list = self.aug_test_rpn(
			self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
		det_bboxes, det_labels = self.aug_test_bboxes(
			self.extract_feats(imgs), img_metas, proposal_list,
			self.test_cfg.rcnn)

		if rescale:
			_det_bboxes = det_bboxes
		else:
			_det_bboxes = det_bboxes.clone()
			_det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
		bbox_results = bbox2result(_det_bboxes, det_labels,
								   self.bbox_head.num_classes)

		# det_bboxes always keep the original scale
		if self.with_mask:
			segm_results = self.aug_test_mask(
				self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
			return bbox_results, segm_results
		else:
			return bbox_results
