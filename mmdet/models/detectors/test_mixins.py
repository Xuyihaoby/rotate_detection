from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
						merge_aug_bboxes, merge_aug_masks, multiclass_nms)


class RPNTestMixin(object):

	def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
		rpn_outs = self.rpn_head(x)
		proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
		proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
		return proposal_list

	def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
		imgs_per_gpu = len(img_metas[0])
		aug_proposals = [[] for _ in range(imgs_per_gpu)]
		for x, img_meta in zip(feats, img_metas):
			proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
			for i, proposals in enumerate(proposal_list):
				aug_proposals[i].append(proposals)
		# after merging, proposals will be rescaled to the original image size
		merged_proposals = [
			merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
			for proposals, img_meta in zip(aug_proposals, img_metas)
		]
		return merged_proposals


class BBoxTestMixin(object):

	def simple_test_bboxes(self,
						   x,
						   img_meta,
						   proposals,
						   rcnn_test_cfg,
						   rescale=False):
		"""Test only det bboxes without augmentation."""
		rois = bbox2roi(proposals)
		roi_feats = self.bbox_roi_extractor(
			x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
		if self.with_shared_head:
			roi_feats = self.shared_head(roi_feats)
		cls_score, bbox_pred = self.bbox_head(roi_feats)
		img_shape = img_meta[0]['img_shape']
		scale_factor = img_meta[0]['scale_factor']
		det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
			rois,
			cls_score,
			bbox_pred,
			img_shape,
			scale_factor,
			rescale=rescale,
			cfg=rcnn_test_cfg)
		return det_bboxes, det_labels

	def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
		aug_bboxes = []
		aug_scores = []
		for x, img_meta in zip(feats, img_metas):
			# only one image in the batch
			img_shape = img_meta[0]['img_shape']
			scale_factor = img_meta[0]['scale_factor']
			flip = img_meta[0]['flip']
			# TODO more flexible
			proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
									 scale_factor, flip)
			rois = bbox2roi([proposals])
			# recompute feature maps to save GPU memory
			roi_feats = self.bbox_roi_extractor(
				x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
			if self.with_shared_head:
				roi_feats = self.shared_head(roi_feats)
			cls_score, bbox_pred = self.bbox_head(roi_feats)
			bboxes, scores = self.bbox_head.get_det_bboxes(
				rois,
				cls_score,
				bbox_pred,
				img_shape,
				scale_factor,
				rescale=False,
				cfg=None)
			aug_bboxes.append(bboxes)
			aug_scores.append(scores)
		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes, merged_scores = merge_aug_bboxes(
			aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
		det_bboxes, det_labels = multiclass_nms(
			merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
			rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
		return det_bboxes, det_labels


class MaskTestMixin(object):

	def simple_test_mask(self,
						 x,
						 img_meta,
						 det_bboxes,
						 det_labels,
						 rescale=False):
		# image shape of the first image in the batch (only one)
		ori_shape = img_meta[0]['ori_shape']
		scale_factor = img_meta[0]['scale_factor']
		if det_bboxes.shape[0] == 0:
			segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
		else:
			# if det_bboxes is rescaled to the original image size, we need to
			# rescale it back to the testing scale to obtain RoIs.
			_bboxes = (det_bboxes[:, :4] * scale_factor
					   if rescale else det_bboxes)
			mask_rois = bbox2roi([_bboxes])
			mask_feats = self.mask_roi_extractor(
				x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
			if self.with_shared_head:
				mask_feats = self.shared_head(mask_feats)
			mask_pred = self.mask_head(mask_feats)
			segm_result = self.mask_head.get_seg_masks(
				mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
				scale_factor, rescale)
		return segm_result

	def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
		if det_bboxes.shape[0] == 0:
			segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
		else:
			aug_masks = []
			for x, img_meta in zip(feats, img_metas):
				img_shape = img_meta[0]['img_shape']
				scale_factor = img_meta[0]['scale_factor']
				flip = img_meta[0]['flip']
				_bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
									   scale_factor, flip)
				mask_rois = bbox2roi([_bboxes])
				mask_feats = self.mask_roi_extractor(
					x[:len(self.mask_roi_extractor.featmap_strides)],
					mask_rois)
				if self.with_shared_head:
					mask_feats = self.shared_head(mask_feats)
				mask_pred = self.mask_head(mask_feats)
				# convert to numpy array to save memory
				aug_masks.append(mask_pred.sigmoid().cpu().numpy())
			merged_masks = merge_aug_masks(aug_masks, img_metas,
										   self.test_cfg.rcnn)

			ori_shape = img_metas[0][0]['ori_shape']
			segm_result = self.mask_head.get_seg_masks(
				merged_masks,
				det_bboxes,
				det_labels,
				self.test_cfg.rcnn,
				ori_shape,
				scale_factor=1.0,
				rescale=False)
		return segm_result

########## Updated By LCZ, time: 2019.1.19 ##########
class FeatureAttenBBoxTestMixin(object):

	def simple_test_bboxes(self,
						   x,
						   img,
						   img_meta,
						   proposals,
						   rcnn_test_cfg,
						   rescale=False):
		"""Test only det bboxes without augmentation."""
		rois = bbox2roi(proposals)
		roi_feats = self.bbox_roi_extractor(
			x[:self.bbox_roi_extractor.num_inputs], rois, norm_img = img)
		if self.with_shared_head:
			raise NotImplementedError
		cls_score, bbox_pred = self.bbox_head(roi_feats)
		img_shape = img_meta[0]['img_shape']
		scale_factor = img_meta[0]['scale_factor']
		det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
			rois,
			cls_score,
			bbox_pred,
			img_shape,
			scale_factor,
			rescale=rescale,
			cfg=rcnn_test_cfg)
		return det_bboxes, det_labels

	def aug_test_bboxes(self, feats, imgs, img_metas, proposal_list, rcnn_test_cfg):
		aug_bboxes = []
		aug_scores = []
		for x, img, img_meta in zip(feats, imgs, img_metas):
			# only one image in the batch
			img_shape = img_meta[0]['img_shape']
			scale_factor = img_meta[0]['scale_factor']
			flip = img_meta[0]['flip']
			# TODO more flexible
			proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
									 scale_factor, flip)
			rois = bbox2roi([proposals])
			# recompute feature maps to save GPU memory
			roi_feats = self.bbox_roi_extractor(
				x[:self.bbox_roi_extractor.num_inputs], rois, norm_img = img)
			if self.with_shared_head:
				raise NotImplementedError
			cls_score, bbox_pred = self.bbox_head(roi_feats)
			bboxes, scores = self.bbox_head.get_det_bboxes(
				rois,
				cls_score,
				bbox_pred,
				img_shape,
				scale_factor,
				rescale=False,
				cfg=None)
			aug_bboxes.append(bboxes)
			aug_scores.append(scores)
		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes, merged_scores = merge_aug_bboxes(
			aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
		det_bboxes, det_labels = multiclass_nms(
			merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
			rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
		return det_bboxes, det_labels

########## Updated By LCZ, time: 2019.3.3 ##########
class BBoxTestMixinDOTA(object):

	def simple_test_bboxes(self,
						   x,
						   img_meta,
						   proposals,
						   rcnn_test_cfg,
						   rescale=False):
		"""Test only det bboxes without augmentation."""
		## list, each is [2000, 5(x1,y1, x2,y2, score)]
		rois = bbox2roi(proposals) ## (2000*5, 5(batch_ind, x1,y1, x2,y2))
		roi_feats = self.bbox_roi_extractor(
			x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
		if self.with_shared_head:
			raise NotImplementedError
		## [2000, 16], [2000, 16*4], [2000, 16], [2000, 16*5]
		cls_score_h, bbox_pred_h, cls_score_r, bbox_pred_r = self.bbox_head(roi_feats)
		img_shape = img_meta[0]['img_shape']
		scale_factor = img_meta[0]['scale_factor']
		## [K, 5(x1,y1, x2,y2, score)]
		## [K(0-start)]
		## [K, 6(x_ctr,y_ctr, w,h, theta, score)]
		## [K(0-start)]
		det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r = self.bbox_head.get_det_bboxes(
			rois,
			cls_score_h,
			bbox_pred_h,
			cls_score_r, 
			bbox_pred_r, 
			img_shape,
			scale_factor,
			rescale=rescale,
			cfg=rcnn_test_cfg)
		return det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r

	def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
		aug_bboxes = []
		aug_scores = []
		for x, img_meta in zip(feats, img_metas):
			# only one image in the batch
			img_shape = img_meta[0]['img_shape']
			scale_factor = img_meta[0]['scale_factor']
			flip = img_meta[0]['flip']
			# TODO more flexible
			proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
									 scale_factor, flip)
			rois = bbox2roi([proposals])
			# recompute feature maps to save GPU memory
			roi_feats = self.bbox_roi_extractor(
				x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
			if self.with_shared_head:
				raise NotImplementedError
			cls_score, bbox_pred = self.bbox_head(roi_feats)
			bboxes, scores = self.bbox_head.get_det_bboxes(
				rois,
				cls_score,
				bbox_pred,
				img_shape,
				scale_factor,
				rescale=False,
				cfg=None)
			aug_bboxes.append(bboxes)
			aug_scores.append(scores)
		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes, merged_scores = merge_aug_bboxes(
			aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
		det_bboxes, det_labels = multiclass_nms(
			merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
			rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
		return det_bboxes, det_labels


########## Updated By LCZ, time: 2019.3.17 ##########
class FeatureAttenCrossLvlBBoxTestMixinDOTA(object):

	def simple_test_bboxes(self,
						   x,
						   img, 
						   img_meta,
						   proposals,
						   rcnn_test_cfg,
						   rescale=False):
		"""Test only det bboxes without augmentation."""
		## list, each is [2000, 5(x1,y1, x2,y2, score)]
		rois = bbox2roi(proposals) ## (2000*5, 5(batch_ind, x1,y1, x2,y2))
		roi_feats, target_lvls = self.bbox_roi_extractor(
			x[:self.bbox_roi_extractor.num_inputs], rois, norm_img = img)
		if self.with_shared_head:
			raise NotImplementedError
		## [2000, 16], [2000, 16*4], [2000, 16], [2000, 16*5]
		cls_score_h, bbox_pred_h, cls_score_r, bbox_pred_r = self.bbox_head(roi_feats, target_lvls)
		img_shape = img_meta[0]['img_shape']
		scale_factor = img_meta[0]['scale_factor']
		## [K, 5(x1,y1, x2,y2, score)]
		## [K(0-start)]
		## [K, 6(x_ctr,y_ctr, w,h, theta, score)]
		## [K(0-start)]
		det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r = self.bbox_head.get_det_bboxes(
			rois,
			cls_score_h,
			bbox_pred_h,
			cls_score_r, 
			bbox_pred_r, 
			img_shape,
			scale_factor,
			rescale=rescale,
			cfg=rcnn_test_cfg)
		return det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r

	def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
		aug_bboxes = []
		aug_scores = []
		for x, img_meta in zip(feats, img_metas):
			# only one image in the batch
			img_shape = img_meta[0]['img_shape']
			scale_factor = img_meta[0]['scale_factor']
			flip = img_meta[0]['flip']
			# TODO more flexible
			proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
									 scale_factor, flip)
			rois = bbox2roi([proposals])
			# recompute feature maps to save GPU memory
			roi_feats = self.bbox_roi_extractor(
				x[:self.bbox_roi_extractor.num_inputs], rois, norm_img = img)
			if self.with_shared_head:
				raise NotImplementedError
			cls_score, bbox_pred = self.bbox_head(roi_feats)
			bboxes, scores = self.bbox_head.get_det_bboxes(
				rois,
				cls_score,
				bbox_pred,
				img_shape,
				scale_factor,
				rescale=False,
				cfg=None)
			aug_bboxes.append(bboxes)
			aug_scores.append(scores)
		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes, merged_scores = merge_aug_bboxes(
			aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
		det_bboxes, det_labels = multiclass_nms(
			merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
			rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
		return det_bboxes, det_labels


########## Updated By LCZ, time: 2019.3.22 ##########
class FeatureAttenBBoxTestMixinDOTA(object):

	def simple_test_bboxes(self,
						   x,
						   img, 
						   img_meta,
						   proposals,
						   rcnn_test_cfg,
						   rescale=False):
		"""Test only det bboxes without augmentation."""
		## list, each is [2000, 5(x1,y1, x2,y2, score)]
		rois = bbox2roi(proposals) ## (2000*5, 5(batch_ind, x1,y1, x2,y2))
		roi_feats = self.bbox_roi_extractor(
			x[:self.bbox_roi_extractor.num_inputs], rois, norm_img = img)
		if self.with_shared_head:
			raise NotImplementedError
		## [2000, 16], [2000, 16*4], [2000, 16], [2000, 16*5]
		cls_score_h, bbox_pred_h, cls_score_r, bbox_pred_r = self.bbox_head(roi_feats)
		img_shape = img_meta[0]['img_shape']
		scale_factor = img_meta[0]['scale_factor']
		## [K, 5(x1,y1, x2,y2, score)]
		## [K(0-start)]
		## [K, 6(x_ctr,y_ctr, w,h, theta, score)]
		## [K(0-start)]
		det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r = self.bbox_head.get_det_bboxes(
			rois,
			cls_score_h,
			bbox_pred_h,
			cls_score_r, 
			bbox_pred_r, 
			img_shape,
			scale_factor,
			rescale=rescale,
			cfg=rcnn_test_cfg)
		return det_bboxes_h, det_labels_h, det_bboxes_r, det_labels_r

	def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
		aug_bboxes = []
		aug_scores = []
		for x, img_meta in zip(feats, img_metas):
			# only one image in the batch
			img_shape = img_meta[0]['img_shape']
			scale_factor = img_meta[0]['scale_factor']
			flip = img_meta[0]['flip']
			# TODO more flexible
			proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
									 scale_factor, flip)
			rois = bbox2roi([proposals])
			# recompute feature maps to save GPU memory
			roi_feats = self.bbox_roi_extractor(
				x[:self.bbox_roi_extractor.num_inputs], rois, norm_img = img)
			if self.with_shared_head:
				raise NotImplementedError
			cls_score, bbox_pred = self.bbox_head(roi_feats)
			bboxes, scores = self.bbox_head.get_det_bboxes(
				rois,
				cls_score,
				bbox_pred,
				img_shape,
				scale_factor,
				rescale=False,
				cfg=None)
			aug_bboxes.append(bboxes)
			aug_scores.append(scores)
		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes, merged_scores = merge_aug_bboxes(
			aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
		det_bboxes, det_labels = multiclass_nms(
			merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
			rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
		return det_bboxes, det_labels