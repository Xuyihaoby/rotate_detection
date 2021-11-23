from ..builder import HEADS
from .rstandard_roi_head import RStandardRoIHead
from .oriented_roi_head import OrientedRoIHead
from mmdet.core import CV_L_Rad2LE_DEF_TORCH


@HEADS.register_module()
class RDoubleHeadRoIHead(RStandardRoIHead):
    """RoI head for Double Head RCNN.

    https://arxiv.org/abs/1904.06493
    """

    def __init__(self, reg_roi_scale_factor, **kwargs):
        super(RDoubleHeadRoIHead, self).__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score_h, bbox_pred_h, cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            cls_score_h=cls_score_h,
            bbox_pred=bbox_pred,
            bbox_pred_h=bbox_pred_h,
            bbox_feats=bbox_cls_feats)
        return bbox_results

@HEADS.register_module()
class DoubleOrientedRoIHead(OrientedRoIHead):
    def __init__(self, **kwargs):
        super(DoubleOrientedRoIHead, self).__init__(**kwargs)

    def _bbox_forward(self, x, rois, roi_format='CV_LEFT'):
        _rois = rois.clone()
        if roi_format == 'CV_LEFT':
            _rois[:, 1:] = CV_L_Rad2LE_DEF_TORCH(_rois[:, 1:])
            _rois[:, -1] = -_rois[:, -1]

        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois)
        # TODO roi scale factor
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)

        # TODO add horizon braunch
        if self.num_bboxtype == 1:
            cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred,
                bbox_feats=bbox_cls_feats)
        elif self.num_bboxtype == 2:
            cls_score_h, bbox_pred_h, cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, cls_score_h=cls_score_h, bbox_pred_h=bbox_pred_h,
                bbox_feats=bbox_cls_feats)

        return bbox_results