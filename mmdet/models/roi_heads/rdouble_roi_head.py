from ..builder import HEADS
from .rstandard_roi_head import RStandardRoIHead


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
