from ..builder import DETECTORS
from .rsingle_stage import RSingleStageDetector


@DETECTORS.register_module()
class RFCOS(RSingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
