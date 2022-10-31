from ..builder import DETECTORS
from .rsingle_stage import RSingleStageDetector


@DETECTORS.register_module()
class RPAA(RSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RPAA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
