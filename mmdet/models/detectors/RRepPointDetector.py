from ..builder import DETECTORS
from .rsingle_stage import RSingleStageDetector


@DETECTORS.register_module()
class RReppointsDetector(RSingleStageDetector):
    def __init__(self, **kwargs):
        super(RReppointsDetector, self).__init__(**kwargs)