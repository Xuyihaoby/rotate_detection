from ..builder import DETECTORS
from .rcascade_rcnn import RCascadeRCNN


@DETECTORS.register_module()
class RoItransformer(RCascadeRCNN):
    def __init__(self, **kwargs):
        super(RoItransformer, self).__init__(**kwargs)
  
        
@DETECTORS.register_module()
class ReDet(RCascadeRCNN):
    def __init__(self, **kwargs):
        super(ReDet, self).__init__(**kwargs)

