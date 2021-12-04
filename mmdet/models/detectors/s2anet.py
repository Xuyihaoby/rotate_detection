from .rretinanet import RRetinaNet
from ..builder import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module()
class S2ANet(RRetinaNet):
    def __init__(self, **kwargs):
        super(S2ANet, self).__init__(**kwargs)