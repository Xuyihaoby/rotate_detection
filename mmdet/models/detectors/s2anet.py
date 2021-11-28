from .rretinanet import RRetinaNet
import torch
from ..builder import DETECTORS

@DETECTORS.register_module()
class S2ANet(RRetinaNet):
    def __init__(self, **kwargs):
        super(S2ANet, self).__init__(**kwargs)