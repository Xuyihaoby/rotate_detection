# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from .polygon_geo import polygon_iou
from .rbbox_geo import rbbox_iou_iof

from .nms import batched_rnms, rnms
from .orn import ORConv2d, RotationInvariantPooling, rotation_invariant_encoding
from .roi_align_rotated import RoIAlignRotated
from .riroi_align import RiRoIAlign
from .fr import FeatureRefineModule
# yapf: enable

__all__ = [
    'polygon_iou',
    "batched_rnms", "rnms",
    "rbbox_iou_iof",
    'RotationInvariantPooling', 'rotation_invariant_encoding',
    'ORConv2d',
    'RoIAlignRotated',
    'RiRoIAlign',
    'FeatureRefineModule'
]
