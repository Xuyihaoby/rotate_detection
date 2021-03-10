# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from .polygon_geo import polygon_iou

# yapf: enable

__all__ = [
    'polygon_iou'
]
