from mmdet.ops import rbbox_iou_iof, obb_overlaps
from mmdet.ops.iou import convex_overlaps
from mmcv.ops import box_iou_rotated
from .builder import IOU_CALCULATORS
from mmdet.core.bbox.rtransforms import obb2poly

@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D(object):
    """2D IoU Calculator"""

    def __init__(self, version='v1'):
        self.version = version

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 5) in <xc, yc, w, h, alpha>
                format, or shape (m, 6) in <xc, yc, w, h, alpha, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 5) in <xc, yc, w, h, alpha>
                format, shape (m, 6) in <xc, yc, w, h, alpha, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5].contiguous()
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5].contiguous()
        if self.version == 'v1':
            return rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        elif self.version == 'v2':
            return rbbox_overlaps_v2(bboxes1, bboxes2, mode, is_aligned)
        elif self.version == 'v3':
            return rbbox_overlaps_v3(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    return rbbox_iou_iof(bboxes1, bboxes2, is_aligned, (mode == 'iof'))


def rbbox_overlaps_v2(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    return box_iou_rotated(bboxes1, bboxes2, mode, is_aligned)


def rbbox_overlaps_v3(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    return obb_overlaps(bboxes1, bboxes2, mode, is_aligned)

@IOU_CALCULATORS.register_module()
class ConvexOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, version='v1'):
        self.version = version

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        if bboxes1.size(-1) == 5:
            bboxes1 = obb2poly(bboxes1, version=self.version)
        assert bboxes1.size(-1) in [0, 8]
        assert bboxes2.size(-1) in [0, 18]
        return convex_overlaps(bboxes1, bboxes2)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
