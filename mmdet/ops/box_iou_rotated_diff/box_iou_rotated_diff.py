"""
Differentiable IoU calculation for rotated boxes
Most of the code is adapted from https://github.com/lilanxiao/Rotated_IoU
"""
import torch
from .box_intersection_2d import oriented_box_intersection_2d
from mmdet.core.bbox.rtransforms import obb2poly


def box_iou_rotated_differentiable(boxes1: torch.Tensor, boxes2: torch.Tensor, iou_only: bool = True, version: str='v1'):
    """Calculate IoU between rotated boxes

    Args:
        version: str
        box1 (torch.Tensor): (n, 5)
        box2 (torch.Tensor): (n, 5)
        iou_only: Whether to keep other vars, e.g., polys, unions. Default True to drop these vars.

    Returns:
        iou (torch.Tensor): (n, )
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        U (torch.Tensor): (n) area1 + area2 - inter_area
    """
    # transform to polygons
    polys1 = obb2poly(boxes1, version)
    polys1 = polys1.view(polys1.shape[0], 4, 2)
    polys2 = obb2poly(boxes2, version)
    polys2 = polys2.view(polys1.shape[0], 4, 2)
    # calculate insection areas
    inter_area, _ = oriented_box_intersection_2d(polys1, polys2)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union = area1 + area2 - inter_area
    iou = inter_area / union
    if iou_only:
        return iou
    else:
        return iou, polys1, polys2, union
