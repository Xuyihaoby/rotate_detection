import torch
import torch.nn as nn
from mmdet.ops import box_iou_rotated_differentiable

from mmdet.models.builder import LOSSES
from .utils import weighted_loss
from mmdet.core.bbox.rtransforms import enclosing_box


@weighted_loss
def iou_loss(pred, target, linear=False, eps=1e-6, version='v1'):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = box_iou_rotated_differentiable(pred, target, version=version).clamp(min=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()
    return loss


@LOSSES.register_module()
class RotatedIoULoss(nn.Module):

    def __init__(self, linear=False, eps=1e-6, reduction='mean', loss_weight=1.0, version='v1'):
        super(RotatedIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            version=self.version,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@weighted_loss
def giou_loss(pred, target, linear=True, eps=1e-6, version='v1', enclosing='smallest'):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious, corners1, corners2, u = box_iou_rotated_differentiable(pred, target, version=version, iou_only=False)
    ious = ious.clamp(min=eps)
    u.clamp(min=eps)
    w, h = enclosing_box(corners1, corners2, enclosing)
    area_c = w * h
    giou = ious - (area_c - u) / area_c
    if linear:
        loss = 1 - giou
    else:
        loss = -giou.log()
    return loss


@LOSSES.register_module()
class RotatedGIoULoss(nn.Module):

    def __init__(self, linear=True, eps=1e-6, reduction='mean', loss_weight=1.0, version='v1', enclosing='smallest'):
        super(RotatedGIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version
        self.enclosing = enclosing

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            version=self.version,
            eps=self.eps,
            enclosing=self.enclosing,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
