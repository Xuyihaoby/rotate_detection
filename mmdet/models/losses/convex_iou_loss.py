import pdb

import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from mmdet.ops.iou import convex_giou

'''
gt box and  convex hull of points set for giou loss
from SDL-GuoZonghao:https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/models/losses/iou_loss.py
'''


class ConvexGIoULossFuction(Function):
    @staticmethod
    def forward(ctx, pred, target, weight=None, reduction=None, avg_factor=None, loss_weight=1.0):
        ctx.save_for_backward(pred)
        # convex_gious: tensor.size(n)  grad: tensor.size(n, 18)
        convex_gious, grad = convex_giou(pred, target)
        loss = 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            # grad = grad * weight.reshape(-1, 1)
            grad = grad * weight[..., None]
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()
            # loss = loss.sum()/avg_factor
        else:
            loss = loss

        # _unvalid_grad_filter
        eps = 1e-6
        unvaild_inds = torch.nonzero((grad > 1).sum(1))[:, 0]
        grad[unvaild_inds] = eps
        #
        # # _reduce_grad
        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None


convex_giou_loss = ConvexGIoULossFuction.apply


@LOSSES.register_module()
class ConvexGIoULoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ConvexGIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * convex_giou_loss(
            pred,
            target,
            weight,
            reduction,
            avg_factor,
            self.loss_weight)
        return loss
