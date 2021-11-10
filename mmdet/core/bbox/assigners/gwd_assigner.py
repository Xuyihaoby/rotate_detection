import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUGWDAssigner(BaseAssigner):
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
                gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        # [num_gt num_anchors]
        # compute the gwd_loss
        gwd_overlaps = (gwd_loss(bboxes.repeat(gt_bboxes.shape[0], 1)[:, :5],
                                gt_bboxes.repeat(1, bboxes.shape[0]).reshape(-1, 5), tau=0.)).reshape(-1, bboxes.shape[0])
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gwd_overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        return assign_result

    def assign_wrt_overlaps(self, overlaps, gwd_distance, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each anchor, the min distance of all gts
        min_gwd_distance, argmin_gwd_distance = gwd_distance.min(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # modified by xyh
        # 3. assign positive: above positive IoU threshold
        # pos_inds = max_overlaps >= self.pos_iou_thr
        # assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
        pos_inds = min_gwd_distance <= 20
        assigned_gt_inds[pos_inds] = argmin_gwd_distance[pos_inds] + 1

        # 暂时先不考虑低质量锚框的情况
        # if self.match_low_quality:
        #     # Low-quality matching will overwirte the assigned_gt_inds assigned
        #     # in Step 3. Thus, the assigned gt might not be the best one for
        #     # prediction.
        #     # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
        #     # bbox 1 will be assigned as the best target for bbox A in step 3.
        #     # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
        #     # assigned_gt_inds will be overwritten to be bbox B.
        #     # This might be the reason that it is not used in ROI Heads.
        #     for i in range(num_gts):
        #         if gt_max_overlaps[i] >= self.min_pos_iou:
        #             if self.gt_max_assign_all:
        #                 max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
        #                 assigned_gt_inds[max_iou_inds] = i + 1
        #             else:
        #                 assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


# --------------------------------------------------------
# gwd pytorch version

# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license
# --------------------------------------------------------

# modified by xyh
from torch.autograd import Variable
import numpy as np
import torch


# from ..builder import BBOX_ASSIGNERS
# from ..iou_calculators import build_iou_calculator
# from .assign_result import AssignResult
# from .base_assigner import BaseAssigner

# from ..builder import LOSSES
# from .utils import weighted_loss


def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False)
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def wasserstein_distance_sigma(sigma1, sigma2):
    wasserstein_distance_item2 = torch.matmul(sigma1, sigma1) + torch.matmul(sigma2,
                                                                             sigma2) - 2 * sqrt_newton_schulz_autograd(
        torch.matmul(torch.matmul(sigma1, torch.matmul(sigma2, sigma2)), sigma1), 20, torch.FloatTensor)
    wasserstein_distance_item2 = trace(wasserstein_distance_item2)

    return wasserstein_distance_item2


# @weighted_loss
def gwds_loss(pred, target, weight, eps=1e-6):
    """IoU loss.
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.
    Args:
        pred (Tensor): Predicted bboxes of format (xc, yc, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    mask = (weight > 0).detach()
    pred = pred[mask]
    target = target[mask]

    x1 = pred[:, 0]
    y1 = pred[:, 1]
    w1 = pred[:, 2]
    h1 = pred[:, 3]
    theta1 = pred[:, 4]

    sigma1_1 = w1 / 2 * torch.cos(theta1) ** 2 + h1 / 2 * torch.sin(theta1) ** 2
    sigma1_2 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_3 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_4 = w1 / 2 * torch.sin(theta1) ** 2 + h1 / 2 * torch.cos(theta1) ** 2
    sigma1 = torch.reshape(
        torch.cat((sigma1_1.unsqueeze(1), sigma1_2.unsqueeze(1), sigma1_3.unsqueeze(1), sigma1_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    x2 = target[:, 0]
    y2 = target[:, 1]
    w2 = target[:, 2]
    h2 = target[:, 3]
    theta2 = target[:, 4]
    sigma2_1 = w2 / 2 * torch.cos(theta2) ** 2 + h2 / 2 * torch.sin(theta2) ** 2
    sigma2_2 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_3 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_4 = w2 / 2 * torch.sin(theta2) ** 2 + h2 / 2 * torch.cos(theta2) ** 2
    sigma2 = torch.reshape(
        torch.cat((sigma2_1.unsqueeze(1), sigma2_2.unsqueeze(1), sigma2_3.unsqueeze(1), sigma2_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    wasserstein_distance_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    wasserstein_distance_item2 = wasserstein_distance_sigma(sigma1, sigma2)
    wasserstein_distance = torch.max(wasserstein_distance_item1 + wasserstein_distance_item2,
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor),
                                              requires_grad=False))
    wasserstein_distance = torch.max(torch.sqrt(wasserstein_distance + eps),
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor),
                                              requires_grad=False))
    wasserstein_similarity = 1 / (wasserstein_distance + 2)
    wasserstein_loss = 1 - wasserstein_similarity

    return wasserstein_loss


def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S


def gwd_loss(pred, target, fun='sqrt', tau=1.0, alpha=1.0, normalize=False):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z
    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2
    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)
    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)
    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    whr_distance = S_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whr_distance = whr_distance + S_t.diagonal(dim1=-2, dim2=-1).square().sum(
        dim=-1)
    _t = Sigma_p.matmul(Sigma_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0)
    # distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        wh_p = pred[..., 2:4].clamp(min=1e-7, max=1e7)
        wh_t = target[..., 2:4].clamp(min=1e-7, max=1e7)
        scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
        distance = distance / scale

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    else:
        raise ValueError('Invalid non-linear function {fun} for gwd loss')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance

    # @LOSSES.register_module()
    # class GWDLoss(nn.Module):

    #     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
    #         super(GWDLoss, self).__init__()
    #         self.eps = eps
    #         self.reduction = reduction
    #         self.loss_weight = loss_weight

    #     def forward(self,
    #                 pred,
    #                 target,
    #                 weight=None,
    #                 avg_factor=None,
    #                 reduction_override=None,
    #                 **kwargs):
    #         assert reduction_override in (None, 'none', 'mean', 'sum')
    #         reduction = (
    #             reduction_override if reduction_override else self.reduction)
    #         if (weight is not None) and (not torch.any(weight > 0)) and (
    #                 reduction != 'none'):
    #             return (pred * weight).sum()  # 0
    #         if weight is not None and weight.dim() > 1:
    #             assert weight.shape == pred.shape
    #             weight = weight.mean(-1)
    #         loss = self.loss_weight * gwd_loss(
    #             pred,
    #             target,
    #             weight,
    #             eps=self.eps,
    #             # reduction=reduction,
    #             # avg_factor=avg_factor,
    #             **kwargs)
    #         return loss


if __name__ == '__main__':
    # pred = torch.FloatTensor([[50, 50, 10, 70, -30 / 180 * np.pi],
    #                           [50, 50, 10, 70, -30 / 180 * np.pi]])
    # target = torch.FloatTensor([[50, 50, 10, 70, -60 / 180 * np.pi],
    #                             [50, 50, 10, 40, -30 / 180 * np.pi]])

    pred = torch.FloatTensor([[50, 50, 10, 70, -30 / 180 * np.pi],
                              [50, 50, 100, 700, -30 / 180 * np.pi]])
    target = torch.FloatTensor([[50, 50, 10, 70, -30 / 180 * np.pi],
                                [52, 50, 108, 680, -30 / 180 * np.pi]])

    weight = torch.FloatTensor([1.0, 1.0])

    print(gwd_loss(pred, target, tau=1.))
