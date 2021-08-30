import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaRXYWHThetaBBoxCoder(BaseBBoxCoder):
    """Delta XYWHA BBox coder

    format (x ,y w, h, a) and (x ,y w, h, a) encode
    this coder is used for rotated objects detection (for example on task1 of DOTA dataset).
    this coder encodes bbox (xc, yc, w, h, a) into delta (dx, dy, dw, dh, da) and
    decodes delta (dx, dy, dw, dh, da) back to original bbox (xc, yc, w, h, a).

    Args:
        target_means (Sequence[float]): denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()

    px, py, pw, ph, pa = (proposals[:, i] for i in range(5))
    gx, gy, gw, gh, ga = (gt[:, i] for i in range(5))

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = ga - pa

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    # Compute rotated angle of each roi
    pa = rois[:, 4].unsqueeze(1).expand_as(da)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    # gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    # gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gx = px + pw * dx
    gy = py + ph * dy
    # Compute angle
    ga = pa + da
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)
    rbboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return rbboxes
