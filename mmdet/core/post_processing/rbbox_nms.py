import torch
# from mmcv.ops.nms import batched_nms
from mmcv.ops import nms_rotated

from mmdet.ops import batched_rnms, ml_nms_rotated, obb_batched_nms
from mmdet.core.bbox.iou_calculators import bbox_overlaps

dota_v1_cats = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')  # 15


def multiclass_nms_r(multi_bboxes,
                     multi_scores,
                     score_thr,
                     nms_cfg,
                     max_num=-1,
                     score_factors=None,
                     return_inds=False,
                     class_agnostic=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
        # [1000, 15, 5]
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
        #
        # [1000, 5] ---> [1000, numclasses, 5]

    scores = multi_scores[:, :-1]  # [1000, 15]
    if score_factors is not None:
        scores = scores * score_factors[:, None]

    labels = torch.arange(num_classes, dtype=torch.long)

    labels = labels.view(1, -1).expand_as(scores)
    # [15] ---> [1, 15] ---> [1000, 15]
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    # 都只剩下两维/one dimension

    # remove low scoring boxes
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_inds:
            return bboxes, labels, inds
        else:
            return bboxes, labels

    nms_version = nms_cfg.get('version', 'v1')

    # updated by xyh
    if isinstance(nms_cfg['iou_threshold'], float):
        if nms_version == 'v1':
            dets, keep = batched_rnms(bboxes, scores, labels, nms_cfg, class_agnostic=class_agnostic)
        elif nms_version == 'v3':
            dets, keep = obb_batched_nms(bboxes, scores, labels, nms_cfg.iou_threshold)
        elif nms_version == 'v2':
            labels = labels.to(bboxes)
            keep = ml_nms_rotated(bboxes, scores, labels, nms_cfg.iou_threshold)
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            if keep.size(0) > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                scores = scores[inds]
                labels = labels[inds]
            return torch.cat([bboxes, scores[:, None]], 1), labels
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

    elif isinstance(nms_cfg['iou_threshold'], dict):
        # import pdb
        # pdb.set_trace()
        det = bboxes.new_ones(0, 6)
        keeps = labels.new_ones(0).to(bboxes.device)
        for i in range(len(dota_v1_cats)):
            nms_cfg_ = nms_cfg.copy()
            nms_cfg_['iou_threshold'] = nms_cfg['iou_threshold'][dota_v1_cats[i]]
            assert isinstance(nms_cfg_['iou_threshold'], float)
            if isinstance(max_num, int):
                max_num_ = max_num
            elif isinstance(max_num, dict):
                max_num_ = max_num[dota_v1_cats[i]]
            index = torch.nonzero(labels == i).squeeze(1).to(inds)
            labels_ = labels[labels == i]
            bboxes_= bboxes[labels == i]
            scores_ = scores[labels == i]
            if bboxes_.size(0) == 0:
                continue
            dets_, keep_ = batched_rnms(bboxes_, scores_, labels_, nms_cfg_, class_agnostic=class_agnostic)
            _, sub_indice = dets_[:, 5].sort(descending=True)
            dets_ = dets_[sub_indice]
            keep_ = keep_[sub_indice]
            # 每一张图片的每一类先进行置信度的排序

            if max_num_ > dets_.size(0):
                max_num_ = dets_.size(0)
            dets_ = dets_[:max_num_]
            keep_ = keep_[:max_num_]
            det = torch.cat([det, dets_], dim=0)
            keeps = torch.cat([keeps, index[keep_]], dim=-1)
            # import pdb
            # pdb.set_trace()
            # max_num_ = torch.where(max_num_>det.size(0), det.size(0), max_num_)


        _ , indice = det[:, 5].sort(descending=True)
        dets = det[indice]
        keep = keeps[indice]
    # TODO: add size check before feed into batched_nms

    if return_inds:
        return dets, labels[keep], keep
    else:
        return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
