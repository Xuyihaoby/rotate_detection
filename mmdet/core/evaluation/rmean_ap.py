from multiprocessing import Pool

import numpy as np
import torch

from mmcv.ops import box_iou_rotated
from mmdet.ops import polygon_iou
from .bbox_overlaps import bbox_overlaps
from .mean_ap import average_precision, print_map_summary

# from mmdet.ops.rotate.rbbox_overlaps import rbbx_overlaps

def rdets2points(rbboxes):
    """Convert detection results to a list of numpy arrays.

    Args:
        rbboxes (np.ndarray): shape (n, 6), xywhap encoded

    Returns:
        rbboxes (np.ndarray): shape (n, 9), x1y1x2y2x3y3x4y4p
    """

    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    prob = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    # cosa = np.cos(a * np.pi/180.0)
    # sina = np.sin(a * np.pi/180.0)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    # if p1x.size == 0:
    #     return np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob], axis=-1)
    # else:
    #     yet = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob], axis=-1)
    #     indice = np.unique((yet < 0).nonzero()[0])
    #     compelete = np.delete(yet, indice, axis=0)
    # return compelete
    return np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob], axis=-1)



def rdets2angle(rbboxes):
    """Convert detection results to a list of numpy arrays.

    Args:


    Returns:

    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4] * 180.0 / np.pi
    prob = rbboxes[:, 5]
    return np.stack([x, y, w, h, a, prob], axis=-1)


def polygon_overlaps(polygons1, polygons2):
    p1 = torch.tensor(polygons1[:, :8], dtype=torch.float32)  # in case the last element of a row is the probability
    p2 = torch.tensor(polygons2[:, :8], dtype=torch.float32)  # in case the last element of a row is the probability
    # return box_iou_rotated(p1, p2).numpy()
    return polygon_iou(p1, p2).numpy()
    # print(type(p1))
    # p1_n = p1.numpy()
    # p2_n = p2.numpy()
    # return rbbx_overlaps(p1_n, p2_n).numpy()


def rtpfp_default(det_bboxes,
                  gt_bboxes,
                  gt_bboxes_ignore=None,
                  iou_thr=0.5,
                  area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 9).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 8).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 8). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """

    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))  # (n+k, 8) 通常ignore都是零

    num_dets = det_bboxes.shape[0]  # m
    num_gts = gt_bboxes.shape[0]  # n
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)  # 1
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)  # (1, m)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)  # (1, m)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:  # n==0
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
            det_areas = det_bboxes[:, 2] * det_bboxes[:, 3]
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
                # 个人认为这一段写的很不对劲
        return tp, fp
    ious = polygon_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
            gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def htpfp_default(det_bboxes,
                  gt_bboxes,
                  gt_bboxes_ignore=None,
                  iou_thr=0.5,
                  area_ranges=None):
    """
    """

    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))  # (n+k, 8) 通常ignore都是零

    num_dets = det_bboxes.shape[0]  # m
    num_gts = gt_bboxes.shape[0]  # n
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)  # 1
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)  # (1, m)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)  # (1, m)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:  # n==0
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
            det_areas = det_bboxes[:, 2] * det_bboxes[:, 3]
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
                # 个人认为这一段写的很不对劲
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # print("how are you")
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
            gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def rget_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [rdets2points(img_res[class_id]) for img_res in det_results]
    # （list）[[每一图片的单个目标的检测以及一个score], ...图片数]
    # cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    # annotations sum is num images
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['polygons'][gt_inds, :])
        # cls_gts.append(ann['bboxes'][gt_inds, :])
        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['polygons_ignore'][ignore_inds, :])
            # cls_gts_ignore.append(ann['bboxes'][ignore_inds, :])
        else:
            cls_gts_ignore.append(torch.empty((0, 8), dtype=torch.float32))
            # cls_gts_ignore.append(torch.empty((0, 5), dtype=torch.float32))
    return cls_dets, cls_gts, cls_gts_ignore


def reval_map(det_results,
              annotations,
              scale_ranges=None,
              iou_thr=0.5,
              dataset=None,
              logger=None,
              nproc=4):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `polygons`: numpy array of shape (n, 8)
            - `labels`: numpy array of shape (n, )
            - `polygons_ignore` (optional): numpy array of shape (k, 8)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    # (list)[(list)[(list)[该list内一共有num_class个array，每个array对应的该类里的[bbox(4
    # 个), score]], .., (test_batchsize)],...(numsamples)]
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    # pool = Pool(0)
    eval_results = []
    for i in range(num_classes):

        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = rget_cls_results(
            det_results, annotations, i)
        tpfp = []
        # tpfp = pool.starmap(
        #     rtpfp_default,
        #     zip(cls_dets, cls_gts, cls_gts_ignore,
        #         [iou_thr for _ in range(num_imgs)],
        #         [area_ranges for _ in range(num_imgs)]))
        for cls_det, cls_gt, cls_gt_ignore, iou_thr_, area_ranges_ in tqdm(zip(cls_dets, cls_gts, cls_gts_ignore, [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)])):
            tpfp.append(rtpfp_default(cls_det, cls_gt, cls_gt_ignore, iou_thr_, area_ranges_))
        tp, fp = tuple(zip(*tpfp))

        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        # 主要用于去掉没有检测目标的array，并把所有图片的检测目标合并
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]  # [[]] ---> []
            num_gts = num_gts.item()
        mode = '11points'  # at least for DOTA dataset we use 11 points
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
        #

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def hget_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    # （list）[[每一图片的单个目标的检测八个角以及一个score], ...图片数]
    # cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    # annotations sum is num images
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        # cls_gts.append(ann['polygons'][gt_inds, :])
        cls_gts.append(ann['hbboxes'][gt_inds, :])
        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            # cls_gts_ignore.append(ann['polygons_ignore'][ignore_inds, :])
            cls_gts_ignore.append(ann['hbboxes'][ignore_inds, :])
        else:
            # cls_gts_ignore.append(torch.empty((0, 8), dtype=torch.float32))
            cls_gts_ignore.append(torch.empty((0, 4), dtype=torch.float32))
    return cls_dets, cls_gts, cls_gts_ignore


def heval_map(det_results,
              annotations,
              scale_ranges=None,
              iou_thr=0.5,
              dataset=None,
              logger=None,
              nproc=4):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `polygons`: numpy array of shape (n, 8)
            - `labels`: numpy array of shape (n, )
            - `polygons_ignore` (optional): numpy array of shape (k, 8)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    # (list)[(list)[(list)[该list内一共有num_class个array，每个array对应的该类里的[bbox(4
    # 个), score]], .., (test_batchsize)],...(numsamples)]
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(16)
    # pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = hget_cls_results(
            det_results, annotations, i)
        # （list）[[每一图片的单个目标的检测四个点的坐标以及一个score], ...图片数]

        tpfp = pool.starmap(
            htpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))

        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = '11points'  # at least for DOTA dataset we use 11 points
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
        #

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results
