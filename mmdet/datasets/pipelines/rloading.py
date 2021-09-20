from ..builder import PIPELINES
from .loading import LoadAnnotations
from mmdet.core import BitmapMasks, PolygonMasks
import pycocotools.mask as maskUtils
import numpy as np


@PIPELINES.register_module()
class RLoadAnnotations(LoadAnnotations):
    """Load annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_rbbox(bool).
            Default: False.(stop)
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 hsp=False
                 ):
        super(RLoadAnnotations, self).__init__(with_bbox=with_bbox,
                                               with_label=with_label,
                                               with_mask=with_mask,
                                               with_seg=with_seg,
                                               poly2mask=poly2mask)
        self.hsp = hsp

    def _load_bboxes(self, results):

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        hor_gt_bboxes = ann_info.get('hbboxes', None)
        if hor_gt_bboxes is not None:
            results['hor_gt_bboxes'] = ann_info['hbboxes'].copy()
            results['bbox_fields'].append('hor_gt_bboxes')
        hor_gt_bboxes_ignore = ann_info.get('hbboxes_ignore', None)
        if hor_gt_bboxes_ignore is not None:
            results['hor_gt_bboxes_ignore'] = ann_info['hbboxes_ignore'].copy()
            results['bbox_fields'].append('hor_gt_bboxes_ignore')

        return results

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """
        # h, w = results['img_info']['height'], results['img_info']['width']
        h = results['img_shape'][0]
        w = results['img_shape'][1]
        gt_masks = results['ann_info']['polygons']

        if self.poly2mask:
            if self.hsp:
                gt_masks = BitmapMasks([self._poly2mask(gt_masks, h, w)], h, w)
            else:
                gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann, np.ndarray):
            if mask_ann.ndim == 1:
                mask_ann = mask_ann[None, :]
            # polys = list(map(list, list(mask_ann)))
            # array[[],[],..] -- list(mask_ann) --> list[arr, arr, ...] -- list(map(list, list(mask_ann))) -->
            # list[[],..[]]
            polys = mask_ann.tolist()
            rles = maskUtils.frPyObjects(polys, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
