from ..builder import PIPELINES
from .loading import LoadAnnotations
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
                 poly2mask=True
                 ):
        super(RLoadAnnotations, self).__init__(with_bbox=with_bbox,
                                               with_label=with_label,
                                               with_mask=with_mask,
                                               with_seg=with_seg,
                                               poly2mask=poly2mask)

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
