from ..builder import PIPELINES
from .transforms import Resize, RandomFlip, RandomCrop
import numpy as np
import mmcv
from  mmdet.core.visualization import imshow_det_bboxes
from mmdet.core.visualization import imshow_det_rbboxes
import time

@PIPELINES.register_module()
class RResize(Resize):
    """
        Resize images & rotated bbox & horizon & masks
        Inherit Resize pipeline class to handle rotated bboxes
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 bbox_clip_border=True):
        super(RResize, self).__init__(img_scale=img_scale,
                                      multiscale_mode=multiscale_mode,
                                      ratio_range=ratio_range,
                                      keep_ratio=False,
                                      bbox_clip_border=bbox_clip_border)

    def _resize_bboxes(self, results):
        for key in results.get('bbox_fields', []):
            if key == 'gt_bboxes' or key == 'gt_bboxes_ignore':
                bboxes = results[key]
                orig_shape = bboxes.shape
                bboxes = bboxes.reshape((-1, 5))
                w_scale, h_scale, _, _ = results['scale_factor']
                bboxes[:, 0] *= w_scale
                bboxes[:, 1] *= h_scale
                bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
                results[key] = bboxes.reshape(orig_shape)
            elif key == 'hor_gt_bboxes' or key == 'hor_gt_bboxes_ignore':
                bboxes = results[key] * results['scale_factor']
                if self.bbox_clip_border:
                    img_shape = results['img_shape']
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                results[key] = bboxes
        # import pdb
        # pdb.set_trace()
        # print(results['hor_gt_bboxes'])

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if results['img_shape'][:2] != (1024, 1024):
                import pdb
                pdb.set_trace()
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])


@PIPELINES.register_module()
class RRandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """
    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            for ratio_item in flip_ratio:
                assert 0 <= ratio_item <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction), \
                'ratio len must equal diretion len'

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally or vertically.

        Args:
            bboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)
        """
        if bboxes.shape[-1] % 5 == 0:
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            flipped = bboxes.copy()
            if direction == 'horizontal':
                flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
            elif direction == 'vertical':
                flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
            else:
                raise ValueError(
                    'Invalid flipping direction "{}"'.format(direction))
            rotated_flag = (bboxes[:, 4] != -np.pi / 2)
            flipped[rotated_flag, 4] = -np.pi / 2 - bboxes[rotated_flag, 4]
            flipped[rotated_flag, 2] = bboxes[rotated_flag, 3],
            flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
            return flipped.reshape(orig_shape)
        elif bboxes.shape[-1] % 4 == 0:
            flipped = bboxes.copy()
            if direction == 'horizontal':
                w = img_shape[1]
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif direction == 'vertical':
                h = img_shape[0]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif direction == 'diagonal':
                w = img_shape[1]
                h = img_shape[0]
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            else:
                raise ValueError(f"Invalid flipping direction '{direction}'")
            return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        aug_dict = {'horizontal': False, 'vertical': False}
        if 'flip' not in results:
            # 翻转方向以列表形式
            if isinstance(self.direction, str):
                # None means non-flip
                # direction_list = self.direction + [None]
                self.direction = [self.direction]
            if isinstance(self.flip_ratio, float):
                self.flip_ratio = [self.flip_ratio]
            # update by xuyihao
            assert len(self.direction) == len(self.flip_ratio)
            index = 0
            for direction_item in self.direction:
                aug_dict[direction_item] = True if np.random.rand() < self.flip_ratio[index] else False
                index += 1


            #  确定随机翻转的方向
            cur_dir = []

            for key in aug_dict.keys():
                if aug_dict[key]:
                    cur_dir.append(key)

            results['flip'] = (len(cur_dir) != 0)

        # ['img_info', 'ann_info', 'img_prefix', 'seg_prefix', 'proposal_file', 'bbox_fields', 'mask_fields',
        # 'seg_fields', 'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape', 'img_fields', 'gt_bboxes',
        # 'gt_bboxes_ignore', 'hor_gt_bboxes', 'hor_gt_bboxes_ignore', 'gt_labels', 'gt_masks', 'scale', 'scale_idx',
        # 'pad_shape', 'scale_factor', 'keep_ratio', 'flip']
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            # imshow_det_rbboxes(results['img'], results['gt_bboxes'], results['gt_labels'], show=False, out_file='ori.png')
            # imshow_det_bboxes(results['img'], results['hor_gt_bboxes'], results['gt_labels'], show=False, out_file='ori.png')
            for key in results.get('img_fields', ['img']):
                for direction in cur_dir:
                    results[key] = mmcv.imflip(
                        results[key], direction=direction)
            # flip bboxes
            for key in results.get('bbox_fields', []):
                for direction in cur_dir:
                    results[key] = self.bbox_flip(results[key],
                                                  results['img_shape'],
                                                  direction)
            # flip masks
            for key in results.get('mask_fields', []):
                for direction in cur_dir:
                    results[key] = results[key].flip(direction)

            # flip segs
            for key in results.get('seg_fields', []):
                for direction in cur_dir:
                    results[key] = mmcv.imflip(
                        results[key], direction=direction)
        # imshow_det_bboxes(results['img'], results['hor_gt_bboxes'], results['gt_labels'], show=False,
        #                   out_file='./images'+results['filename'])
        # if results['rotate']:
        #     imshow_det_rbboxes(results['img'], results['gt_bboxes'], results['gt_labels'], show=False, out_file='./images/' + str(time.time())+'.png')
        #     imshow_det_bboxes(results['img'], results['hor_gt_bboxes'], results['gt_labels'], show=False, out_file='./images/' + str(time.time())+results['img_info']['filename'])
        return results