from ..builder import PIPELINES
from .transforms import Resize, RandomFlip, RandomCrop
import numpy as np
import mmcv

@PIPELINES.register_module()
class RResize(Resize):
    """
        Resize images & rotated bbox & horizon box
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
                                      keep_ratio=True,
                                      bbox_clip_border=bbox_clip_border)

    def _resize_bboxes(self, results):
        # import pdb
        # pdb.set_trace()
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


@PIPELINES.register_module()
class RRandomFlip(RandomFlip):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

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

        if 'flip' not in results:
            # 翻转方向以列表形式
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            # 处理翻转的概率
            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            # 随机选择该图片时水平竖值还是不反转
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results