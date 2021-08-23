from ..builder import PIPELINES
from .transforms import Resize, RandomFlip, RandomCrop
import numpy as np
import mmcv
import random
from mmdet.core.visualization import imshow_det_bboxes
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
        #     # imgname = results['img_info']['filename']
        #     imshow_det_rbboxes(results['img'], results['gt_bboxes'], results['gt_labels'], show=False, \
        #                        out_file='/home/lzy/xyh/Netmodel/s2anet/imgaes/' + str(int(time.time() % 1000)) + '.png')
        #     imshow_det_bboxes(results['img'], results['hor_gt_bboxes'], results['gt_labels'], show=False, \
        #                       out_file='/home/lzy/xyh/Netmodel/s2anet/imgaesh/' + str(int(time.time() % 1000)) + '.png')
        return results

@PIPELINES.register_module()
class RMixUp:
    """
     The mixup transform steps are as follows::
        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.
    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
           Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
           Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
           Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
           iterations is greater than `max_iters`, but gt_bbox is still
           empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=1.0,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 change_scale=False):
        assert isinstance(img_scale, tuple)
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.change_scale=change_scale

    def __call__(self, results):
        """Call function to make a mixup of image.
        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with mixup transformed.
        """

        results = self._mixup_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.
        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset)-1)
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break

        return index

    def _mixup_transform(self, results):
        """MixUp transform function.
        Args:
            results (dict): Result dict.
        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        if 'scale' in results:
            self.dynamic_scale = results['scale']

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        # now we just keep the image the origin size
        if self.change_scale:
            scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                              self.dynamic_scale[1] / retrieve_img.shape[1])
            retrieve_img = mmcv.imresize(
                retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                               int(retrieve_img.shape[0] * scale_ratio)))

            # 2. paste
            out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

            # 3. scale jit
            scale_ratio *= jit_factor
            out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                              int(out_img.shape[0] * jit_factor)))
        else:
            out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img
        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                             x_offset:x_offset + target_w]

        # 6. adjust bbox
        if self.change_scale:
            retrieve_gt_bboxes = retrieve_results['gt_bboxes']
            retrieve_gt_bboxes[:, 0::2] = np.clip(
                retrieve_gt_bboxes[:, 0::2] * scale_ratio, 0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(
                retrieve_gt_bboxes[:, 1::2] * scale_ratio, 0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                    origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        if self.change_scale:
            cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2] - x_offset, 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2] - y_offset, 0, target_h)
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                    cp_retrieve_gt_bboxes.T)

        # 8. mix up
            if keep_list.sum() >= 1.0:
                ori_img = ori_img.astype(np.float32)
                mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(
                    np.float32)

                retrieve_gt_labels = retrieve_results['gt_labels'][keep_list]
                retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]
                mixup_gt_bboxes = np.concatenate(
                    (results['gt_bboxes'], retrieve_gt_bboxes), axis=0)
                mixup_gt_labels = np.concatenate(
                    (results['gt_labels'], retrieve_gt_labels), axis=0)

                results['img'] = mixup_img
                results['img_shape'] = mixup_img.shape
                results['gt_bboxes'] = mixup_gt_bboxes
                results['gt_labels'] = mixup_gt_labels
        else:
            ori_img = ori_img.astype(np.float32)
            mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(
                np.float32)
            retrieve_gt_labels = retrieve_results['gt_labels']
            retrieve_gt_bboxes = retrieve_results['gt_bboxes']
            retrieve_hor_gt_bboxes = retrieve_results['hor_gt_bboxes']

            retrive_polygons = retrieve_results['ann_info']['polygons']

            mixup_gt_bboxes = np.concatenate(
                (results['gt_bboxes'], retrieve_gt_bboxes), axis=0)
            mixup_hor_gt_bboxes = np.concatenate(
                (results['hor_gt_bboxes'], retrieve_hor_gt_bboxes), axis=0)
            mixup_gt_labels = np.concatenate(
                (results['gt_labels'], retrieve_gt_labels), axis=0)
            mixup_polygons = np.concatenate(
                (results['ann_info']['polygons'], retrive_polygons), axis=0)

            results['img'] = mixup_img
            results['img_shape'] = mixup_img.shape
            results['gt_bboxes'] = mixup_gt_bboxes
            results['hor_gt_bboxes'] = mixup_hor_gt_bboxes
            results['gt_labels'] = mixup_gt_labels
            results['ann_info']['polygons'] = mixup_polygons
        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:
        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range})'
        repr_str += f'flip_ratio={self.flip_ratio})'
        repr_str += f'pad_val={self.pad_val})'
        repr_str += f'max_iters={self.max_iters})'
        repr_str += f'min_bbox_size={self.min_bbox_size})'
        repr_str += f'min_area_ratio={self.min_area_ratio})'
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio})'
        return repr_str
