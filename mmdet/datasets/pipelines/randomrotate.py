from ..builder import PIPELINES
import numpy as np
from numpy import random
import cv2 as cv
import copy
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from functools import partial


def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)


def rotate_poly_single(h, w, new_h, new_w, rotate_matrix_T, poly):
    poly[::2] = poly[::2] - (w - 0) * 0.5
    poly[1::2] = poly[1::2] - (h - 0) * 0.5
    coords = poly.reshape(4, 2)
    new_coords = np.matmul(coords, rotate_matrix_T) + np.array([(new_w - 0) * 0.5, (new_h - 0) * 0.5])
    rotated_polys = new_coords.reshape(-1, ).clip(min=0, max=new_h).tolist()

    return rotated_polys


def rotate_poly(h, w, new_h, new_w, rotate_matrix_T, polys):
    rotate_poly_fn = partial(rotate_poly_single, h, w, new_h, new_w, rotate_matrix_T)
    rotated_polys = list(map(rotate_poly_fn, polys))
    # for poly in polys:
    #     rotated_polys = rotate_poly_single(h, w, new_h, new_w, rotate_matrix_T, poly)

    return rotated_polys


# the code is adapted from https://github.com/csuhan/ReDet
@PIPELINES.register_module()
class Randomrotate(object):

    def __init__(self,
                 CLASSES=None,
                 scale=1.0,
                 border_value=0,
                 auto_bound=True,
                 rotate_range=(-180, 180),
                 rotate_ratio=1.0,
                 rotate_values=[0, 45, 90, 135, 180, 225, 270, 315],
                 rotate_mode='range',
                 small_filter=4,
                 with_masks=False):
        self.CLASSES = CLASSES
        self.scale = scale
        self.border_value = border_value
        self.auto_bound = auto_bound
        self.rotate_range = rotate_range
        self.rotate_ratio = rotate_ratio
        self.rotate_values = rotate_values
        self.rotate_mode = rotate_mode
        self.small_filter = small_filter
        self.with_masks = with_masks

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

    def __call__(self, results):
        if np.random.rand() > self.rotate_ratio:
            results['rotate'] = False
        elif self.rotate_mode == 'range':
            results['rotate'] = True
            discrete_range = [90, 180, -90, -180]
            for label in results['gt_labels']:
                # print('label: ', label)
                cls = self.CLASSES[label]
                # print('cls: ', cls)
                if (cls == 'storage-tank') or (cls == 'roundabout') or (cls == 'airport'):
                    random.shuffle(discrete_range)
                    angle = discrete_range[0]
        elif self.rotate_mode == 'value':
            results['rotate'] = True
            random.shuffle(self.rotate_values)
            angle = self.rotate_values[0]
        if results['rotate']:
            h, w, _ = results['img'].shape
            old_h, old_w = h, w
            center = ((w - 0) * 0.5, (h - 0) * 0.5)
            matrix = cv.getRotationMatrix2D(center, -angle, self.scale)
            matrix_T = copy.deepcopy(matrix[:2, :2]).T
            # 自适应图片边框大小
            if self.auto_bound:
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                new_w = h * sin + w * cos
                new_h = h * cos + w * sin
                matrix[0, 2] += (new_w - w) * 0.5
                matrix[1, 2] += (new_h - h) * 0.5
                w = int(np.round(new_w))
                h = int(np.round(new_h))
            results['img'] = cv.warpAffine(results['img'], matrix, (w, h), borderValue=self.border_value)

            # check the correctness
            # mask_np = results['gt_masks'].to_ndarray().transpose(1, 2, 0)
            # mask_rotate_np = cv.warpAffine(mask_np, matrix, (w, h), borderValue=self.border_value)

            fourpoint = results['ann_info']['polygons']
            rotfourpoint = np.array(rotate_poly(old_h, old_w, h, w, matrix_T, fourpoint)).astype(np.int32)
            # results['old_hor_gt_bboxes'] = results['hor_gt_bboxes']
            results['hor_gt_bboxes'] = poly2bbox(rotfourpoint).astype(np.float32)
            # if (results['hor_gt_bboxes'] < 0).any():
            #     import pdb
            #     pdb.set_trace()
            gt_bboxes = []
            val_inds = []
            for ind, polygon in enumerate(rotfourpoint):
                bboxps = np.array(polygon).reshape((4, 2)).astype(np.float32)
                rbbox = cv.minAreaRect(bboxps)
                x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    continue
                while not 0 > a >= -90:
                    if a >= 0:
                        a -= 90
                        w, h = h, w
                    else:
                        a += 90
                        w, h = h, w
                a = a / 180 * np.pi
                assert 0 > a >= -np.pi / 2
                gt_bboxes.append([x, y, w, h, a])
                val_inds.append(ind)
            # gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            results['hor_gt_bboxes'] = results['hor_gt_bboxes'][val_inds]
            results['gt_labels'] = results['gt_labels'][val_inds]
            if self.with_masks:
                rotfourpoint = rotfourpoint[val_inds]
                results['gt_masks'] = BitmapMasks([self._poly2mask(rotfourpoint, h, w)], h, w)
            # check the rightness
            # plt.imshow(results['gt_masks'].to_ndarray().squeeze(), interpolation='nearest')
            # plt.savefig('mask1.png')
            # plt.imshow(results['img'], interpolation='nearest')
            # plt.savefig('img1.png')
            # plt.imshow(results['gt_masks'].to_ndarray().squeeze(), interpolation='nearest')
            # plt.savefig('mask2.png')
            # import pdb
            # pdb.set_trace()
            assert results['gt_bboxes'].shape[0] == results['hor_gt_bboxes'].shape[0] == results['gt_labels'].shape[0]
            # assert results['gt_labels'].shape[0] > 0
            if results['gt_labels'].shape[0] == 0:
                results['gt_bboxes'] = results['gt_bboxes_ignore']
                results['hor_gt_bboxes'] = results['hor_gt_bboxes_ignore']
        return results
