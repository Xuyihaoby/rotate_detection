import numpy as np
import xml.etree.ElementTree as ET

import os
import os.path as osp
import mmcv
import random
import pickle as pkl
# import imgaug
import math
import time
from skimage import transform as skitf
from mmcv.parallel import DataContainer as DC
from pycocotools.coco import maskUtils

from .custom import CustomDataset
from .utils import to_tensor, random_scale
from ..models.utils import get_base_name, transQuadrangle2Rotate, transXyxyxyxy2Xyxy, transRotate2Quadrangle


class DotaDataset(CustomDataset):
    ## v1
    CLASSES_v1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                  'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    ## v1.5
    CLASSES_v15 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                   'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                   'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter', 'container-crane')

    ## /data3/DOTA/VOC_format/images|||labelTxt
    ## folder in xml: /data3/DOTA/VOC_format/val1024/images

    def __init__(self, with_single_mask=False, **kwargs):
        super(DotaDataset, self).__init__(**kwargs)

        if 'v1.5' in self.ann_file:
            print('**' * 10 + ' V1.5')
            self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES_v15)}
        else:
            print('**' * 10 + ' V1')
            self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES_v1)}

        ## with_mask: cross level mask
        ## with_single_mask: single level mask
        self.with_single_mask = with_single_mask
        if self.with_mask and self.with_single_mask:
            raise ValueError('Only support one mode.')

        print('with_mask: ', self.with_mask)
        print('with_single_mask: ', self.with_single_mask)

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform(max_num_gts=100)
        # self.rotate_angles = [90, 180, 270] ## degree
        if self.with_mask or self.with_single_mask:
            self.mask_transform = MaskTransform()

        ## roundabout(turntable) may drop when do arbitrary rotation,
        ## so we only do 90*k rotation.
        self.ign_rotate_cls = np.array([12], dtype=np.int32)

    def load_annotations(self, ann_file):
        # cache_p = './data/dota_cache/'
        cache_p = osp.join('/data2/lczheng/detection/mmdetection', 'data/dota_cache/')
        if not osp.exists(cache_p):
            os.makedirs(cache_p)
        cache_file = osp.join(cache_p, get_base_name(ann_file) + '.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as f:
                img_infos = pkl.load(f)
            print('read from cache {}'.format(cache_file))
        else:
            img_infos = []
            anno_ps = mmcv.list_from_file(ann_file)  ## annotation file path.
            for ii, anno_p in enumerate(anno_ps):
                if ii % 1000 == 0:
                    print('process {}/{}'.format(ii, len(anno_ps)))
                filename = anno_p.replace('labelTxt', 'images').replace('.xml', '.png')  ## BGR
                tree = ET.parse(anno_p)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                img_infos.append(
                    dict(id=get_base_name(anno_p), filename=filename, anno_path=anno_p,
                         width=width, height=height))
            with open(cache_file, 'wb') as f:
                pkl.dump(img_infos, f, protocol=-1)
            print('save into cache {}'.format(cache_file))
        # if not self.test_mode:
        #    print('shuffle')
        #    random.shuffle(img_infos)
        #print('All images = {}'.format(len(img_infos)))
        return img_infos

    def get_ann_info(self, idx):

        xml_path = self.img_infos[idx]['anno_path']
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            # difficult = 0
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('x0').text),
                int(bnd_box.find('y0').text),
                int(bnd_box.find('x1').text),
                int(bnd_box.find('y1').text),
                int(bnd_box.find('x2').text),
                int(bnd_box.find('y2').text),
                int(bnd_box.find('x3').text),
                int(bnd_box.find('y3').text)
            ]
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 8))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 8))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        ann = dict(
            bboxes=bboxes.astype(np.float32),  ## [G, 8]
            labels=labels.astype(np.int64),  ## [G]
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # print(img_info)
        # load image
        img = mmcv.imread(img_info['filename'])  ## cv2 read, ori is BGR, so img is RGB.
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 8 or proposals.shape[1] == 9):
                raise AssertionError(
                    'proposals should have shapes (n, 8) or (n, 9), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 9:
                scores = proposals[:, 8, None]
                proposals = proposals[:, :8]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']  ## [G, 8]
        gt_labels = ann['labels']  ## [G]
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']  ## [0, 8]

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        lr_flip = True if np.random.rand() < self.flip_ratio[0] else False
        up_down_flip = True if np.random.rand() < self.flip_ratio[1] else False
        rotate = True if np.random.rand() < self.flip_ratio[2] else False
        # 上下翻转，左右翻转，旋转
        # rotate_ang = random.randint(-180, 180) if rotate else None
        ########## Updated By LCZ, time: 2019.3.11, fix ignore rotate ##########
        is_ignore_rotate = len(np.intersect1d(self.ign_rotate_cls, gt_labels)) > 0
        # rotate_ang = random.randint(-180, 180) if rotate and is_ignore_rotate
        if rotate and is_ignore_rotate:
            rotate_ang = random.choice([90, -90, 180])
        elif rotate and not is_ignore_rotate:
            rotate_ang = random.randint(-180, 180)
        else:
            rotate_ang = None

        aug_order = []
        aug_order = aug_order + ['lr_flip'] if lr_flip else aug_order
        aug_order = aug_order + ['up_down_flip'] if up_down_flip else aug_order
        aug_order = aug_order + ['rotate'] if rotate else aug_order
        random.shuffle(aug_order)
        aug_order.append(rotate_ang)
        # print(aug_order)

        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, aug_order, keep_ratio=self.resize_keep_ratio)  ## img:[C,H,W], img_shape:(H,W,C)
        img = img.copy()
        if self.proposals is not None:
            raise NotImplementedError
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            lr_flip, up_down_flip, lr_up_down)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals

        gt_bboxes, sel_idx, rm_idx = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                                         aug_order)  ## [G, 8(x1,y1, x2,y2, x3,y3, x4,y4)]

        if sel_idx is not None:
            # rm_idx = np.array(list(set(np.arange(gt_bboxes.shape[0])) - set(sel_idx)), dtype = np.int64)
            # gt_bboxes_ignore = gt_bboxes[rm_idx] ## [?, 8]
            if rm_idx is not None:
                gt_bboxes_ignore = gt_bboxes[rm_idx]  ## [G'', 8]
            else:
                gt_bboxes_ignore = np.zeros((0, 8), dtype=np.float32)
            gt_bboxes = gt_bboxes[sel_idx]  ## [G, 8]
            gt_labels = gt_labels[sel_idx]  ## [G]

        else:
            return None

        ## Debug
        # ## c,h,w
        # _img = mmcv.imdenormalize(img.transpose(1,2,0), np.array([123.675, 116.28, 103.53], dtype = np.float32),
        # 								np.array([58.395, 57.12, 57.375], dtype = np.float32), False)
        # mmcv.imwrite(_img, './data/{}.png'.format(idx))
        # with open('./data/{}.txt'.format(idx), 'w') as f:
        # 	for iii in range(len(gt_bboxes)):
        # 		f.write(' '.join(list(map(str, list(gt_bboxes[iii].astype(np.int32))))) + ' ' + str(gt_labels[iii]) + '\n')

        # import pdb
        # pdb.set_trace()
        hor_gt_boxes = transXyxyxyxy2Xyxy(boxes=gt_bboxes, with_label_last=False)  ## [G, 4(x1,y1, x2,y2)]
        hor_gt_bboxes_ignore = transXyxyxyxy2Xyxy(boxes=gt_bboxes_ignore,
                                                  with_label_last=False)  ## [G'', 4(x1,y1, x2,y2)]
        rot_gt_bboxes = transQuadrangle2Rotate(coordinates=gt_bboxes,
                                               with_label_last=False)  ## [G, 5(x_ctr,y_ctr, w,h, theta)]

        # t0 = time.time()
        ########## Updated By LCZ, time: 2019.3.21, add pyramid rotate mask. ##########
        if self.with_mask:
            gt_masks = []  ## list(5), each is [height, width]
            gt_masks_weights = []  ## list(5)
            ## lvls: [N]
            ## lvls_fake: list, each is [X]
            lvls, lvls_fake = map_roi_levels_np(rois=hor_gt_boxes, num_levels=5)
            for ll in range(5):
                height = img_shape[0]
                width = img_shape[1]
                ## True lvls
                if (lvls == ll).any():
                    gt_bboxes_ = gt_bboxes[lvls == ll]
                    rot_gt_bboxes_ = rot_gt_bboxes[lvls == ll]
                    gt_labels_ = gt_labels[lvls == ll]
                    # m = poly2mask(polys = bboxes[lvls == ll], height = height, width = width)
                    # gt_masks.append(m)
                    ## m: 1 means pos, 0 means neg
                    ## m_weight: 1 means add loss, 0 means no add.
                    ########
                    # m, m_weight = scaledPoly2mask_fly(polys = gt_bboxes_,
                    # 									rot_boxes = rot_gt_bboxes_,
                    # 									height = height,
                    # 									width = width,
                    # 									gt_labels = gt_labels_)
                    ########
                    m = poly2mask(gt_bboxes_, height, width)  ## 1 means pos, 0 means neg, -1 means not add loss.
                    m_weight = np.ones([height, width], dtype=np.uint8)

                else:
                    m = np.zeros([height, width]).astype(np.uint8)
                    m_weight = np.ones([height, width]).astype(np.uint8)

                ## Fake lvls
                b_idx = np.array([ii if ll in lf else -1 for ii, lf in enumerate(lvls_fake)])
                if (b_idx > -1).any():
                    # label_fake = gt_labels[b_idx > -1]
                    # fly_labels = np.where((label_fake == 1)|(label_fake == 15))[0]
                    # if len(fly_labels) > 0:
                    if False:
                        box_fk_fly = rot_gt_bboxes[b_idx[b_idx > -1][fly_labels]]  ## [x,y, w,h, theta]
                        box_fk_fly[:, 2] = box_fk_fly[:, 2] * 0.5
                        box_fk_fly[:, 3] = box_fk_fly[:, 3] * 0.5
                        box_fk_fly = transRotate2Quadrangle(box_fk_fly, False)  ## [xyxyxyxy...]
                        fake_m = poly2mask(polys=box_fk_fly, height=height, width=width)
                        # m[np.logical_and(fake_m == 1, m != 1)] = -1
                        m_weight[np.logical_and(fake_m == 1, m != 1)] = 0

                        otr_label = np.where((label_fake != 1) & (label_fake != 15))[0]
                        if len(otr_label) > 0:
                            box_fk_otr = rot_gt_bboxes[b_idx[b_idx > -1][otr_label]]  ## [x,y, w,h, theta]
                            box_fk_otr = transRotate2Quadrangle(box_fk_otr, False)  ## [xyxyxyxy...]
                            fake_m = poly2mask(polys=box_fk_otr, height=height, width=width)
                            # m[np.logical_and(fake_m == 1, m != 1)] = -1
                            m_weight[np.logical_and(fake_m == 1, m != 1)] = 0
                    else:
                        b_idx = b_idx[b_idx > -1]
                        box_fk = rot_gt_bboxes[b_idx]  ## [x,y, w,h, theta]
                        box_fk = transRotate2Quadrangle(box_fk, False)  ## [xyxyxyxy...]
                        fake_m = poly2mask(polys=box_fk, height=height, width=width)
                        # m[np.logical_and(fake_m == 1, m != 1)] = -1
                        m_weight[np.logical_and(fake_m == 1, m != 1)] = 0

                gt_masks.append(m)
                gt_masks_weights.append(m_weight)

            ## [5, H, W]
            # gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, aug_order)
            # gt_masks_weights = self.mask_transform(gt_masks, pad_shape, scale_factor, aug_order)
            gt_masks = np.stack(gt_masks, axis=-1)  ## [H, W, 5]
            gt_masks_weights = np.stack(gt_masks_weights, axis=-1)  ## [H, W, 5]

            gt_masks = mmcv.impad(gt_masks, pad_shape[:2], pad_val=0)
            gt_masks_weights = mmcv.impad(gt_masks_weights, pad_shape[:2], pad_val=0)

            strides = [1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
            gt_masks = np.transpose(gt_masks, [2, 0, 1])  ## [5, H, W]
            gt_masks_weights = np.transpose(gt_masks_weights, [2, 0, 1])  ## [5, H, W]

            gt_masks = [np.expand_dims(mmcv.imrescale(mm, scale=ss), axis=0)
                        for mm, ss in zip(gt_masks, strides)]  ## each is [1, H, W]
            gt_masks_weights = [np.expand_dims(mmcv.imrescale(mm, scale=ss), axis=0)
                                for mm, ss in zip(gt_masks_weights, strides)]  ## each is [1, H, W]

        # # Debug
        # ## c,h,w
        # import cv2
        # for ii,m in enumerate(gt_masks):
        # 	_img = mmcv.imdenormalize(img.transpose(1,2,0), np.array([123.675, 116.28, 103.53], dtype = np.float32),
        # 									np.array([58.395, 57.12, 57.375], dtype = np.float32), False)
        # 	m = m.astype(np.float32)
        # 	m[gt_masks_weights[ii] == 0] = 0.5
        # 	cam = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)
        # 	cam = np.float32(cam) + np.float32(_img)
        # 	cam = 255 * cam / np.max(cam)
        # 	cam = cam.astype(np.uint8)

        # 	mmcv.imwrite(cam, './data/mask_test/{}_stage{}.png'.format(idx, ii))
        # with open('./data/{}.txt'.format(idx), 'w') as f:
        # 	for iii in range(len(gt_bboxes)):
        # 		f.write(' '.join(list(map(str, list(gt_bboxes[iii].astype(np.int32))))) + ' ' + str(gt_labels[iii]) + '\n')
        # print('*'*10+'time = {}s'.format(time.time() - t0))

        ########## Updated By LCZ, time: 2019.3.27, add single level rotate mask. ##########
        ########## Updated By LCZ, time: 2019.4.8. ##########
        if self.with_single_mask:
            ## each one is uint8
            ## [H, W], [H, W]
            ########
            # gt_masks, gt_masks_weights = scaledPoly2mask_fly(polys = gt_bboxes, ## [G, 8]
            # 												rot_boxes = rot_gt_bboxes, ## [G, 5]
            # 												height = img_shape[0],
            # 												width = img_shape[1],
            # 												gt_labels = gt_labels) ## [G]
            ########
            height = img_shape[0]
            width = img_shape[1]
            gt_masks_weights = np.ones([height, width], dtype=np.uint8)  ## [h,w]
            gt_masks = poly2mask(gt_bboxes, height, width)  ## 1 means pos, 0 means neg

            ######## TODO add ignore
            if len(gt_bboxes_ignore) > 0:
                masks_ignore = poly2mask(gt_bboxes_ignore, height, width)
                gt_masks_weights[np.logical_and(masks_ignore == 1, gt_masks == 0)] = 0

            ## [1,h,w]
            gt_masks = np.expand_dims(mmcv.impad(gt_masks, pad_shape[:2], pad_val=0),
                                      axis=0)  ## like padded image shape
            gt_masks_weights = np.expand_dims(mmcv.impad(gt_masks_weights, pad_shape[:2], pad_val=0),
                                              axis=0)  ## like padded image shape

        ## target: P3-style, stride = 8, [1,H,W]
        # gt_masks = np.expand_dims(mmcv.imrescale(gt_masks, scale = 1/8), axis = 0)
        # gt_masks_weights = np.expand_dims(mmcv.imrescale(gt_masks_weights, scale = 1/8), axis = 0)
        # print('1', img_shape)
        # print('2', gt_masks.shape)
        # print('3', gt_masks_weights.shape)

        # # Debug
        # ## c,h,w
        # import cv2
        # _img = mmcv.imdenormalize(img.transpose(1,2,0), np.array([123.675, 116.28, 103.53], dtype = np.float32),
        # 								np.array([58.395, 57.12, 57.375], dtype = np.float32), False)
        # # gt_masks_ = mmcv.imrescale(gt_masks, scale = 8)
        # # gt_masks_weights_ = mmcv.imrescale(gt_masks_weights, scale = 8)
        # gt_masks_ = gt_masks[0]
        # gt_masks_weights_ = gt_masks_weights[0]
        # gt_masks_ = gt_masks_.astype(np.float32)
        # gt_masks_[gt_masks_weights_ == 0] = 0.5
        # cam = cv2.applyColorMap(np.uint8(255 * gt_masks_), cv2.COLORMAP_JET)
        # cam = np.float32(cam) + np.float32(_img)
        # cam = 255 * cam / np.max(cam)
        # cam = cam.astype(np.uint8)

        # mmcv.imwrite(cam, './data/mask_test/{}_single.png'.format(idx))
        # with open('./data/{}.txt'.format(idx), 'w') as f:
        # 	for iii in range(len(gt_bboxes)):
        # 		f.write(' '.join(list(map(str, list(gt_bboxes[iii].astype(np.int32))))) + ' ' + str(gt_labels[iii]) + '\n')

        aug_order_meta = [int(lr_flip), int(up_down_flip), int(rotate)]
        if lr_flip:
            aug_order_meta[0] = aug_order.index('lr_flip') + 1
        if up_down_flip:
            aug_order_meta[1] = aug_order.index('up_down_flip') + 1
        if rotate:
            aug_order_meta[2] = aug_order.index('rotate') + 1
            aug_order_meta.append(aug_order[-1])

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            aug_order=aug_order_meta, )
        # lr_flip = lr_flip,
        # up_down_flip = up_down_flip,
        # lr_up_down = lr_up_down)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(rot_gt_bboxes)),  ## [G, 5(x_ctr,y_ctr, w,h, theta)]
            hor_gt_boxes=DC(to_tensor(hor_gt_boxes)))  ## [G, 4(x1,y1, x2,y2)]
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))  ## [G]
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(hor_gt_bboxes_ignore))  ## [?, 4]
        if self.with_mask:
            # data['gt_masks'] = DC(gt_masks, cpu_only=True) ## [5, H, W]
            # data['gt_masks_weights'] = DC(gt_masks_weights, cpu_only = True) ## [5, H, W]
            data['gt_masks_p2'] = DC(to_tensor(gt_masks[0]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_weights_p2'] = DC(to_tensor(gt_masks_weights[0]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_p3'] = DC(to_tensor(gt_masks[1]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_weights_p3'] = DC(to_tensor(gt_masks_weights[1]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_p4'] = DC(to_tensor(gt_masks[2]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_weights_p4'] = DC(to_tensor(gt_masks_weights[2]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_p5'] = DC(to_tensor(gt_masks[3]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_weights_p5'] = DC(to_tensor(gt_masks_weights[3]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_p6'] = DC(to_tensor(gt_masks[4]), stack=True, cpu_only=False)  ## [1, H, W]
            data['gt_masks_weights_p6'] = DC(to_tensor(gt_masks_weights[4]), stack=True, cpu_only=False)  ## [1, H, W]
        if self.with_single_mask:
            data['gt_masks'] = DC(to_tensor(gt_masks), stack=True)  ## [1, H, W]
            data['gt_masks_weights'] = DC(to_tensor(gt_masks_weights), stack=True)  ## [1, H, W]
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(img_info['filename'])  ## cv2 read, ori is BGR, so img is RGB.
        if self.proposals is not None:
            raise NotImplementedError
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 8 or proposal.shape[1] == 9):
                raise AssertionError(
                    'proposals should have shapes (n, 8) or (n, 9), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, lr_flip, up_down_flip, rotate, rotate_ang, proposal=None):
            ## rotate_ang: positive values mean clockwise

            aug_order = []
            aug_order = aug_order + ['lr_flip'] if lr_flip else aug_order
            aug_order = aug_order + ['up_down_flip'] if up_down_flip else aug_order
            aug_order = aug_order + ['rotate'] if rotate else aug_order
            if rotate_ang is not None:
                aug_order.append(rotate_ang)
            else:
                aug_order.append(None)

            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, aug_order, keep_ratio=self.resize_keep_ratio, auto_bound=True, test_mode=True)
            _img = to_tensor(_img)

            aug_order_meta = [int(lr_flip), int(up_down_flip), int(rotate)]
            if rotate_ang is not None:
                aug_order_meta.append(rotate_ang)
            else:
                aug_order_meta.append(0)

            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                aug_order=aug_order_meta
            )

            if proposal is not None:
                if proposal.shape[1] == 9:
                    score = proposal[:, 8, None]
                    proposal = proposal[:, :8]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, lr_flip, up_down_flip, lr_up_down)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, False, False, None, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio[0] > 0:  ## if lr_flip
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, False, False, None, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)

            if self.flip_ratio[1] > 0:  ## if up_down_flip
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, True, False, None, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)

            if self.flip_ratio[2] > 0:  ## rotate by 90*k, clockwise
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, False, True, 90, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)

                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, False, True, 180, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)

                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, False, True, 270, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)

        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data


def bbox_lr_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::2] = w - bboxes[..., 0::2] - 1  ## ???
    # flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def bbox_up_down_flip(bboxes, img_shape):
    """Flip bboxes vertically.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    h = img_shape[0]
    flipped = bboxes.copy()
    flipped[..., 1::2] = h - bboxes[..., 1::2] - 1  ## ???
    # flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def img_lr_flip(img):
    return mmcv.imflip(img, direction='horizontal')


def img_up_down_flip(img):
    return mmcv.imflip(img, direction='vertical')


def boxesRotate(boxes, angle, img_shape):
    '''
    Rotate boxes by angle degree. reference: https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/geometric.py#L730

    Arguments:
        boxes: [N, 8]
        angle: int
        img_shape：(h, w), rotated image.shape, not the original image shape!!

    Returns:
        boxes: [N, 8]
    '''
    height, width = img_shape[0], img_shape[1]
    boxes_ = boxes.reshape([-1, 2])
    shift_x = width / 2.0 - 0.5
    shift_y = height / 2.0 - 0.5
    matrix_transforms = skitf.AffineTransform(
        rotation=math.radians(angle)
    )
    matrix_to_topleft = skitf.SimilarityTransform(translation=[-shift_x, -shift_y])

    matrix_to_center = skitf.SimilarityTransform(translation=[shift_x, shift_y])
    matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)

    boxes_aug = skitf.matrix_transform(boxes_, matrix.params)
    boxes_aug = boxes_aug.reshape(-1, 8)
    # ctr_x = np.mean(boxes_aug[:, 0::2], axis = -1)
    # ctr_y = np.mean(boxes_aug[:, 1::2], axis = -1)
    # valid1 = np.where((ctr_x > 0) & (ctr_x < w))[0]
    # valid2 = np.where((ctr_y > 0) & (ctr_y < h))[0]
    # valid = np.intersect1d(valid1, valid2)
    return boxes_aug


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, aug_order=[None]):
        gt_bboxes = bboxes * scale_factor

        if len(aug_order) == 1 and aug_order[0] == None:
            gt_bboxes = gt_bboxes
            isAug = False
        else:
            for do in aug_order[:-1]:
                if do == 'rotate':
                    gt_bboxes = boxesRotate(boxes=gt_bboxes, angle=aug_order[-1], img_shape=img_shape[:2])
                else:
                    gt_bboxes = globals()['bbox_' + do](bboxes=gt_bboxes, img_shape=img_shape[:2])
            isAug = True

        #### Check center x,y
        if isAug:
            ctr_x = np.mean(gt_bboxes[:, 0::2], axis=-1)
            ctr_y = np.mean(gt_bboxes[:, 1::2], axis=-1)
            valid1 = np.where((ctr_x > 0) & (ctr_x < img_shape[1]))[0]
            valid2 = np.where((ctr_y > 0) & (ctr_y < img_shape[0]))[0]
            valid = np.intersect1d(valid1, valid2)
            # gt_bboxes = gt_bboxes[valid]
            if len(valid) == 0:
                return None, None, None
        else:
            valid = np.arange(len(gt_bboxes))

        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes.astype(np.float32), valid, None
        else:
            # num_gts = gt_bboxes.shape[0]
            num_gts = len(valid)
            if num_gts <= self.max_num_gts:
                return gt_bboxes.astype(np.float32), valid, None
            # padded_bboxes = np.zeros((self.max_num_gts, 8), dtype=np.float32)
            sel_idx = np.random.choice(valid, self.max_num_gts, replace=False)
            rm_idx = np.array(list(set(valid) - set(sel_idx)), dtype=np.int32)
            # padded_bboxes[:self.max_num_gts, :] = gt_bboxes[sel_idx]
            return gt_bboxes.astype(np.float32), sel_idx, rm_idx


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, aug_order=[None], keep_ratio=True, auto_bound=False, test_mode=False):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)

        if len(aug_order) == 1 and aug_order[0] == None:
            img = img
        else:
            for do in aug_order[:-1]:
                if do == 'rotate':
                    img = mmcv.imrotate(img=img, angle=aug_order[-1], auto_bound=auto_bound)
                else:
                    img = globals()['img_' + do](img=img)
                    # TODO: 目前不是很懂这个函数是干什么用的
        if not test_mode:
            assert img.shape == img_shape
        img_shape = img.shape

        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)  ## [h,w,3] --> [3,h,w]
        return img, img_shape, pad_shape, scale_factor


########## Updated By LCZ, time: 2019.3.22, add mask transform. ##########
class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, aug_order=[None]):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]

        if len(aug_order) == 1 and aug_order[0] == None:
            masks = masks
        else:
            for do in aug_order[:-1]:
                if do == 'rotate':
                    masks = [mmcv.imrotate(img=mask, angle=aug_order[-1]) for mask in masks]
                else:
                    masks = [globals()['img_' + do](img=mask) for mask in masks]

        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)  ## [N, H, W]
        return padded_masks


########## Updated By LCZ, time: 2019.3.21, add rotate mask. ##########
def map_roi_levels_np(rois, num_levels):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale: level 0
    - finest_scale <= scale < finest_scale * 2: level 1
    - finest_scale * 2 <= scale < finest_scale * 4: level 2
    - scale >= finest_scale * 4: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 4(x1,y1, x2,y2)).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = np.sqrt(
        (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
    target_lvls = np.floor(np.log2(scale / 56 + 1e-6))
    target_lvls_true = np.clip(target_lvls, a_min=0, a_max=num_levels - 1)
    target_lvls_fake = list(target_lvls_true)

    extra_lvls = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
    target_lvls_fake = [extra_lvls[ll] for ll in target_lvls_fake]

    return target_lvls_true, target_lvls_fake


def poly2mask(polys, height, width):
    ## polys: array, [N, 8]

    assert isinstance(polys, np.ndarray) or isinstance(polys, list), 'Unknow input type.'
    if isinstance(polys, np.ndarray):
        polys = list(map(list, list(polys)))  ## convert array to list.
    rles = maskUtils.frPyObjects(polys, height, width)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)  ## [height, width], uint8

    return m


def scaledPoly2mask_fly(polys, rot_boxes, height, width, gt_labels):
    ## polys: array, [N, 8]
    ## rot_boxes: array, [N, 5]

    assert isinstance(polys, np.ndarray), 'Unknow input type.'
    assert len(polys) == len(gt_labels) == len(rot_boxes), 'boxes and labels not match.'

    ## plane or helicopter
    fly_labels = np.where((gt_labels == 1) | (gt_labels == 15))[0]
    weights = np.ones([height, width], dtype=np.uint8)
    if len(fly_labels) > 0:
        # fly_polys = transQuadrangle2Rotate(polys[fly_labels], False) ## [x,y,h,w,theta]
        fly_polys = rot_boxes[fly_labels]
        ## only set center to mask, when cls is plane or helicopter
        fly_polys[:, 2] = fly_polys[:, 2] * 0.5
        fly_polys[:, 3] = fly_polys[:, 3] * 0.5
        fly_polys = transRotate2Quadrangle(fly_polys, False)  ## [x,y,x,y,x,y...]
        # fly_polys = list(map(list, list(fly_polys))) ## convert array to list.
        # rles = maskUtils.frPyObjects(fly_polys, height, width)
        # rle = maskUtils.merge(rles)
        # m_fly_half = maskUtils.decode(rle) ## [height, width], uint8
        m_fly_half = poly2mask(fly_polys, height, width)

        ## other cls
        otr_label = np.where((gt_labels != 1) & (gt_labels != 15))[0]
        if len(otr_label) > 0:
            # otr_polys = list(map(list, list(polys[otr_label]))) ## convert array to list.
            # rles = maskUtils.frPyObjects(otr_polys, height, width)
            # rle = maskUtils.merge(rles)
            # m_otr = maskUtils.decode(rle) ## [height, width], uint8
            m_otr = poly2mask(polys[otr_label], height, width)
            m = m_fly_half + m_otr
            m[m > 1] = 1
        else:
            m = m_fly_half

        ## set the fly ring to -1.
        m_fly = poly2mask(polys[fly_labels], height, width)
        weight_ignore = m_fly - m_fly_half
        weights[weight_ignore == 1] = 0  ## 1 means add loss, 0 means no add,
    # m = m.astype(np.int32)
    # m[weight_ignore == 1] = -1 ## 1 means pos, 0 means neg, -1 means not add loss.
    else:
        # polys = list(map(list, list(polys))) ## convert array to list.
        # rles = maskUtils.frPyObjects(polys, height, width)
        # rle = maskUtils.merge(rles)
        # m = maskUtils.decode(rle) ## [height, width], uint8
        m = poly2mask(polys, height, width)  ## 1 means pos, 0 means neg, -1 means not add loss.
    # m = m.astype(np.int32)

    return m, weights  ## uint8
