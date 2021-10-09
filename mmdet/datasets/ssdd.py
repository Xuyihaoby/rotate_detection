from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls, reval_map
from .builder import DATASETS
from .xml_style import XMLDataset

import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from itertools import chain


@DATASETS.register_module()
class SSDD(XMLDataset):

    CLASSES = ('ship', )

    def __init__(self, **kwargs):
        super(SSDD, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            self.year = 'SSDD'
    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        hbboxes = []
        polygons = []
        labels = []
        hbboxes_ignore = []
        bboxes_ignore = []
        polygons_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)

            # find hbboxes
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            hbbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]

            # find polygon
            _polygon = obj.find('rotated_bndbox')
            polygon = [
                int(float(_polygon.find('x1').text)),
                int(float(_polygon.find('y1').text)),
                int(float(_polygon.find('x2').text)),
                int(float(_polygon.find('y2').text)),
                int(float(_polygon.find('x3').text)),
                int(float(_polygon.find('y3').text)),
                int(float(_polygon.find('x4').text)),
                int(float(_polygon.find('y4').text))
            ]

            # find rotate
            bboxps = np.array(polygon).reshape(
                (4, 2)).astype(np.float32)
            rbbox = cv2.minAreaRect(bboxps)
            x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
            if w == 0 or h == 0:
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
            bbox = [x, y, w, h, a]

            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = hbbox[2] - hbbox[0]
                h = hbbox[3] - hbbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                assert difficult == 0
                hbboxes_ignore.append(hbbox)
                labels_ignore.append(label)
            else:
                hbboxes.append(hbbox)
                polygons.append(polygon)
                labels.append(label)
                bboxes.append(bbox)
        if not hbboxes or not polygons or not bboxes:
            hbboxes = np.zeros((0, 4))
            polygons = np.zeros((0, 8))
            labels = np.zeros((0, ))
            bboxes = np.zeros((0, 5))
        else:
            hbboxes = np.array(hbboxes, ndmin=2) - 1
            polygons = np.array(polygons, ndmin=2) - 1
            labels = np.array(labels)
            bboxes = np.array(bboxes, ndmin=2)
        if not hbboxes_ignore or not polygons_ignore or not bboxes_ignore:
            hbboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
            polygons_ignore = np.zeros((0, 8))
            bboxes_ignore = np.zeros((0, 5))
        else:

            hbboxes_ignore = np.array(hbboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
            polygons_ignore = np.array(polygons_ignore, ndmin=2) - 1
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)

        # import pdb
        # pdb.set_trace()
        # self.visual(polygons, img_id, dst_path='/home/lzy/xyh/Netmodel/s2anet/GTimages/')


        ann = dict(
            hbboxes=hbboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            polygons=polygons.astype(np.float32),
            bboxes = bboxes.astype(np.float32),
            hbboxes_ignore=hbboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
            polygons_ignore=polygons_ignore.astype(np.float32),
            bboxes_ignore = bboxes_ignore.astype(np.float32))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                if results[0][0].shape[1] == 5:
                    mean_ap, _ = eval_map(
                        results,
                        annotations,
                        scale_ranges=None,
                        iou_thr=iou_thr,
                        dataset=ds_name,
                        logger=logger)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                elif results[0][0].shape[1] == 6:
                    mean_ap, _ = reval_map(
                        results,
                        annotations,
                        scale_ranges=scale_ranges,
                        iou_thr=iou_thr,
                        dataset=self.CLASSES,
                        logger=logger)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def visual(self, polygons, img_id, dst_path, path='/data1/public_dataset/SAR/SSDD/JPEGImages'):
        img_name = osp.join(path, img_id + '.jpg')
        img = cv2.imread(img_name)
        nplines = []

        nplines = polygons.reshape(-1, 4, 2)
        cv2.polylines(img, nplines, isClosed=True, color=(255, 125, 125), thickness=3)
        cv2.imwrite(osp.join(dst_path, img_id + '.jpg'), img)