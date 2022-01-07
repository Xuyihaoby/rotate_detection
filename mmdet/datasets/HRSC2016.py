# modified from https://github.com/csuhan/s2anet/blob/master/mmdet/datasets/hrsc2016.py
import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

import cv2 as cv
from .builder import DATASETS
from .xml_style import XMLDataset

import tempfile
from mmcv.utils import print_log


from mmdet.core import reval_map, rdets2points, heval_map
from mmdet.core.bbox.rtransforms import poly2obb_np
from tqdm import trange

import math


@DATASETS.register_module
class HRSC(XMLDataset):
    CLASSES = ('ship',)

    def __init__(self,  version='v1', *args, **kwargs):
        self.version = version
        super(HRSC, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        self.img_names = []

        for img_id in img_ids:
            filename = f'AllImages/{img_id}.bmp'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)
            self.img_names.append(img_id)
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos

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

        for obj in root.findall('HRSC_Objects')[0].findall('HRSC_Object'):
            label = self.cat2label['ship']
            difficult = int(obj.find('difficult').text)
            bbox = []
            hbbox = []
            for key in ['mbox_cx', 'mbox_cy', 'mbox_w', 'mbox_h', 'mbox_ang']:
                bbox.append(obj.find(key).text)
            for hkey in ['box_xmin', 'box_ymin', 'box_xmax', 'box_ymax']:
                hbbox.append(obj.find(hkey).text)

            # Coordinates may be float type
            cx, cy, w, h, a = list(map(float, bbox))
            xmin, ymin, xmax, ymax = list(map(float, hbbox))

            bbox = [cx, cy, w, h, a]
            poly = np.array(self.coordinate_convert_r(bbox), dtype=np.int32)
            rbbox = list(poly2obb_np(poly, self.version))
            hbbox = [xmin, ymin, xmax, ymax]

            bboxes.append(rbbox)
            hbboxes.append(hbbox)
            labels.append(label)
            polygons.append(poly)
        if not bboxes:
            bboxes = np.zeros((0, 5))
            hbboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
            polygons = np.zeros((0, 8))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            hbboxes = np.array(hbboxes, ndmin=2)
            labels = np.array(labels)
            polygons = np.array(polygons, ndmin=2)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            hbboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
            polygons_ignore = np.zeros((0, 8))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            hbboxes_ignore = np.array(hbboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
            polygons_ignore = np.array(polygons_ignore, ndmin=2)
        ann = dict(
            hbboxes=hbboxes.astype(np.float32),
            bboxes=bboxes.astype(np.float32),
            polygons=polygons.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            hbboxes_ignore=hbboxes_ignore.astype(np.float32),
            polygons_ignore=polygons_ignore.astype(np.float32),
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
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('HRSC_Objects')[0].findall('HRSC_Object'):
            label = self.cat2label['ship']
            cat_ids.append(label)

        return cat_ids

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('HRSC_Objects'):
                    name = obj.findall('HRSC_Object')
                    if len(name) >= 1:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def coordinate_convert_r(self, box):
        w, h = box[2:-1]
        theta = -box[-1]
        x_lu, y_lu = -w / 2, h / 2
        x_ru, y_ru = w / 2, h / 2
        x_ld, y_ld = -w / 2, -h / 2
        x_rd, y_rd = w / 2, -h / 2

        x_lu_ = math.cos(theta) * x_lu + math.sin(theta) * y_lu + box[0]
        y_lu_ = -math.sin(theta) * x_lu + math.cos(theta) * y_lu + box[1]

        x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
        y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

        x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
        y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

        x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
        y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

        convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

        return convert_box

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            if results[0][0].shape[1] == 5:
                mean_ap, _ = heval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
            else:
                mean_ap, _ = reval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
            eval_results['mAP'] = mean_ap
        return eval_results

