import numpy as np
import glob
import os.path as osp
import cv2
import os
import tempfile
from mmcv.utils import print_log

from .custom import CustomDataset
from .builder import DATASETS

from mmdet.core import reval_map, rdets2points, heval_map
from ..models.utils import transXyxyxyxy2Xyxy
from tqdm import trange


@DATASETS.register_module()
class RSAI(CustomDataset):
    """
        https://captain-whu.github.io/DOTA/dataset.html
    """
    CLASSES = ('A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H',
               'I', 'J', 'K')

    # 有待商榷

    def __init__(self, *args, **kwargs):
        self.difficulty_thresh = kwargs.pop('difficulty_thresh', 2)
        # set the default threshold to a big value. we take all gts as not difficult by default
        super(RSAI, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_folder):
        cls_map = {c: i for i, c in enumerate(
            self.CLASSES)}  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase, use image folder as ann_folder to generate pseudo annotations
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = dict()
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = dict()
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_info['ann']['hbboxes'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = dict()
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = dict()
                gt_bboxes = []
                hor_gt_boxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []
                hor_gt_boxes_ignore = []
                with open(ann_file) as f:
                    s = f.readlines()
                    # gsd = s[1].split(':')[-1]
                    # data_info['gsd'] = float(gsd) if gsd != 'null\n' else None
                    s = s[0:]
                    for si in s:
                        bbox_info = si.split()
                        # update by xyh at 2021.06.20
                        if bbox_info[9] == 2:
                            continue
                        bbox = bbox_info[:8]
                        bbox = [*map(lambda x: float(x), bbox)]

                        # transform
                        x1, y1, x2, y2 = transXyxyxyxy2Xyxy(boxes=bbox, with_label_last=False, is_single=True)

                        bboxps = np.array(bbox).reshape(
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

                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]

                        if difficulty > self.difficulty_thresh:
                            gt_bboxes_ignore.append([x, y, w, h, a])
                            gt_labels_ignore.append(label)
                            gt_polygons_ignore.append(bbox)
                            hor_gt_boxes_ignore.append([x1, y1, x2, y2])
                        elif difficulty <= 1:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(bbox)
                            hor_gt_boxes.append([x1, y1, x2, y2])

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                    data_info['ann']['hbboxes'] = np.array(
                        hor_gt_boxes, dtype=np.float32
                    )
                else:
                    data_info['ann']['bboxes'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros(
                        (0, 8), dtype=np.float32)
                    data_info['ann']['hbboxes'] = np.zeros(
                        (0, 4), dtype=np.float32
                    )

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                    data_info['ann']['hbboxes_ignore'] = np.array(
                        hor_gt_boxes_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)
                    data_info['ann']['hbboxes_ignore'] = np.zeros(
                        (0, 4), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

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
            print_log(f'\n{"-" * 18}iou_thr: {iou_thr}{"-" * 18}')
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

    def _det2str(self, results):
        mcls_results = {cls: '' for cls in self.CLASSES}
        for idx in trange(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                if result[label].shape[1] == 6:
                    bboxes = rdets2points(result[label])
                    resstr = '{:s} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                elif result[label].shape[1] == 5:
                    bboxes = result[label]
                    resstr = '{:s} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                cls_name = self.CLASSES[label]
                for i in range(bboxes.shape[0]):
                    # resstr = '{:s} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                    ps = list(bboxes[i][:-1])
                    score = float(bboxes[i][-1])
                    resstr_ = resstr.format(img_id, score, *ps)
                    mcls_results[cls_name] += resstr_
        return mcls_results

    def _results2submission_h(self, results, out_folder=None):
        out_folder = out_folder + '_h'
        dota_results = self._det2str(results)
        if out_folder is not None:
            os.makedirs(out_folder, exist_ok=True)
            for cls in dota_results:
                fname = f'Task2_{cls}.txt'
                fname = os.path.join(out_folder, fname)
                with open(fname, 'w') as f:
                    f.write(dota_results[cls])
        return dota_results

    def _results2submission(self, results, out_folder=None):
        dota_results = self._det2str(results)
        out_folder = out_folder + '_r'
        if out_folder is not None:
            os.makedirs(out_folder, exist_ok=True)
            for cls in dota_results:
                fname = f'Task1_{cls}.txt'
                fname = os.path.join(out_folder, fname)
                with open(fname, 'w') as f:
                    f.write(dota_results[cls])
        return dota_results

    def format_results(self, results, submission_dir=None, type='all', **kwargs):
        """Format the results to submission text (standard format for DOTA evaluation).
        Args:
            results (list/dict): Testing results of the dataset.
            submission_dir (str | None): The folder that contains submission files.
                If not specified, a temp folder will be created. Default: None.
        """
        # results (list)[(list)[[bbox],....(classnum)],...,numimg]
        assert isinstance(results, list), 'results must be a list'
        if isinstance(results[0], list):
            assert len(results) == len(self), (
                'The length of results is not equal to the dataset len: {} != {}'.
                    format(len(results), len(self)))
            if results[0][0].shape[-1] == 5:
                results_h = results
            if results[0][0].shape[-1] == 6:
                results_r = results
        if isinstance(results[0], dict):
            assert len(results) == len(self), (
                'The length of results is not equal to the dataset len: {} != {}'.
                    format(len(results), len(self)))
            results_h = []
            results_r = []
            for i in range(len(results)):
                if 'hbb' in results[i].keys():
                    results_h.extend(results[i]['hbb'])
                results_r.extend(results[i]['rbb'])
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None
        if type == 'all':
            print('begin rotate bbox format')
            result_files_r = self._results2submission(results_r, submission_dir)
            print('rotate bbox complete and horizon bbox begin')
            result_files_h = self._results2submission_h(results_h, submission_dir)
            print('all done')
        if type == 'OBB':
            print('only rotate')
            result_files_r = self._results2submission(results_r, submission_dir)
            print('done')
        if type == 'HBB':
            print('rotate bbox complete and horizon bbox begin')
            result_files_h = self._results2submission_h(results_h, submission_dir)
            print('done')
