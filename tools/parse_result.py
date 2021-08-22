import os

import mmcv
import numpy as np
import argparse
import os.path as osp
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from DOTA_devkit.ResultMerge_multi_process import mergebypoly_multiprocess
from DOTA_devkit.dota_evaluation_task1 import voc_eval
from terminaltables import AsciiTable
from mmcv.utils import print_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pkl', help='det results .pkl')
    parser.add_argument('nms', type=str, default='Y', help='det results .pkl')
    parser.add_argument('--type',type=str, default='OBB', help='boxmode')
    parser.add_argument('--eval', type=str, default='Y', help='whether to local evaluate')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    outputpath = cfg.work_dir
    outputs = mmcv.load(args.pkl)
    detpath = osp.join(outputpath, 'submission_test_r' + '/Task1_{:s}.txt')
    dataset.format_results(outputs, outputpath + '/submission_test', type=args.type)
    if args.nms == 'Y':
        print('start_nms')
        os.makedirs(osp.join(outputpath, 'submission_test_r_nms'), exist_ok=True)
        mergebypoly_multiprocess(osp.join(outputpath, 'submission_test_r'), osp.join(outputpath, 'submission_test_r_nms'))
        print('finish_nms')
        detpath = osp.join(outputpath, 'submission_test_r_nms' + '/Task1_{:s}.txt')

    if args.eval == 'Y':
        # annopath = '/data/xuyihao/mmdetection/dataset/DOTA/val/labelTxt/{:s}.txt'
        annopath = osp.join(cfg.data.val.ann_file, '{:s}.txt')
        imagesetfile = r'/data1/public_dataset/rsai/origin/val.txt'
        classnames = dataset.CLASSES
        classaps = []
        map = 0
        header = ['class', 'gts', 'dets', 'recall', 'ap']
        table_data = [header]
        for classname in classnames:
            print('classname:', classname)
            rec, prec, ap, npos, num_dets = voc_eval(detpath,
                                                     annopath,
                                                     imagesetfile,
                                                     classname,
                                                     ovthresh=0.5,
                                                     use_07_metric=True)
            map = map + ap
            classaps.append(ap)

        map = map / len(classnames)
        table_data.append(['mAP', '', '', '', f'{map:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table)
        print('map:', map)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)
        with open(outputpath + '/eval_results.txt', 'w') as f:
            res_str = 'mAP:' + str(map) + '\n'
            res_str += 'classaps: ' + ' '.join([str(x) for x in classaps])
            f.write(res_str)
