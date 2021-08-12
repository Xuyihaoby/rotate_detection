from mmcv import Config
import mmcv
from mmdet.datasets import build_dataloader, build_dataset
from DOTA_devkit.ResultMerge_multi_process import mergebypoly_multiprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('pkl', help='det results .pkl')
    parser.add_argument('--type', help='boxmode')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    outputpath = cfg.work_dir
    outputs = mmcv.load(args.pkl)
    dataset.format_results(outputs, outputpath+'./submission_test', type=args.type)
    mergebypoly_multiprocess(outputpath + r'./submisssion_test_r', outputpath + r'./submisssion_test_r')

