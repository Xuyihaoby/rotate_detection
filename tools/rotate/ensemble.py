import argparse
import glob
import os

from mmcv import Config
import os.path as osp
from DOTA_devkit.ResultMerge_multi_process import mergebypoly_multiprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='*', help='need ensemble config')
    args = parser.parse_args()

    ensemble = {}
    len_conf = len(args.configs)
    dst_dir_name = ''
    for config in args.configs:
        print(config)
        dst_dir_name += config[-10:-3]
        cfg = Config.fromfile(config)
        # print(dst_dir_name)
        filelist = glob.glob(cfg.work_dir+'/submission_test_r/'+'*.txt')
        for file in filelist:
            with open(file, 'r') as f:
                item = f.readlines()
                key = osp.basename(file)
                print('key:', key, '--->', 'len(item)=', len(item))
                if ensemble.get(key, None) is None:
                    ensemble[key]=[]
                ensemble[key].extend(item)
    dst_dir_name = osp.join(cfg.work_dir[:cfg.work_dir.find('checkpoints') + 12], 'ensemble', dst_dir_name)
    os.makedirs(dst_dir_name, exist_ok=True)
    for dst_key in ensemble.keys():
        dst_file = osp.join(dst_dir_name, dst_key)
        with open(dst_file, 'w') as fwrite:
            for content in ensemble[dst_key]:
                fwrite.write(content)
    os.makedirs(dst_dir_name+'_nms', exist_ok=True)
    print('start_nms')
    mergebypoly_multiprocess(dst_dir_name, dst_dir_name+'_nms')
    print('finish_nms')
    print('dst_path', dst_dir_name+'_nms')



