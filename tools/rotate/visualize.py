from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
import numpy as np
import cv2 as cv
import random
import glob
import os
import os.path as osp
import argparse
import tqdm

def visualGT(cfg, num, dstpath, sp=None):
    dstpath = osp.join(dstpath, 'GT')
    os.makedirs(dstpath, exist_ok=True)
    img_path = cfg.data.train.img_prefix
    if cfg.data.train.type in ['SSDD']:
        img_path = osp.join(img_path, 'JPEGImages')
    if sp == 'None':
        imglist = glob.glob(img_path + '/*')
        selectnum = min(num, len(imglist))
        _tovis = random.sample(imglist, selectnum)
        for _singlevis in tqdm.tqdm(_tovis):
            img = cv.imread(_singlevis)
            ann_path = _singlevis.replace('images', 'labelTxt').replace('png', 'txt')
            assert osp.exists(ann_path), 'ann_file must be a file'
            with open(ann_path, 'r') as fread:
                lines = fread.readlines()
                if len(lines) == 0:
                    continue
                nplines = []
                # read lines
                for line in lines:
                    line = line.split()
                    npline = np.array(line[:8], dtype=np.float32).astype(np.int32)
                    nplines.append(npline[np.newaxis])
                nplines = np.concatenate(nplines, 0).reshape(-1, 4, 2)
                cv.polylines(img, nplines, isClosed=True, color=(255, 125, 125), thickness=3)
                cv.imwrite(osp.join(dstpath, osp.basename(_singlevis)), img)
    else:
        img_path = osp.join(img_path, sp)
        img = cv.imread(img_path)
        ann_path = img_path.replace('images', 'labelTxt').replace('png', 'txt')
        assert osp.exists(ann_path), 'ann_file must be a file'
        with open(ann_path, 'r') as fread:
            lines = fread.readlines()
            nplines = []
            # read lines
            for line in lines:
                line = line.split()
                npline = np.array(line[:8], dtype=np.float32).astype(np.int32)
                nplines.append(npline[np.newaxis])
            nplines = np.concatenate(nplines, 0).reshape(-1, 4, 2)
            cv.polylines(img, nplines, isClosed=True, color=(255, 125, 125), thickness=3)
            cv.imwrite(osp.join(dstpath, osp.basename(img)), img)
        pass




def visualINF(cfg, num, dstpath):
    dstpath = osp.join(dstpath, 'INF')
    os.makedirs(dstpath, exist_ok=True)
    device = 'cuda:0'
    # init a detector
    model = init_detector(cfg, checkpoint_file, device=device)
    img_path = cfg.data.test.img_prefix
    if cfg.data.test.type in ['SSDD']:
        with open(osp.join(img_path, 'test.txt'), 'r') as f:
            lines = f.readlines()
        imglist = [glob.glob(osp.join(img_path, 'JPEGImages', lines[i].strip()+'*'))[0] for i in range(len(lines))]
        print(len(imglist))
    else:
        imglist = glob.glob(img_path + '/*')
    selectnum = min(num, len(imglist))
    _tovis = random.sample(imglist, selectnum)
    for _singlevis in tqdm.tqdm(_tovis):
        base_name = os.path.basename(_singlevis)
        dst_name = os.path.join(dstpath, base_name)
        # inference the demo image
        result = inference_detector(model, _singlevis)
        show_result_pyplot(model, _singlevis, result, dst_name)

def visualSP(cfg, dstpath, name):
    dstpath = osp.join(dstpath, 'SP')
    os.makedirs(dstpath, exist_ok=True)
    device = 'cuda:0'
    # init a detector
    model = init_detector(cfg, checkpoint_file, device=device)
    img_path = cfg.data.test.img_prefix
    img_name = osp.join(img_path, name)
    dst_name = os.path.join(dstpath, name)
    result = inference_detector(model, img_name)
    show_result_pyplot(model, img_name, result, dst_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('pth', help='the pth need to load')
    parser.add_argument('--mode', default='INF', help='the way you need to visualize the result(GT/INF/SP)')
    parser.add_argument('--dst', default='/home/lzy/xyh/Netmodel/rotate_detection/checkpoints/visual')
    parser.add_argument('--num', type=int, default=5, help='the number of images you want to visual')
    parser.add_argument('--gtpath', default='None')
    parser.add_argument('--img', default='None')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    os.makedirs(args.dst, exist_ok=True)
    checkpoint_file = osp.join(cfg.work_dir, 'latest.pth')

    if args.mode == 'GT':
        visualGT(cfg, args.num, args.dst, args.img)

    elif args.mode == 'INF':
        visualINF(cfg, args.num, args.dst)

    elif args.mode == 'SP':
        visualSP(cfg, args.dst, args.img)


