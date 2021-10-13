import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
import argparse
import time
import os
import cv2
from mmcv import Config
import json
import numpy as np
import tqdm


def alg(input_dir, output_dir, device):
    CLASSES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K')
    # Specify the path to model config and checkpoint file
    config_file = '/work/configs/rsai/swin_pafpn_2x.py'
    checkpoint_file = '/work/configs/rsai/work_dir/epoch_24.pth'

    cfg = Config.fromfile(config_file)
    cfg.data.test.pipeline[1]['img_scale'] = (4096, 4096)

    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint_file, device)

    # 确定路径
    # 提交之前需要将这一行打开
    filelist = os.listdir(input_dir)
    outputfile = os.path.join(output_dir, 'aircraft_results.json')

    jsondict = []

    # 提交之前需要将这两行注释掉(用于调试)
    # input_dir = '/data/xuyihao/mmdetection/dataset/competetion/data/images'
    # filelist = os.listdir(input_dir)
    # outputfile = './aircraft_results.json'

    for imgname in tqdm.tqdm(filelist):
        # 每张图片为一个字典(key:"img_name", "labels")，所有标注组成一个列表
        # 单个标注为一个字典(key:"category_id", "points", "confidence")
        singlefile = {}
        singlefilelabels = []
        singlefile["image_name"] = imgname
        imgpath = os.path.join(input_dir, imgname)
        # test a single image and show the results
        results = inference_detector(model, imgpath)

        # 遍历11个种类的每一个类别的飞机标注
        for idx, result in enumerate(results):
            # 遍历每一类标注的每一个标注信息
            for singlebox in result:
                singlefilesinglelabel = {}
                singlefilesinglelabel["category_id"] = CLASSES[idx]
                tupbox = ((singlebox[0], singlebox[1]), (singlebox[2], singlebox[3]), singlebox[4] * 180 / np.pi)
                singlefilesinglelabel["points"] = cv2.boxPoints(tupbox).tolist()
                singlefilesinglelabel["confidence"] = float(singlebox[5])
                # TODO check rightness visualize
                singlefilelabels.append(singlefilesinglelabel)  # 字典组成列表
        singlefile["labels"] = singlefilelabels
        jsondict.append(singlefile)  # 单张图片字典组成所有列表

    # 写入json文件
    file = open(outputfile, 'w', encoding='utf-8')
    json.dump(jsondict, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument("--input_dir", default='/input_path', help="input path", type=str)
    # 输出路径
    parser.add_argument("--output_dir", default='/output_path', help="output path", type=str)
    args = parser.parse_args()
    start_time = time.time()
    torch.backends.cudnn.benchmark = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    alg(args.input_dir, args.output_dir, device)
    print('total time:', time.time() - start_time)
