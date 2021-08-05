import os
import dota_utils as utils
import glob
import pdb
import cv2 as cv
import numpy as np
from multiprocessing import Pool
from functools import partial

DOTA_1 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
          'tennis-court',
          'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
          'helicopter']

DOTA_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
           'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
           'helicopter', 'container-crane']


def write(ann_file, src_imgpath, dst_imgpath, dst_labelTxt, categary):
    img_path = os.path.join(src_imgpath, os.path.splitext(os.path.basename(ann_file))[0]) + '.png'
    img = cv.imread(img_path)
    print(ann_file)
    h_img, w_img, c_img = img.shape
    # 保存图片到指定路径
    cv.imwrite(os.path.join(dst_imgpath, os.path.splitext(os.path.basename(ann_file))[0] + '.png'), img)
    with open(ann_file, 'r') as f:
        # 打开注释文件开始逐行读取
        str_format = '{:d} {:f} {:f} {:f} {:f} {:f}\n'
        str = []
        s = f.readlines()
        for sitem in s:
            sitem = sitem.strip().split()
            bbox = sitem[:8]
            bbox = [*map(lambda x: float(x), bbox)]
            bboxps = np.array(bbox).reshape(
                (4, 2)).astype(np.float32)
            rbbox = cv.minAreaRect(bboxps)
            x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
            while not 0 > a >= -90:
                if a >= 0:
                    a -= 90
                    w, h = h, w
                else:
                    a += 90
                    w, h = h, w
            catid = categary[sitem[8]]
            x_yolo, y_yolo, w_yolo, h_yolo, theta_yolo = x / w_img, y / h_img, w / w_img, h / h_img, \
                                                         a * np.pi / 180.0
            str_ = str_format.format(catid, x_yolo, y_yolo, w_yolo, h_yolo, theta_yolo)
            str.append(str_)
        dstannfile = os.path.join(dst_labelTxt, os.path.basename(ann_file))
        with open(dstannfile, 'w') as f_out:
            for towrite in str:
                f_out.write(towrite)


def DOTA2YOLO(srcpath, dstpath, cls_names):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    src_imgpath = os.path.join(srcpath, 'images')
    dst_imgpath = os.path.join(dstpath, 'images')
    src_labelTxt = os.path.join(srcpath, 'labelTxt')
    dst_labelTxt = os.path.join(dstpath, 'labelTxt')

    if not os.path.exists(dst_imgpath):
        os.mkdir(dst_imgpath)

    if not os.path.exists(dst_labelTxt):
        os.mkdir(dst_labelTxt)

    categary = {}
    for idex, name in enumerate(cls_names):
        categary[name] = idex
    ann_files = glob.glob(src_labelTxt + '/*.txt')
    if not ann_files:
        raise FileExistsError
    else:
        pool = Pool(16)
        write_fn = partial(write, src_imgpath=src_imgpath, dst_imgpath=dst_imgpath, dst_labelTxt=dst_labelTxt, \
                           categary=categary)
        pool.map(write_fn, ann_files)
        pool.close()
        # 单进程程序示例
        # # 开始读每一个注释文件
        # for ann_file in ann_files:
        #     # 得到注释文件名之后先得到对应的图片的长宽
        #     img_path = os.path.join(src_imgpath, os.path.splitext(os.path.basename(ann_file))[0])+'.png'
        #     img = cv.imread(img_path)
        #     h_img, w_img, c_img = img.shape
        #     # 保存图片到指定路径
        #     cv.imwrite(os.path.join(dst_imgpath, os.path.splitext(os.path.basename(ann_file))[0]+'.png'), img)
        #     with open(ann_file, 'r') as f:
        #         # 打开注释文件开始逐行读取
        #         str_format = '{:d} {:f} {:f} {:f} {:f} {:f}\n'
        #         str = []
        #         s = f.readlines()
        #         for sitem in s:
        #             sitem = sitem.strip().split()
        #             bbox = sitem[:8]
        #             bbox = [*map(lambda x: float(x), bbox)]
        #             bboxps = np.array(bbox).reshape(
        #                 (4, 2)).astype(np.float32)
        #             rbbox = cv.minAreaRect(bboxps)
        #             x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
        #             while not 0 > a >= -90:
        #                 if a >= 0:
        #                     a -= 90
        #                     w, h = h, w
        #                 else:
        #                     a += 90
        #                     w, h = h, w
        #             catid = categary[sitem[8]]
        #             x_yolo, y_yolo, w_yolo, h_yolo, theta_yolo = x / w_img, y / h_img, w / w_img, h / h_img, \
        #                                                          a * np.pi / 180.0
        #             str_ = str_format.format(catid, x_yolo, y_yolo, w_yolo, h_yolo, theta_yolo)
        #             str.append(str_)
        #         dstannfile = os.path.join(dst_labelTxt, os.path.basename(ann_file))
        #         with open(dstannfile, 'w') as f_out:
        #             for towrite in str:
        #                 f_out.write(towrite)


if __name__ == '__main__':
    DOTA2YOLO(r'/data/xuyihao/mmdetection/dataset/DOTA/examplesplit',
              r'/data/xuyihao/mmdetection/dataset/DOTA/exampleyolosplit',
              DOTA_1)
    # DOTA2YOLO(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024',
    #               r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024/DOTA_test1024.json',
    #               DOTA_1)
