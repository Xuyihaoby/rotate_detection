import os
import dota_utils as util
from multiprocessing import Pool
import cv2
import numpy as np
from functools import partial
import codecs

cate = {'roundabout': 0, 'tennis-court': 0, 'swimming-pool': 0, 'storage-tank': 0,
        'soccer-ball-field': 0, 'small-vehicle': 0, 'ship': 0, 'plane': 0,
        'large-vehicle': 0, 'helicopter': 0, 'harbor': 0, 'ground-track-field': 0,
        'bridge': 0, 'basketball-court': 0, 'baseball-diamond': 0,
        'container-crane': 0,
        'airport': 0, 'helipad': 0}

cate_2 = {'A': 0, 'B': 0, 'C': 0, 'D': 0,
        'E': 0, 'F': 0, 'G': 0, 'H': 0,
        'I': 0, 'J': 0, 'K': 0}

def rotate_single_run(name, srcpath):
    """

    """
    src_labelTxt = os.path.join(srcpath, 'labelTxt')

    objs = util.parse_dota_poly2(os.path.join(src_labelTxt, name + '.txt'))

    # cv2.imwrite(os.path.join(dst_imgpath, name + '_90.png'), img_90)
    # cv2.imwrite(os.path.join(dst_imgpath, name + '_180.png'), img_180)
    # cv2.imwrite(os.path.join(dst_imgpath, name + '_270.png'), img_270)
    for stem in objs:
        print('1')
        cate[stem['name']] += 1



if __name__ == '__main__':

    # srcpath = '/data/xuyihao/mmdetection/dataset/competetion/data'
    srcpath = '/data1/public_dataset/rsai/split/all'
    # pool = Pool(32)
    imgnames = util.GetFileFromThisRootDir(os.path.join(srcpath, 'images'))
    names = [util.custombasename(x) for x in imgnames]
    #
    for name in names:

        src_labelTxt = os.path.join(srcpath, 'labelTxt')

        objs = util.parse_dota_poly2(os.path.join(src_labelTxt, name + '.txt'))
        for stem in objs:
            cate_2[stem['name']] += 1


    print(cate_2)


