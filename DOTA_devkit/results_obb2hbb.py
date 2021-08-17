import DOTA_devkit.utils as util
import os

import argparse
# dota2_annopath = r'data/dota2_test-dev/labelTxt/{:s}.txt'
# dota2_imagesetfile = r'data/dota2_test-dev/test.txt'


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument(r'--path', default=r'/home/dj/code/mmdetection_DOTA/work_dirs/faster_rcnn_r50_fpn_1x_dota_RoITrans_v2/save1_nms2000')
#     # parser.add_argument('--version', default='dota_v1',
#     #                     help='dota version')
#     args = parser.parse_args()
#
#     return args

def OBB2HBB(srcpath, dstpath):
    assert os.path.exists(srcpath), 'srcpath error'
    filenames = util.GetFileFromThisRootDir(srcpath)
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    for file in filenames:
        with open(file, 'r') as f_in:
            tempname = os.path.basename(file).replace('1', '2', 1)
            with open(os.path.join(dstpath, tempname), 'w') as f_out:
                lines = f_in.readlines()
                splitlines = [x.strip().split() for x in lines]
                for index, splitline in enumerate(splitlines):
                    imgname = splitline[0]
                    score = splitline[1]
                    poly = splitline[2:]
                    poly = list(map(float, poly))
                    xmin, xmax, ymin, ymax = min(poly[0::2]), max(poly[0::2]), min(poly[1::2]), max(poly[1::2])
                    rec_poly = [xmin, ymin, xmax, ymax]
                    outline = imgname + ' ' + score + ' ' + ' '.join(map(str, rec_poly))
                    if index != (len(splitlines) - 1):
                        outline = outline + '\n'
                    f_out.write(outline)

if __name__ == '__main__':
    # args = parse_args()
    # obb_results_path = '/data/xuyihao/mmdetection/mycode/11epoch_HSP_R50_fakep3_7ratio_lr005_r'
    obb_results_path = '/home/lzy/xyh/Netmodel/rotate_detection/checkpoints/DOTA2_0/faster_rcnn_r50_1x/submission_test_r_nms'
    hbb_results_path = '/home/lzy/xyh/Netmodel/rotate_detection/checkpoints/DOTA2_0/faster_rcnn_r50_1x/submission_test_h'
    OBB2HBB(obb_results_path, hbb_results_path)