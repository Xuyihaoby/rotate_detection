import cv2 as cv
import numpy as np
import time
import os
import os.path as osp

def vispolysave(img, poly, dstpath):
    _img = img.deepcopy()
    if poly.shape[-1] == 5:
        fourpoint = []
        for single in poly:
            singlefourpoint = cv.boxPoints(((single[0], single[1]), (single[1], single[2]), single[3] * 180/np.pi))
            fourpoint.append(singlefourpoint)
        cv.polylines(_img, np.array(fourpoint))
        cv.imwrite(osp.join(dstpath, str(int(time.time()%1000))+'.png'), _img)