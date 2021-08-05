import glob
import os

annofile = '/data/xuyihao/mmdetection/dataset/DOTA/val/images/'
destdile = '/data/xuyihao/mmdetection/dataset/DOTA/val.txt'
allname = glob.glob(annofile + '*.png')
basename = []
for i in range(len(allname)):
    basename.append(os.path.basename(os.path.splitext(allname[i])[0]))

with open(destdile, 'w') as fn:
    for i in range(len(basename)):
        fn.write(basename[i] + '\n')
