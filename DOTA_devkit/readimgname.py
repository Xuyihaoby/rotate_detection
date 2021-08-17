import glob
import os

annofile = '/data1/public_dataset/rsai/origin/val/images/'
destdile = '/data1/public_dataset/rsai/origin/val.txt'
allname = glob.glob(annofile + '*.png')
basename = []
for i in range(len(allname)):
    basename.append(os.path.basename(os.path.splitext(allname[i])[0]))

with open(destdile, 'w') as fn:
    for i in range(len(basename)):
        fn.write(basename[i] + '\n')
