import torch
from iou_rotate_wrapper import iou_rotate
import time

time1 = time.time()
boxes1 = torch.cuda.FloatTensor([[2.5, 3, 4, 3, -90]])
boxes2 = torch.cuda.FloatTensor([[2.5, 3, 4, 3, -90]])
iou = iou_rotate(boxes1, boxes2)
print(time.time() - time1)

import pdb
pdb.set_trace()
