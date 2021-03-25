import numpy as np
import torch
import cv2

from .rbbox_overlaps import rbbx_overlaps
from .iou_cpu import get_iou_matrix


def iou_rotate(boxes1, boxes2):
	'''
	GPU or CPU rotate IOU.
	
	Arguments:
		boxes1: [N1, 5(x_ctr, y_ctr, w, h, theta(angle))]
		boxes2: [N2, 5(x_ctr, y_ctr, w, h, theta(angle))]

	Returns:
		IoU: [N1, N2]
	'''
	if isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor):
		is_tensor = True
		if boxes1.is_cuda:
			device_id = boxes1.get_device()
		else:
			device_id = None
		boxes1_np = boxes1.detach().cpu().numpy()
		boxes2_np = boxes2.detach().cpu().numpy()
	elif isinstance(boxes1, np.ndarray) and isinstance(boxes2, np.ndarray):
		is_tensor = False
		boxes1_np = boxes1
		boxes2_np = boxes2
	else:
		raise TypeError(
			'boxes1 and boxes2 must have the same type, but got {} for boxes1 and {} for boxes2'.format(
				type(boxes1), type(boxes2)))

	iou_matrix = (rbbx_overlaps(boxes1_np, boxes2_np, device_id = device_id)
					if device_id is not None else get_iou_matrix(boxes1_np, boxes2_np))

	if is_tensor:
		iou_matrix = boxes1.new_tensor(iou_matrix, dtype = torch.float)
	else:
		iou_matrix = np.array(iou_matrix, dtype = np.float32)
	return iou_matrix

if __name__ == '__main__':
	import time

	time1 = time.time()
	boxes1 = torch.cuda.FloatTensor([[2.5, 3, 4, 3, -90]])
	boxes2 = torch.cuda.FloatTensor([[2.5, 3, 4, 3, -90]])
	iou = iou_rotate(boxes1, boxes2)
	print(time.time() - time1)

	import pdb
	pdb.set_trace()



# def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):
# 	'''

# 	:param boxes_list1:[N, 8] tensor
# 	:param boxes_list2: [M, 8] tensor
# 	:return:
# 	'''

# 	boxes1 = tf.cast(boxes1, tf.float32)
# 	boxes2 = tf.cast(boxes2, tf.float32)
# 	if use_gpu:

# 		iou_matrix = tf.py_func(rbbx_overlaps,
# 								inp=[boxes1, boxes2, gpu_id],
# 								Tout=tf.float32)
# 	else:
# 		iou_matrix = tf.py_func(get_iou_matrix, inp=[boxes1, boxes2],
# 								Tout=tf.float32)

# 	iou_matrix = tf.reshape(iou_matrix, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

# 	return iou_matrix


# def iou_rotate_calculate1(boxes1, boxes2, use_gpu=True, gpu_id=0):

# 	# start = time.time()
# 	if use_gpu:
# 		ious = rbbx_overlaps(boxes1, boxes2, gpu_id)
# 	else:
# 		area1 = boxes1[:, 2] * boxes1[:, 3]
# 		area2 = boxes2[:, 2] * boxes2[:, 3]
# 		ious = []
# 		for i, box1 in enumerate(boxes1):
# 			temp_ious = []
# 			r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
# 			for j, box2 in enumerate(boxes2):
# 				r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

# 				int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
# 				if int_pts is not None:
# 					order_pts = cv2.convexHull(int_pts, returnPoints=True)

# 					int_area = cv2.contourArea(order_pts)

# 					inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
# 					temp_ious.append(inter)
# 				else:
# 					temp_ious.append(0.0)
# 			ious.append(temp_ious)

# 	# print('{}s'.format(time.time() - start))

# 	return np.array(ious, dtype=np.float32)


# if __name__ == '__main__':
#	 import os
#	 os.environ["CUDA_VISIBLE_DEVICES"] = '13'
#	 boxes1 = np.array([[50, 50, 100, 300, 0],
#						[60, 60, 100, 200, 0]], np.float32)

#	 boxes2 = np.array([[50, 50, 100, 300, -45.],
#						[200, 200, 100, 200, 0.]], np.float32)

#	 start = time.time()
#	 with tf.Session() as sess:
#		 ious = iou_rotate_calculate1(boxes1, boxes2, use_gpu=False)
#		 print(sess.run(ious))
#		 print('{}s'.format(time.time() - start))

#	 # start = time.time()
#	 # for _ in range(10):
#	 #	 ious = rbbox_overlaps.rbbx_overlaps(boxes1, boxes2)
#	 # print('{}s'.format(time.time() - start))
#	 # print(ious)

#	 # print(ovr)



