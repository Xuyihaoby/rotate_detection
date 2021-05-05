import numpy as np
import torch
import cv2

from .rotate_polygon_nms import rotate_gpu_nms
from .cpu_soft_nms_rotate import cpu_soft_nms_rotate


def nms_rotate(dets, iou_thr, device_id = None):
	'''
	GPU or CPU rotate NMS.
	
	Arguments:
		dets: [N, 6(x_ctr, y_ctr, w, h, theta, score)]
		iou_thr: float

	Returns:
		dets', index.
	'''
	if isinstance(dets, torch.Tensor):
		is_tensor = True
		if dets.is_cuda:
			device_id = dets.get_device()
		else:
			device_id = None
		dets_np = dets.detach().cpu().numpy()
	elif isinstance(dets, np.ndarray):
		is_tensor = False
		dets_np = dets
	else:
		raise TypeError(
			'dets must be either a Tensor or numpy array, but got {}'.format(
				type(dets)))

	if dets_np.shape[0] == 0:
		inds = []
	else:
		inds = (rotate_gpu_nms(dets_np, iou_thr, device_id = device_id)
				if device_id is not None else rotate_cpu_nms(dets_np, iou_thr))

	if is_tensor:
		inds = dets.new_tensor(inds, dtype = torch.long)
	else:
		inds = np.array(inds, dtype = np.int64)
	return dets[inds, :], inds


def soft_nms_rotate(dets, iou_thr, method = 'linear', sigma = 0.5, min_score = 1e-3):
	'''
	GPU or CPU rotate soft NMS.
	
	Arguments:
		dets: [N, 6(x_ctr, y_ctr, w, h, theta, score)]
		iou_thr: float

	Returns:
		dets', index.
	'''
	if isinstance(dets, torch.Tensor):
		is_tensor = True
		dets_np = dets.detach().cpu().numpy()
	elif isinstance(dets, np.ndarray):
		is_tensor = False
		dets_np = dets
	else:
		raise TypeError(
			'dets must be either a Tensor or numpy array, but got {}'.format(
				type(dets)))

	method_codes = {'linear': 1, 'gaussian': 2}
	if method not in method_codes:
		raise ValueError('Invalid method for SoftNMS: {}'.format(method))
	new_dets, inds = cpu_soft_nms_rotate(
		dets_np,
		iou_thr,
		method = method_codes[method],
		sigma = sigma,
		min_score = min_score)

	if is_tensor:
		return dets.new_tensor(new_dets), dets.new_tensor(
			inds, dtype = torch.long)
	else:
		return new_dets.astype(np.float32), inds.astype(np.int64)



def rotate_cpu_nms(dets, iou_thr):

	keep = []

	boxes = dets[:, :5]
	scores = dets[:, 5]
	order = scores.argsort()[::-1]
	num = boxes.shape[0]

	suppressed = np.zeros((num), dtype=np.int)

	for _i in range(num):

		i = order[_i]
		if suppressed[i] == 1:
			continue
		keep.append(i)
		r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
		area_r1 = boxes[i, 2] * boxes[i, 3]
		for _j in range(_i + 1, num):
			j = order[_j]
			if suppressed[i] == 1:
				continue
			r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
			area_r2 = boxes[j, 2] * boxes[j, 3]
			inter = 0.0

			int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
			if int_pts is not None:
				order_pts = cv2.convexHull(int_pts, returnPoints=True)

				int_area = cv2.contourArea(order_pts)

				inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-8)

			if inter >= iou_thr:
				suppressed[j] = 1

	return np.array(keep, np.int64)



# if __name__ == '__main__':
# 	boxes = np.array([[50, 50, 100, 100, 0],
# 					  [60, 60, 100, 100, 0],
# 					  [50, 50, 100, 100, -45.],
# 					  [200, 200, 100, 100, 0.]])

# 	scores = np.array([0.99, 0.88, 0.66, 0.77])

# 	keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
# 					  0.7, 5)

# 	import os
# 	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 	with tf.Session() as sess:
# 		print(sess.run(keep))
