'''
Writen by Chengzheng Li
'''
import numpy as np
cimport numpy as np
import cv2



# cdef np.ndarray[np.int32_t, ndim = 1] 
def cpu_soft_nms_rotate(
			np.ndarray[np.float32_t, ndim = 2] boxes_in, ## dets: [N, 6(x_ctr, y_ctr, w, h, theta, score)]
			np.float32_t iou_thr, 
			unsigned int method = 1,
			np.float32_t sigma = 0.5,
			np.float32_t min_score = 0.001
			):
	boxes = boxes_in.copy()
	cdef unsigned int N = boxes.shape[0]
	cdef int pos = 0
	cdef float maxscore = 0
	cdef int maxpos = 0
	cdef float x, y, w, h, theta, score, area, tx, ty, tw, th, ttheta, tscore, tarea
	cdef float weight, overlap, inter_area
	cdef int i, ti
	cdef np.ndarray[np.float32_t, ndim = 3] inter_


	inds = np.arange(N)

	for i in range(N):
		maxscore = boxes[i, 5]
		maxpos = i

		tx = boxes[i, 0]
		ty = boxes[i, 1]
		tw = boxes[i, 2]
		th = boxes[i, 3]
		ttheta = boxes[i, 4]
		tscore = boxes[i, 5]
		ti = inds[i]

		pos = i + 1

		## get max box
		while pos < N:
			if maxscore < boxes[pos, 5]:
				maxscore = boxes[pos, 5]
				maxpos = pos
			pos = pos + 1

		## add max box as a detection
		boxes[i, 0] = boxes[maxpos, 0]
		boxes[i, 1] = boxes[maxpos, 1]
		boxes[i, 2] = boxes[maxpos, 2]
		boxes[i, 3] = boxes[maxpos, 3]
		boxes[i, 4] = boxes[maxpos, 4]
		boxes[i, 5] = boxes[maxpos, 5]
		inds[i] = inds[maxpos]

		## swap ith box with postion of max box
		boxes[maxpos, 0] = tx
		boxes[maxpos, 1] = ty
		boxes[maxpos, 2] = tw
		boxes[maxpos, 3] = th
		boxes[maxpos, 4] = ttheta
		boxes[maxpos, 5] = tscore
		inds[maxpos] = ti

		tx = boxes[i, 0]
		ty = boxes[i, 1]
		tw = boxes[i, 2]
		th = boxes[i, 3]
		ttheta = boxes[i, 4]
		tscore = boxes[i, 5]

		pos = i + 1
		## NMS iterations, note that N changes if detection boxes fall below
		## threshold
		while pos < N:
			x = boxes[pos, 0]
			y = boxes[pos, 1]
			w = boxes[pos, 2]
			h = boxes[pos, 3]
			theta = boxes[pos, 4]
			# score = boxes[pos, 5]

			## ctr_x, ctr_y, w, h, theta(degree)
			poly = ((x,y), (w,h), theta)
			tpoly = ((tx,ty), (tw,th), ttheta)
			## return (0/1/2, points)
			## 0 means separate, 1 means cross, 2 means include.
			## None, points, points
			inter = cv2.rotatedRectangleIntersection(poly, tpoly)[1]
			if inter is not None: ## cross or inclue
				## inter is unordered, use convexHull to order it
				inter_ = cv2.convexHull(inter, returnPoints = True) ## [N, 1, 2]
				inter_area = cv2.contourArea(inter_)
				area = w * h
				tarea = tw * th
				overlap = inter_area / (area + tarea - inter_area)

				if method == 1: ## linear
					if overlap > iou_thr:
						weight = 1. - overlap
					else:
						weight = 1.
				elif method == 2: ## gaussian
					weight = np.exp(-(overlap * overlap) / sigma)
				else: ## original nms
					if overlap > iou_thr:
						weight = 0.
					else:
						weight = 1.

				boxes[pos, 5] = weight * boxes[pos, 5]

				## if box score falls below threshold, discard the box by
				## swapping with the last box, and update N.
				if boxes[pos, 5] < min_score:
					boxes[pos, 0] = boxes[N - 1, 0]
					boxes[pos, 1] = boxes[N - 1, 1]
					boxes[pos, 2] = boxes[N - 1, 2]
					boxes[pos, 3] = boxes[N - 1, 3]
					boxes[pos, 4] = boxes[N - 1, 4]
					boxes[pos, 5] = boxes[N - 1, 5]
					inds[pos] = inds[N - 1]
					N = N - 1
					pos = pos - 1
			pos = pos + 1

	return boxes[:N], inds[:N]
