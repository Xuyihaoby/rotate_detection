import mmcv
import numpy as np
import cv2
import torch
# from ..bbox import bbox_mapping_back
# from mmdet.models.utils import transRotate2Quadrangle

########## refine rotate by horizontal
def bbox2delta_rotate(proposals, gt, 
						means=[0., 0., 0., 0., 0.], 
						stds=[0.1, 0.1, 0.2, 0.2, 0.2], norm_degree_by90 = False):
	## proposals: [N, 4(x1,y1, x2,y2)]
	## gt: [N, 5(x_ctr,y_ctr, w,h, theta)], theta is degree, -90<=theta<0.
	assert proposals.size()[1] + 1 == gt.size()[1]
	assert proposals.size()[0] == gt.size()[0]

	proposals = proposals.float()
	gt = gt.float()
	px = (proposals[..., 0] + proposals[..., 2]) * 0.5
	py = (proposals[..., 1] + proposals[..., 3]) * 0.5
	pw = proposals[..., 2] - proposals[..., 0] + 1.0
	ph = proposals[..., 3] - proposals[..., 1] + 1.0
	if norm_degree_by90:
		########## Updated By LCZ, time: 2019.3.18, norm by / 90 ##########
		#### It seems to no help.
		ptheta = proposals.new_ones(proposals.size()[0]).float() * (-90.) ## degree
	else:
		ptheta = proposals.new_ones(proposals.size()[0]).float() * (-np.pi / 2.) ## radian


	# gx = (gt[..., 0] + gt[..., 2]) * 0.5
	# gy = (gt[..., 1] + gt[..., 3]) * 0.5
	# gw = gt[..., 2] - gt[..., 0] + 1.0
	# gh = gt[..., 3] - gt[..., 1] + 1.0
	gx = gt[..., 0]
	gy = gt[..., 1]
	gw = gt[..., 2]
	gh = gt[..., 3]
	if norm_degree_by90:
		gtheta = gt[..., 4] ## degree
	else:
		gtheta = gt[..., 4] * np.pi / 180. ## degree --> radian


	dx = (gx - px) / pw
	dy = (gy - py) / ph
	dw = torch.log(gw / pw)
	dh = torch.log(gh / ph)
	if norm_degree_by90:
		dtheta = (gtheta - ptheta) / 90. ## degree difference
	else:
		dtheta = gtheta - ptheta ## radian difference
	deltas = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)

	means = deltas.new_tensor(means).unsqueeze(0)
	stds = deltas.new_tensor(stds).unsqueeze(0)
	deltas = deltas.sub_(means).div_(stds)

	return deltas


def delta2bbox_rotate(rois, deltas,
			   means=[0., 0., 0., 0., 0.],
			   stds=[0.1, 0.1, 0.2, 0.2, 0.2],
			   max_shape=None,
			   wh_ratio_clip=16 / 1000,
			   norm_degree_by90 = False):
	## rois: [N, 4(x1,y1, x2,y2)]
	## deltas: [N, 5(dx,dy, dw,dh, dtheta)], theta is radian.
	means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
	stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
	denorm_deltas = deltas * stds + means
	dx = denorm_deltas[:, 0::5] ## [N, 1]
	dy = denorm_deltas[:, 1::5]
	dw = denorm_deltas[:, 2::5]
	dh = denorm_deltas[:, 3::5]
	if norm_degree_by90:
		dtheta = denorm_deltas[:, 4::5] * 90. ## degree
	else:
		dtheta = denorm_deltas[:, 4::5] * 180. / np.pi ## radian --> degree

	max_ratio = np.abs(np.log(wh_ratio_clip))
	dw = dw.clamp(min=-max_ratio, max=max_ratio)
	dh = dh.clamp(min=-max_ratio, max=max_ratio)
	px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx) ## [N, 1]
	py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
	pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
	ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
	ptheta = rois.new_ones(rois.size()[0]).float().unsqueeze(1).expand_as(dtheta) * (-90.) ## degree
	gw = pw * dw.exp()
	gh = ph * dh.exp()
	gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
	gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
	gtheta = ptheta + dtheta
	# x1 = gx - gw * 0.5 + 0.5
	# y1 = gy - gh * 0.5 + 0.5
	# x2 = gx + gw * 0.5 - 0.5
	# y2 = gy + gh * 0.5 - 0.5
	if max_shape is not None:
		gx = gx.clamp(min=0, max=max_shape[1] - 1)
		gy = gy.clamp(min=0, max=max_shape[0] - 1)
		gw = gw.clamp(min=0, max=max_shape[1] - 1)
		gh = gh.clamp(min=0, max=max_shape[0] - 1)
		gtheta = gtheta.clamp(min = -90, max = -0.00001)
	bboxes = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view_as(deltas)
	return bboxes




########## refine rotate by rotate
def bbox2delta_rotate_from_rotate(proposals, gt, 
									means=[0., 0., 0., 0., 0.], 
									stds=[0.1, 0.1, 0.2, 0.2, 0.2], norm_degree_by90 = False):
	## proposals: [N, 5(x_ctr,y_ctr, w,h, theta)], theta, is radian
	## gt: [N, 5(x_ctr,y_ctr, w,h, theta)], theta is degree, -90<=theta<0.
	## returns: [N, 5(dx,dy, dw,dh, dtheta)], theta is radian.
	assert proposals.size()[1] == gt.size()[1]
	assert proposals.size()[0] == gt.size()[0]

	proposals = proposals.float()
	gt = gt.float()
	px = proposals[..., 0]
	py = proposals[..., 1]
	pw = proposals[..., 2]
	ph = proposals[..., 3]
	if norm_degree_by90:
		########## Updated By LCZ, time: 2019.3.18, norm by / 90 ##########
		#### It seems to no help.
		# ptheta = proposals.new_ones(proposals.size()[0]).float() * (-90.) ## degree
		raise NotImplementedError
	else:
		# ptheta = proposals.new_ones(proposals.size()[0]).float() * (-np.pi / 2.) ## radian
		ptheta = proposals[..., 4] ## radian


	# gx = (gt[..., 0] + gt[..., 2]) * 0.5
	# gy = (gt[..., 1] + gt[..., 3]) * 0.5
	# gw = gt[..., 2] - gt[..., 0] + 1.0
	# gh = gt[..., 3] - gt[..., 1] + 1.0
	gx = gt[..., 0]
	gy = gt[..., 1]
	gw = gt[..., 2]
	gh = gt[..., 3]
	if norm_degree_by90:
		# gtheta = gt[..., 4] ## degree
		raise NotImplementedError
	else:
		gtheta = gt[..., 4] * np.pi / 180. ## degree --> radian


	dx = (gx - px) / pw
	dy = (gy - py) / ph
	dw = torch.log(gw / pw)
	dh = torch.log(gh / ph)
	if norm_degree_by90:
		# dtheta = (gtheta - ptheta) / 90. ## degree difference
		raise NotImplementedError
	else:
		dtheta = gtheta - ptheta ## radian difference
	deltas = torch.stack([dx, dy, dw, dh, dtheta], dim=-1)

	means = deltas.new_tensor(means).unsqueeze(0)
	stds = deltas.new_tensor(stds).unsqueeze(0)
	deltas = deltas.sub_(means).div_(stds)

	return deltas


def delta2bbox_rotate_from_rotate(rois, deltas,
			   means=[0., 0., 0., 0., 0.],
			   stds=[0.1, 0.1, 0.2, 0.2, 0.2],
			   max_shape=None,
			   wh_ratio_clip=16 / 1000,
			   norm_degree_by90 = False):
	## rois: [N, 5(x_ctr,y_ctr, w,h, theta)], theta is radian.
	## deltas: [N, 5(dx,dy, dw,dh, dtheta)], theta is radian.
	## returns: [N, 5(x_ctr,y_ctr, w,h, theta)], theta is radian.
	means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
	stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
	denorm_deltas = deltas * stds + means
	dx = denorm_deltas[:, 0::5] ## [N, 1]
	dy = denorm_deltas[:, 1::5]
	dw = denorm_deltas[:, 2::5]
	dh = denorm_deltas[:, 3::5]
	if norm_degree_by90:
		# dtheta = denorm_deltas[:, 4::5] * 90. ## degree
		raise NotImplementedError
	else:
		dtheta = denorm_deltas[:, 4::5] ## radian

	max_ratio = np.abs(np.log(wh_ratio_clip))
	dw = dw.clamp(min=-max_ratio, max=max_ratio)
	dh = dh.clamp(min=-max_ratio, max=max_ratio)
	px = rois[:, 0].unsqueeze(1).expand_as(dx) ## [N, 1]
	py = rois[:, 1].unsqueeze(1).expand_as(dy)
	pw = rois[:, 2].unsqueeze(1).expand_as(dw)
	ph = rois[:, 3].unsqueeze(1).expand_as(dh)
	# ptheta = rois.new_ones(rois.size()[0]).float().unsqueeze(1).expand_as(dtheta) * (-90.) ## degree
	ptheta = rois[:, 4].unsqueeze(1).expand_as(dtheta) ## radian
	gw = pw * dw.exp()
	gh = ph * dh.exp()
	gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
	gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
	gtheta = ptheta + dtheta ## radian
	# x1 = gx - gw * 0.5 + 0.5
	# y1 = gy - gh * 0.5 + 0.5
	# x2 = gx + gw * 0.5 - 0.5
	# y2 = gy + gh * 0.5 - 0.5
	if max_shape is not None:
		gx = gx.clamp(min=0, max=max_shape[1] - 1)
		gy = gy.clamp(min=0, max=max_shape[0] - 1)
		gw = gw.clamp(min=0, max=max_shape[1] - 1)
		gh = gh.clamp(min=0, max=max_shape[0] - 1)
		gtheta = gtheta.clamp(min = -90 * np.pi / 180., max = -0.00001 * np.pi / 180.)
	bboxes = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view_as(deltas)
	return bboxes




# def bbox_flip(bboxes, img_shape):
# 	"""Flip bboxes horizontally.

# 	Args:
# 		bboxes(Tensor or ndarray): Shape (..., 4*k)
# 		img_shape(tuple): Image shape.

# 	Returns:
# 		Same type as `bboxes`: Flipped bboxes.
# 	"""
# 	if isinstance(bboxes, torch.Tensor):
# 		assert bboxes.shape[-1] % 4 == 0
# 		flipped = bboxes.clone()
# 		flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
# 		flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
# 		return flipped
# 	elif isinstance(bboxes, np.ndarray):
# 		return mmcv.bbox_flip(bboxes, img_shape)


# def bbox_mapping(bboxes, img_shape, scale_factor, flip):
# 	"""Map bboxes from the original image scale to testing scale"""
# 	new_bboxes = bboxes * scale_factor
# 	if flip:
# 		new_bboxes = bbox_flip(new_bboxes, img_shape)
# 	return new_bboxes


# def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
# 	"""Map bboxes from testing scale to original image scale"""
# 	new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
# 	new_bboxes = new_bboxes / scale_factor
# 	return new_bboxes


def bbox2result_rotate(bboxes_h, labels_h, bboxes_r, labels_r, is_box_voting = False):
	"""Convert detection results to a list of numpy arrays.

	Args:
		bboxes_h (Tensor): shape (n, 5)
		labels_h (Tensor): shape (n, ), 0-start.
		bboxes_r (Tensor): shape (n, 6)/(n, 9)
		labels_r (Tensor): shape (n, ), 0-start.

	Returns:
		list(dict): bbox results of each type
	"""
	## horizontal
	if bboxes_h.shape[0] == 0:
		results_h = np.zeros((0, 6), dtype = np.float32) ## [n, 6(x1,y1, x2,y2, score, label(0-start.))]
	else:
		## [n, 6(x1,y1, x2,y2, score, label(1-start.))]
		## convert from 0-start to 1-start, just for use merge_func
		results_h = torch.cat([bboxes_h.float(), labels_h.unsqueeze(1).float() + 1.], dim = -1).cpu().numpy()

	if bboxes_r is None and labels_r is None:
		return dict(horizontal = results_h)
	else:
		## rotate
		if bboxes_r.shape[0] == 0:
			results_r = np.zeros((0, 10), dtype = np.float32) ## [n, 10(x1,y1, x2,y2, x3,y3, x4,y4, score, label(0-start.))]
		else:
			bboxes_r = bboxes_r.float().cpu().numpy() ## [n, 6]/[n, 9]
			labels_r = labels_r.float().cpu().numpy() ## [n]

			if not is_box_voting:
				## [n, 10(x1,y1, x2,y2, x3,y3, x4,y4, score, label(1-start.))]
				## convert from 0-start to 1-start, just for use merge_func
				bboxes_r = transRotate2Quadrangle(coordinates = bboxes_r, with_label_last = True)
			else:
				bboxes_r = bboxes_r
			results_r = np.concatenate([bboxes_r, np.expand_dims(labels_r, axis = -1) + 1.], axis = -1)

		return dict(horizontal = results_h, rotate = results_r)



def transRotate2Quadrangle(coordinates, with_label_last = False):
	"""
	Transform boxes from (x_ctr, y_ctr, w, h, theta(label)) to (x1,y1, x2,y2, x3,y3, x4,y4(label)).

	Arguments:
		coordinates: ndarray, [N, (x_ctr, y_ctr, w, h, theta)] 

	Returns:
		coordinates: ndarray, [N, (x1,y1, x2,y2, x3,y3, x4,y4)] 
	"""
	if with_label_last:
		tp = coordinates[:, :-1] ## [N, 5]
		label = coordinates[:, -1] ## [N]
	else:
		tp = coordinates ## [N, 5]

	result = []
	for cd in tp:
		quad = cv2.boxPoints(((cd[0], cd[1]), (cd[2], cd[3]), cd[4]))
		result.append(np.reshape(quad, [-1, ]))
	result = np.array(result, dtype = np.float32) ## [N, 8]

	if with_label_last:
		result = np.concatenate([result, np.expand_dims(label.astype(np.float32), axis = -1)], axis = -1)

	return result


def rbbox2roi(bbox_list):
	"""Convert a list of bboxes to roi format.

	Args:
		bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
			of images.

	Returns:
		Tensor: shape (n, 6), [batch_ind, x_ctr, y_ctr, w, h, theta]
	"""
	rois_list = []
	for img_id, bboxes in enumerate(bbox_list):
		if bboxes.size(0) > 0:
			img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
			rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
		else:
			rois = bboxes.new_zeros((0, 6))
		rois_list.append(rois)
	rois = torch.cat(rois_list, dim = 0)
	return rois
