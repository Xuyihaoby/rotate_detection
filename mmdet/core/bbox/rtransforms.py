import mmcv
import numpy as np
import cv2
import torch

def rbbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::5] = img_shape[1] - bboxes[..., 0::5]
    elif direction == 'vertical':
        flipped[..., 1::5] = img_shape[0] - bboxes[..., 1::5]
    flipped[:, 4::5] = -np.pi / 2 - bboxes[:, 4::5]
    flipped[:, 2::5] = bboxes[:, 3::5]
    flipped[:, 3::5] = bboxes[:, 2::5]
    return flipped

def rbbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = rbbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    scale_factor = torch.from_numpy(scale_factor).to(bboxes) # array ---> tensor
    scale_factor = torch.cat([scale_factor, scale_factor.new_ones(1)])
    _scale_factor = scale_factor.clone().to(new_bboxes)
    new_bboxes = new_bboxes.view(-1, 5) / _scale_factor
    return new_bboxes.view(bboxes.shape)

def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 6)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


########## refine rotate by rotate
def bbox2delta_rotate_from_rotate(proposals, gt,
                                  means=[0., 0., 0., 0., 0.],
                                  stds=[0.1, 0.1, 0.2, 0.2, 0.2], norm_degree_by90=False):
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
        ptheta = proposals[..., 4]  ## radian

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
        gtheta = gt[..., 4] * np.pi / 180.  ## degree --> radian

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    if norm_degree_by90:
        # dtheta = (gtheta - ptheta) / 90. ## degree difference
        raise NotImplementedError
    else:
        dtheta = gtheta - ptheta  ## radian difference
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
                                  norm_degree_by90=False):
    ## rois: [N, 5(x_ctr,y_ctr, w,h, theta)], theta is radian.
    ## deltas: [N, 5(dx,dy, dw,dh, dtheta)], theta is radian.
    ## returns: [N, 5(x_ctr,y_ctr, w,h, theta)], theta is radian.
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]  ## [N, 1]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    if norm_degree_by90:
        # dtheta = denorm_deltas[:, 4::5] * 90. ## degree
        raise NotImplementedError
    else:
        dtheta = denorm_deltas[:, 4::5]  ## radian

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = rois[:, 0].unsqueeze(1).expand_as(dx)  ## [N, 1]
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    # ptheta = rois.new_ones(rois.size()[0]).float().unsqueeze(1).expand_as(dtheta) * (-90.) ## degree
    ptheta = rois[:, 4].unsqueeze(1).expand_as(dtheta)  ## radian
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gtheta = ptheta + dtheta  ## radian
    # x1 = gx - gw * 0.5 + 0.5
    # y1 = gy - gh * 0.5 + 0.5
    # x2 = gx + gw * 0.5 - 0.5
    # y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)
        gw = gw.clamp(min=0, max=max_shape[1] - 1)
        gh = gh.clamp(min=0, max=max_shape[0] - 1)
        gtheta = gtheta.clamp(min=-90 * np.pi / 180., max=-0.00001 * np.pi / 180.)
    bboxes = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view_as(deltas)
    return bboxes


def transRotate2Quadrangle(coordinates, with_label_last=False):
    """
	Transform boxes from (x_ctr, y_ctr, w, h, theta(label)) to (x1,y1, x2,y2, x3,y3, x4,y4(label)).

	Arguments:
		coordinates: ndarray, [N, (x_ctr, y_ctr, w, h, theta)] 

	Returns:
		coordinates: ndarray, [N, (x1,y1, x2,y2, x3,y3, x4,y4)] 
	"""
    if with_label_last:
        tp = coordinates[:, :-1]  ## [N, 5]
        label = coordinates[:, -1]  ## [N]
    else:
        tp = coordinates  ## [N, 5]

    result = []
    for cd in tp:
        quad = cv2.boxPoints(((cd[0], cd[1]), (cd[2], cd[3]), cd[4]))
        result.append(np.reshape(quad, [-1, ]))
    result = np.array(result, dtype=np.float32)  ## [N, 8]

    if with_label_last:
        result = np.concatenate([result, np.expand_dims(label.astype(np.float32), axis=-1)], axis=-1)

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
    rois = torch.cat(rois_list, dim=0)
    return rois


def CV_L_Rad2LT_RB_TORCH(coordinates):
    assert coordinates.shape[-1] == 5
    devices = coordinates.device
    if coordinates.shape[0] == 0:
        return torch.zeros((0, 4), device=devices)
    _coor = coordinates.clone().cpu().numpy()
    _fourpoints = []
    for cd in _coor:
        quad = cv2.boxPoints(((cd[0], cd[1]), (cd[2], cd[3]), cd[4]*180/np.pi))
        _fourpoints.append(np.reshape(quad, [-1, ]))
    _result = np.array(_fourpoints, dtype=np.float32)
    xs = _result[:, 0::2]
    ys = _result[:, 1::2]
    x1 = np.min(xs, axis=-1).clip(min=0)
    y1 = np.min(ys, axis=-1).clip(min=0)
    x2 = np.max(xs, axis=-1).clip(min=0)
    y2 = np.max(ys, axis=-1).clip(min=0)
    _temp = [x1, y1, x2, y2]
    _twopoint = np.stack(_temp, axis=1)
    result = torch.from_numpy(_twopoint).to(devices)
    return result

