import mmcv
import numpy as np
import cv2
import torch
import math


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
    scale_factor = torch.from_numpy(scale_factor).to(bboxes)  # array ---> tensor
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
        quad = cv2.boxPoints(((cd[0], cd[1]), (cd[2], cd[3]), cd[4] * 180 / np.pi))
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
    assert result.requires_grad == coordinates.requires_grad
    return result


def CV_L_Rad2LE_DEF_TORCH(coordinates):
    assert coordinates.shape[-1] == 5
    new_coordinates = coordinates.clone()
    x, y, w, h, theta = coordinates.split((1, 1, 1, 1, 1), dim=-1)
    inds = (w > h) * (theta < 0)
    new_coordinates[inds.squeeze(1), 2], new_coordinates[inds.squeeze(1), 3] = new_coordinates[inds.squeeze(1), 3], new_coordinates[inds.squeeze(1), 2]
    new_coordinates[inds.squeeze(1), 4] = new_coordinates[inds.squeeze(1), 4] + np.pi / 2
    return new_coordinates


# modified from https://github.com/SJTU-Thinklab-Det/r3det-pytorch/blob/532b1b6a438175f5082af3e26f4db5743250b864
# /r3det/core/bbox/rtransforms.py#L190
def poly2obb_np(polys, version='v1'):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'v1':
        results = poly2obb_np_v1(polys)
    elif version == 'v2':
        results = poly2obb_np_v2(polys)
    elif version == 'v3':
        results = poly2obb_np_v3(polys)
    else:
        raise NotImplementedError
    return results

def poly2obb_np_v1(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    while not 0 > a >= -90:
        if a >= 0:
            a -= 90
            w, h = h, w
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 > a >= -np.pi / 2
    return x, y, w, h, a


def poly2obb_np_v2(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'v2')
    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np_v3(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return

    a = -a / 180 * np.pi
    if cv2.__version__ >='4.5.1':
        if w < h:
            w, h = h, w
            a += np.pi / 2
    elif cv2.__version__ <'4.5.1':
        if w < h:
            w, h = h, w
            a -= np.pi / 2

    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a

def norm_angle(angle, angle_range):
    """Limit the range of angles.
    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.
    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'v1':
        return angle
    elif angle_range == 'v2':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'v3':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')

def obb2poly_np(rbboxes, version='v1'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'v1':
        results = obb2poly_np_v1(rbboxes)
    elif version == 'v2':
        results = obb2poly_np_v2(rbboxes)
    elif version == 'v3':
        results = obb2poly_np_v3(rbboxes)
    else:
        raise NotImplementedError
    return results

def obb2poly_np_v1(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)


def obb2poly_np_v2(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_v3(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, -h / 2 * Cos], axis=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return np.concatenate([point1, point2, point3, point4, score], axis=-1)

def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).
    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))

def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]
    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))