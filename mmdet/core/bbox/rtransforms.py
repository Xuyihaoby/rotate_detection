import mmcv
import numpy as np
import cv2
import torch
import math
from mmdet.ops.batch_svd import svd


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
    new_coordinates[inds.squeeze(1), 2], new_coordinates[inds.squeeze(1), 3] = new_coordinates[inds.squeeze(1), 3], \
                                                                               new_coordinates[inds.squeeze(1), 2]
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
    if cv2.__version__ >= '4.5.1':
        if w < h:
            w, h = h, w
            a += np.pi / 2
    elif cv2.__version__ < '4.5.1':
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


def obb2poly(rbboxes, version='v1'):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'v1':
        results = obb2poly_v1(rbboxes)
    elif version == 'v2':
        results = obb2poly_v2(rbboxes)
    elif version == 'v3':
        results = obb2poly_v3(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_v1(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_v2(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_v3(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    # M.shape=[N,2,2]
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    # polys:[N,8]
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def poly2obb(polys, version='v1'):
    """Convert polygons to oriented bounding boxes.

    (r3det version)
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'v1':
        results = poly2obb_v1(polys)
    elif version == 'v2':
        results = poly2obb_v2(polys)
    elif version == 'v3':
        results = poly2obb_v3(polys)
    else:
        raise NotImplementedError
    return results


def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).
    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)


def poly2obb_v1(polys):
    """Convert polygons to oriented bounding boxes.

    (r3det version)
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (-np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, -np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes


def poly2obb_v2(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'v2')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_v3(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'v3')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


# https://github.com/lilanxiao/Rotated_IoU
def enclosing_box(corners1: torch.Tensor, corners2: torch.Tensor, enclosing_type: str = "smallest"):
    if enclosing_type == "aligned":
        return enclosing_box_aligned(corners1, corners2)
    elif enclosing_type == "pca":
        return enclosing_box_pca(corners1, corners2)
    elif enclosing_type == "smallest":
        return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    else:
        ValueError("Unknow type enclosing. Supported: aligned, pca, smallest")


def enclosing_box_aligned(corners1: torch.Tensor, corners2: torch.Tensor):
    """calculate the smallest enclosing box (axis-aligned)
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    x1_max = torch.max(corners1[..., 0], dim=2)[0]  # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]  # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]

    x2_max = torch.max(corners2[..., 0], dim=2)[0]  # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]  # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    w = x_max - x_min  # (B, N)
    h = y_max - y_min
    return w, h


def enclosing_box_pca(corners1: torch.Tensor, corners2: torch.Tensor):
    """calculate the rotated smallest enclosing box using PCA
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    B = corners1.size()[0]
    c = torch.cat([corners1, corners2], dim=2)  # (B, N, 8, 2)
    c = c - torch.mean(c, dim=2, keepdim=True)  # normalization
    c = c.view([-1, 8, 2])  # (B*N, 8, 2)
    ct = c.transpose(1, 2)  # (B*N, 2, 8)
    ctc = torch.bmm(ct, c)  # (B*N, 2, 2)
    # NOTE: the build in symeig is slow!
    # _, v = ctc.symeig(eigenvectors=True)
    # v1 = v[:, 0, :].unsqueeze(1)
    # v2 = v[:, 1, :].unsqueeze(1)
    v1, v2 = eigenvector_22(ctc)
    v1 = v1.unsqueeze(1)  # (B*N, 1, 2), eigen value
    v2 = v2.unsqueeze(1)
    p1 = torch.sum(c * v1, dim=-1)  # (B*N, 8), first principle component
    p2 = torch.sum(c * v2, dim=-1)  # (B*N, 8), second principle component
    w = p1.max(dim=-1)[0] - p1.min(dim=-1)[0]  # (B*N, ),  width of rotated enclosing box
    h = p2.max(dim=-1)[0] - p2.min(dim=-1)[0]  # (B*N, ),  height of rotated enclosing box
    return w.view([B, -1]), h.view([B, -1])


def eigenvector_22(x: torch.Tensor):
    """return eigenvector of 2x2 symmetric matrix using closed form

    https://math.stackexchange.com/questions/8672/eigenvalues-and-eigenvectors-of-2-times-2-matrix

    The calculation is done by using double precision
    Args:
        x (torch.Tensor): (..., 2, 2), symmetric, semi-definite

    Return:
        v1 (torch.Tensor): (..., 2)
        v2 (torch.Tensor): (..., 2)
    """
    # NOTE: must use doule precision here! with float the back-prop is very unstable
    a = x[..., 0, 0].double()
    c = x[..., 0, 1].double()
    b = x[..., 1, 1].double()  # (..., )
    delta = torch.sqrt(a * a + 4 * c * c - 2 * a * b + b * b)
    v1 = (a - b - delta) / 2. / c
    v1 = torch.stack([v1, torch.ones_like(v1, dtype=torch.double, device=v1.device)], dim=-1)  # (..., 2)
    v2 = (a - b + delta) / 2. / c
    v2 = torch.stack([v2, torch.ones_like(v2, dtype=torch.double, device=v2.device)], dim=-1)  # (..., 2)
    n1 = torch.sum(v1 * v1, keepdim=True, dim=-1).sqrt()
    n2 = torch.sum(v2 * v2, keepdim=True, dim=-1).sqrt()
    v1 = v1 / n1
    v2 = v2 / n2
    return v1.float(), v2.float()


def smallest_bounding_box(corners: torch.Tensor, verbose=False):
    """return width and length of the smallest bouding box which encloses two boxes.
    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        verbose (bool, optional): If True, return area and index. Defaults to False.
    Returns:
        (torch.Tensor): width (..., 24)
        (torch.Tensor): height (..., 24)
        (torch.Tensor): area (..., )
        (torch.Tensor): index of candiatae (..., )
    """
    lines, points, _, _ = gather_lines_points(corners)
    proj = point_line_projection_range(lines, points)  # (..., 24)
    dist = point_line_distance_range(lines, points)  # (..., 24)
    area = proj * dist
    # remove area with 0 when the two points of the line have the same coordinates
    zero_mask = (area == 0).type(corners.dtype)
    fake = torch.ones_like(zero_mask, dtype=corners.dtype, device=corners.device) * 1e8 * zero_mask
    area += fake  # add large value to zero_mask
    area_min, idx = torch.min(area, dim=-1, keepdim=True)  # (..., 1)
    w = torch.gather(proj, dim=-1, index=idx)
    h = torch.gather(dist, dim=-1, index=idx)  # (..., 1)
    w = w.squeeze(-1).float()
    h = h.squeeze(-1).float()
    area_min = area_min.squeeze(-1).float()
    if verbose:
        return w, h, area_min, idx.squeeze(-1)
    else:
        return w, h


'''
find the smallest bounding box which enclosing two rectangles. It can be used to calculate the GIoU or DIoU
loss for rotated object detection.
Observation: a side of a minimum-area enclosing box must be collinear with a side of the convex polygon.
https://en.wikipedia.org/wiki/Minimum_bounding_box_algorithms
Since two rectangles have 8 points, brutal force method should be enough. That is, calculate the enclosing box
area for every possible side of the polygon and take the mininum. Their should be 8x7/2 = 28 combinations and 4
of them are impossible (4 diagonal of two boxes). So the function brutally searches in the 24 candidates.
The index of box corners follows the following convention:
  0---1        4---5
  |   |        |   |
  3---2        7---6
author: Lanxiao Li
2020.08
'''


def generate_table():
    """generate candidates of hull polygon edges and the the other 6 points
    Returns:
        lines: (24, 2)
        points: (24, 6)
    """
    skip = [[0, 2], [1, 3], [5, 7], [4, 6]]  # impossible hull edge
    line = []
    points = []

    def all_except_two(o1, o2):
        a = []
        for i in range(8):
            if i != o1 and i != o2:
                a.append(i)
        return a

    for i in range(8):
        for j in range(i + 1, 8):
            if [i, j] not in skip:
                line.append([i, j])
                points.append(all_except_two(i, j))
    return line, points


LINES, POINTS = generate_table()
LINES = np.array(LINES).astype(np.int)
POINTS = np.array(POINTS).astype(np.int)


def gather_lines_points(corners: torch.Tensor):
    """get hull edge candidates and the rest points using the index
    Args:
        corners (torch.Tensor): (..., 8, 2)

    Return:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)
        idx_lines (torch.Tensor): Long (..., 24, 2, 2)
        idx_points (torch.Tensor): Long (..., 24, 6, 2)
    """
    dim = corners.dim()
    idx_lines = torch.LongTensor(LINES).to(corners.device).unsqueeze(-1)  # (24, 2, 1)
    idx_points = torch.LongTensor(POINTS).to(corners.device).unsqueeze(-1)  # (24, 6, 1)
    idx_lines = idx_lines.repeat(1, 1, 2)  # (24, 2, 2)
    idx_points = idx_points.repeat(1, 1, 2)  # (24, 6, 2)
    if dim > 2:
        for _ in range(dim - 2):
            idx_lines = torch.unsqueeze(idx_lines, 0)
            idx_points = torch.unsqueeze(idx_points, 0)
        idx_points = idx_points.repeat(*corners.size()[:-2], 1, 1, 1)  # (..., 24, 2, 2)
        idx_lines = idx_lines.repeat(*corners.size()[:-2], 1, 1, 1)  # (..., 24, 6, 2)
    corners_ext = corners.unsqueeze(-3).repeat(*([1] * (dim - 2)), 24, 1, 1)  # (..., 24, 8, 2)
    lines = torch.gather(corners_ext, dim=-2, index=idx_lines)  # (..., 24, 2, 2)
    points = torch.gather(corners_ext, dim=-2, index=idx_points)  # (..., 24, 6, 2)

    return lines, points, idx_lines, idx_points


def point_line_distance_range(lines: torch.Tensor, points: torch.Tensor):
    """calculate the maximal distance between the points in the direction perpendicular to the line
    methode: point-line-distance
    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)

    Return:
        torch.Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]  # (..., 24, 1)
    y1 = lines[..., 0:1, 1]  # (..., 24, 1)
    x2 = lines[..., 1:2, 0]  # (..., 24, 1)
    y2 = lines[..., 1:2, 1]  # (..., 24, 1)
    x = points[..., 0]  # (..., 24, 6)
    y = points[..., 1]  # (..., 24, 6)
    den = (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1
    # NOTE: the backward pass of torch.sqrt(x) generates NaN if x==0
    num = torch.sqrt((y2 - y1).square() + (x2 - x1).square() + 1e-14)
    d = den / num  # (..., 24, 6)
    d_max = d.max(dim=-1)[0]  # (..., 24)
    d_min = d.min(dim=-1)[0]  # (..., 24)
    d1 = d_max - d_min  # suppose points on different side
    d2 = torch.max(d.abs(), dim=-1)[0]  # or, all points are on the same side
    # NOTE: if x1 = x2 and y1 = y2, this will return 0
    return torch.max(d1, d2)


def point_line_projection_range(lines: torch.Tensor, points: torch.Tensor):
    """calculate the maximal distance between the points in the direction parallel to the line
    methode: point-line projection
    Args:
        lines (torch.Tensor): (..., 24, 2, 2)
        points (torch.Tensor): (..., 24, 6, 2)

    Return:
        torch.Tensor: (..., 24)
    """
    x1 = lines[..., 0:1, 0]  # (..., 24, 1)
    y1 = lines[..., 0:1, 1]  # (..., 24, 1)
    x2 = lines[..., 1:2, 0]  # (..., 24, 1)
    y2 = lines[..., 1:2, 1]  # (..., 24, 1)
    k = (y2 - y1) / (x2 - x1 + 1e-8)  # (..., 24, 1)
    vec = torch.cat([torch.ones_like(k, dtype=k.dtype, device=k.device), k], dim=-1)  # (..., 24, 2)
    vec = vec.unsqueeze(-2)  # (..., 24, 1, 2)
    points_ext = torch.cat([lines, points], dim=-2)  # (..., 24, 8), consider all 8 points
    den = torch.sum(points_ext * vec, dim=-1)  # (..., 24, 8)
    proj = den / torch.norm(vec, dim=-1, keepdim=False)  # (..., 24, 8)
    proj_max = proj.max(dim=-1)[0]  # (..., 24)
    proj_min = proj.min(dim=-1)[0]  # (..., 24)
    return proj_max - proj_min


def distance2rbbox(points, distance, version='v1'):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 5
            boundaries (left, top, right, bottom, radian).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    distance, theta = distance.split([4, 1], dim=1)
    wh = distance[:, :2] + distance[:, 2:]


    if version == 'v1':
        theta = torch.where(theta >= 0, theta-torch.tensor(np.pi/2), theta)
        theta = torch.where(theta < torch.tensor(-np.pi/2), theta+torch.tensor(np.pi/2), theta)
        Cos, Sin = torch.cos(theta), torch.sin(theta)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=1).reshape(-1, 2, 2)
    elif version == 'v2':
        Cos, Sin = torch.cos(theta), torch.sin(theta)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=1).reshape(-1, 2, 2)
    else:
        Cos, Sin = torch.cos(theta), torch.sin(theta)
        Matrix = torch.cat([Cos, Sin, -Sin, Cos], dim=1).reshape(-1, 2, 2)
    offset_t = (distance[:, 2:] - distance[:, :2]) / 2
    offset_t = offset_t.unsqueeze(2)

    offset = torch.bmm(Matrix, offset_t).squeeze(2)
    ctr = points + offset
    return torch.cat([ctr, wh, theta], dim=1)

def gaussian2bbox(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    L = 3
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)
    assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    U, s, Vt = svd(var)
    # bacth_svd U与V其实均没有转置
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    mu = mu.repeat(1, 4, 1)
    dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                     dtype=torch.float32,
                                     device=size_half.device)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes

def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 4, 2).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))

