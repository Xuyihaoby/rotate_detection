import torch


def ranchor_inside_flags(flat_ranchors,
                         valid_flags,
                         img_shape,
                         allowed_border=0):
    img_h, img_w = img_shape[:2]
    # cx, cy, w, h, a = (flat_ranchors[:, i] for i in range(5))
    # sina, cosa = torch.sin(a), torch.cos(a)
    # wx, wy = cosa * w / 2, sina * w / 2  # wx>0, wy<0
    # hx, hy = -sina * h / 2, cosa * h / 2  # hx>0, hy>0
    # lx = cx - wx - hx
    # ty = cy - wy + hy
    # rx = cx + wx + hx
    # by = cy + wy - hy
    # inside_flags = valid_flags & \
    #                (lx >= -allowed_border) & \
    #                (by >= -allowed_border) & \
    #                (rx < img_w + allowed_border) & \
    #                (ty < img_h + allowed_border)
    cx, cy = (flat_ranchors[:, i] for i in range(2))
    inside_flags = valid_flags & \
                   (cx >= -allowed_border) & \
                   (cy >= -allowed_border) & \
                   (cx < img_w + allowed_border) & \
                   (cy < img_h + allowed_border)

    return inside_flags
