from . import rbbox_geo_cuda
# import rbbox_geo_cuda

def rbbox_iou_iof(rb1, rb2, vec=False, iof=False):
    if vec:
        return rbbox_geo_cuda.vec_iou_iof(rb1, rb2, iof)
    else:
        return rbbox_geo_cuda.mat_iou_iof(rb1, rb2, iof)
