from . import polygon_geo_cpu


def polygon_iou(poly1, poly2):
    return polygon_geo_cpu.polygon_iou(poly1, poly2)
