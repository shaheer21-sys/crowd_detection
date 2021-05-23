from shapely.geometry import Polygon
from shapely.geometry import box


def calculate_iou(box1,box2):
    poly_1 = box(box1)
    poly_2 = box(box2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou



print(calculate_iou())
