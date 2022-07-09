import mmcv
import numpy as np
import shapely
from shapely.geometry import Polygon


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    assert tensor.dim() == 4
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    try:
        if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
            iou = 0
        else:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occurred, iou set to 0')
        iou = 0
    return iou