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


def polygon_nms(polygons, scores, iou_threshold=0.5):
    """
    Apply nms to polygons, returns flags, which polygons to leave
    ref: https://github.com/MhLiao/RRD/blob/master/examples/text/nms.py
    """

    indices = sorted(range(len(scores)), key=lambda k: -scores[k])
    box_num = len(polygons)
    nms_flag = np.asarray([True] * box_num)

    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue

        for j in range(box_num):
            jj = indices[j]

            if j == i or not nms_flag[jj]:
                continue

            polygon1, polygon2 = polygons[ii], polygons[jj]
            score1, score2 = scores[ii], scores[jj]

            iou = polygon_iou(polygon1, polygon2)

            if iou > iou_threshold:
                if score1 > score2:
                    nms_flag[jj] = False
                if score1 == score2 and polygon1.area > polygon2.area:
                    nms_flag[jj] = False
                if score1 == score2 and polygon1.area <= polygon2.area:
                    nms_flag[ii] = False
                    break

    return nms_flag
