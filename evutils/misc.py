# -*- coding: utf-8 -*-
import os
import math
import json
import random
import mmcv
import torch
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


def img_slide_window(cv2_img, cell_size=[512, 512]):
    H, W, C = cv2_img.shape
    cell_w, cell_h = cell_size
    row = math.ceil(H / cell_h)
    col = math.ceil(W / cell_w)

    h_new = row * cell_h
    w_new = col * cell_w

    if h_new != H or w_new != W:
        if len(cv2_img.shape) == 2:
            tmp = np.zeros((h_new, w_new))
            tmp[:cv2_img.shape[0], :cv2_img.shape[1]] = cv2_img
        else:
            tmp = np.zeros((h_new, w_new, C))
            tmp[:cv2_img.shape[0], :cv2_img.shape[1], :] = cv2_img
        H, W = tmp.shape[:2]
    else:
        tmp = cv2_img
    
    index = []
    result = []
    for idr in range(row):
        for idc in range(col):
            if len(tmp.shape) == 2:
                result.append(tmp[(idr * cell_h):((idr + 1) * cell_h), \
                                (idc * cell_w):((idc + 1) * cell_w)])
            else:
                result.append(tmp[(idr * cell_h):((idr + 1) * cell_h), \
                                (idc * cell_w):((idc + 1) * cell_w), :])
            index.append([idr, idc])
    return index, result


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def setup_seed(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False
