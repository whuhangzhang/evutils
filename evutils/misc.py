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


def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False


def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def url_to_torch(url, size=(384, 384)):
    img = get_image_from_url(url)
    img = img.resize(size, Image.ANTIALIAS)
    img = torch.from_numpy(np.asarray(img)).float()
    img = img.permute(2, 0, 1)
    img.div_(255)
    return img


def pil_to_batched_tensor(img):
    return ToTensor()(img).unsqueeze(0)


def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)
