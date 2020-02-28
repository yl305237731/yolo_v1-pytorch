import numpy as np


def softnms(dests, obj_thresh, sigma=0.5, top_k =-1, eps=1e-5):
    """
    :param dests: box array  [[x1, y1, x2, y2, score],[x1, y1, x2, y2, score]...]
    :param obj_thresh: object threshold
    :param sigma: the parameter in score re-computation.
                scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
    :param eps: a small number to avoid 0 as denominator.
    :return: picked_box_scores (K, 5): results of NMS.
    """
    x1 = dests[:, 0]
    y1 = dests[:, 1]
    x2 = dests[:, 2]
    y2 = dests[:, 3]
    scores = dests[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep_box = []

    while order.size > 0:
        cur_ind = order[0]
        keep_box.append(dests[cur_ind, :])
        if len(keep_box) == top_k > 0 or dests.shape[0] == 1:
            break

        xlt = np.maximum(x1[cur_ind], x1[order[1:]])
        ylt = np.maximum(y1[cur_ind], y1[order[1:]])
        xrd = np.minimum(x2[cur_ind], x2[order[1:]])
        yrd = np.minimum(y2[cur_ind], y2[order[1:]])

        inter_w = np.maximum(xrd - xlt + 1, 0.0)
        inter_h = np.maximum(yrd - ylt + 1, 0.0)
        inter = inter_h * inter_w
        iou_area = inter / (areas[cur_ind] + areas[order[1:]] - inter + eps)
        #   scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        dests[order[1:], -1] = dests[order[1:], -1] * np.exp(-(iou_area * iou_area) / sigma)
        #   去掉小于 obj_thresh 的 box
        ind = np.where(dests[order[1:], -1] >= obj_thresh)[0]
        order = order[ind + 1]
    return keep_box


def iou(box_a, box_b):
    """
    :param box_a: [x1, y1, x2, y2]
    :param box_b: [x1, y1, x2, y2]
    :return: iou
    """
    area_boxa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_boxb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    def intersection(box1, box2):
        x_lt = max(box1[0], box2[0])
        y_lt = max(box1[1], box2[1])
        x_br = min(box1[2], box2[2])
        y_br = min(box1[3], box2[3])
        inter_w = max(x_br - x_lt, 0)
        inter_h = max(y_br - y_lt, 0)
        return float(inter_w * inter_h)
    area_inter = intersection(box_a, box_b)
    return area_inter / (area_boxa + area_boxb - area_inter)


def nms(dests, thresh=0.7, top_k=-1):
    """
    :param dests: box array  [[x1, y1, x2, y2, score],[x1, y1, x2, y2, score]...]
    :param thresh: nms ignore threshold
    :return: picked_box_scores (K, 5): results of NMS.
    """
    x1 = dests[:, 0]
    y1 = dests[:, 1]
    x2 = dests[:, 2]
    y2 = dests[:, 3]
    scores = dests[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep_box = []

    while order.size > 0:
        cur_ind = order[0]
        keep_box.append(dests[cur_ind, :])
        if len(keep_box) == top_k > 0 or dests.shape[0] == 1:
            break

        xlt = np.maximum(x1[cur_ind], x1[order[1:]])
        ylt = np.maximum(y1[cur_ind], y1[order[1:]])
        xrd = np.minimum(x2[cur_ind], x2[order[1:]])
        yrd = np.minimum(y2[cur_ind], y2[order[1:]])

        inter_w = np.maximum(xrd - xlt + 1, 0.0)
        inter_h = np.maximum(yrd - ylt + 1, 0.0)
        inter = inter_h * inter_w
        iou_area = inter / (areas[cur_ind] + areas[order[1:]] - inter)
        # 筛选出 IOU 小于阈值的 box 在 iou_area 中的index, 大于阈值的box作为冗余被去除
        ind = np.where(iou_area <= thresh)[0]
        # 将筛选出的 box 的 index 重新构成 order, iou_area 中 ind 在 order中对应为 ind + 1
        order = order[ind + 1]
    return keep_box
