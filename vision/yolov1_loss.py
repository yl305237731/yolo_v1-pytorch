import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloV1Loss(nn.Module):
    """

    """
    def __init__(self, B, S=7, target_size=(448, 448), lambda_coord=5, lambda_noobj=0.15, use_gpu=False):
        super(YoloV1Loss, self).__init__()
        self.B = B
        self.S = S
        self.target_size = target_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.use_gpu = use_gpu

    def forward(self, predicts, targets):
        """

        :param predicts: [[conf1, x, y, w_sqrt, h_sqrt, conf2, x, y, w_sqrt, h_sqrt, c1, c2 ...]
        :param targets: [[conf1, x, y, w_sqrt, h_sqrt, conf2, x, y, w_sqrt, h_sqrt, c1, c2 ...]
        :return:
        """
        box1 = predicts[..., :5]
        box2 = predicts[..., 5:10]
        # non_object confidence loss
        nobj = 1 - targets[..., 0]

        nobj_box1_loss = self.confidence_loss(box1[..., 0] * nobj, targets[..., 0] * nobj)
        nobj_box2_loss = self.confidence_loss(box2[..., 0] * nobj, targets[..., 0] * nobj)
        nobj_conf_loss = nobj_box1_loss.sum() + nobj_box2_loss.sum()

        # contain object loss
        response_mask = self.response_box(predicts, targets)

        # 1.confidence loss and coord loss of response grid

        box1_coord_loss = self.coord_loss(box1, targets) * targets[..., 0] * response_mask[..., 0]
        box2_coord_loss = self.coord_loss(box2, targets) * targets[..., 0] * response_mask[..., 1]
        boxes_coord_loss = box1_coord_loss.sum() + box2_coord_loss.sum()

        box1_response_conf_loss = self.confidence_loss(box1[..., 0] * response_mask[..., 0] * targets[..., 0], targets[..., 0] * response_mask[...,0])
        box2_response_conf_loss = self.confidence_loss(box2[..., 0] * response_mask[..., 1] * targets[..., 0], targets[..., 0] * response_mask[..., 1])
        response_conf_loss_obj = box1_response_conf_loss.sum() + box2_response_conf_loss.sum()

        # 2. only confidence loss of not response grid
        box1_not_response_conf_loss = self.confidence_loss(box1[..., 0] * response_mask[..., 1] * targets[..., 0], targets[..., 0] * response_mask[..., 1])
        box2_not_response_conf_loss = self.confidence_loss(box2[..., 0] * response_mask[..., 0] * targets[..., 0], targets[..., 0] * response_mask[..., 0])
        not_response_conf_loss_obj = box1_not_response_conf_loss.sum() + box2_not_response_conf_loss.sum()

        obj_conf_loss = response_conf_loss_obj + not_response_conf_loss_obj

        # 3.classification loss
        class_loss = ((predicts[..., self.B * 5:] - targets[..., self.B * 5:]) ** 2 * torch.unsqueeze(targets[..., 0], dim=3)).sum()

        print("coord loss: {}, conf_loss_obj: {}, conf_loss_nobj: {}, class_loss: {}".format(self.lambda_coord * boxes_coord_loss, obj_conf_loss, nobj_conf_loss * self.lambda_noobj, class_loss))

        total_loss = self.lambda_coord * boxes_coord_loss + obj_conf_loss + nobj_conf_loss * self.lambda_noobj + class_loss
        return total_loss

    def coord_loss(self, box, target):
        # x_loss = (box[..., 1] - target[..., 1]) ** 2
        # y_loss = (box[..., 2] - target[..., 2]) ** 2
        # w_loss = (box[..., 3] - target[..., 3]) ** 2
        # h_loss = (box[..., 4] - target[..., 4]) ** 2
        x_loss = F.smooth_l1_loss(box[..., 1], target[..., 1], reduction="none")
        y_loss = F.smooth_l1_loss(box[..., 2], target[..., 2], reduction="none")
        w_loss = F.smooth_l1_loss(box[..., 3], target[..., 3], reduction="none")
        h_loss = F.smooth_l1_loss(box[..., 4], target[..., 4], reduction="none")
        return x_loss + y_loss + w_loss + h_loss

    def confidence_loss(self, predict, target):
        # return self.focal_loss(predict, target)
        return (predict - target) ** 2

    def response_box(self, predicts, targets):
        """
        choose which box to calculate loss
        :param predicts:
        :param targets:
        :return:
        """
        response_mask = torch.zeros([predicts.size(0), predicts.size(1), predicts.size(1), self.B])
        if self.use_gpu:
            response_mask = response_mask.cuda()
        # predictons box
        box1_xy = self.offset2absolute(predicts[..., 1:3])
        box1_wh = self.norm_wh2absolute(predicts[..., 3:5])
        b1x1y1 = box1_xy - box1_wh / 2
        b1x2y2 = box1_xy + box1_wh / 2
        box2_xy = self.offset2absolute(predicts[..., 6:8])
        box2_wh = self.norm_wh2absolute(predicts[..., 8:10])
        b2x1y1 = box2_xy - box2_wh / 2
        b2x2y2 = box2_xy + box2_wh / 2
        boxes1 = torch.cat([b1x1y1, b1x2y2], dim=3)
        boxes2 = torch.cat([b2x1y1, b2x2y2], dim=3)
        # true box, the two true boxes are same
        t_box_xy = self.offset2absolute(targets[..., 1:3])
        t_box_wh = self.norm_wh2absolute(targets[..., 3:5])
        t_b1x1y1 = t_box_xy - t_box_wh / 2
        t_b1x2y2 = t_box_xy + t_box_wh / 2
        t_coords = torch.cat([t_b1x1y1, t_b1x2y2], dim=3)

        box1_iou = self.compute_iou(boxes1, t_coords)
        box2_iou = self.compute_iou(boxes2, t_coords)
        response_mask[..., 0] = torch.ge(box1_iou, box2_iou).type(torch.float)
        response_mask[..., 1] = torch.le(box1_iou, box2_iou).type(torch.float)
        return response_mask

    def offset2absolute(self, x):
        """
        offset value of grid to absolute value in image
        :param x:
        :return:
        """
        absol = torch.zeros_like(x)
        if self.use_gpu:
            absol = absol.cuda()
        for ii in range(self.S):
            for jj in range(self.S):
                absol[:, ii, jj, 0] = (x[:, ii, jj, 0] + jj) / self.S * self.target_size[0]
                absol[:, ii, jj, 1] = (x[:, ii, jj, 1] + ii) / self.S * self.target_size[1]
        return absol

    def norm_wh2absolute(self, x):
        """
        normalized  width and height to absolute value of image
        :param x:
        :return:
        """
        return torch.cat([torch.unsqueeze(x[..., 0] ** 2 * self.target_size[0], 3), torch.unsqueeze(x[..., 1] ** 2 *
                                                                                                    self.target_size[1], dim=3)], dim=3)

    def compute_iou(self, boxes1, boxes2):
        """
        :param box1: [N, S, S, 4]
        :param box2: [N, S, S, 4]
        :return: iou [N, S, S, 1]
        """

        tl = torch.max(boxes1[..., :2], boxes2[..., :2])
        br = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = br - tl
        wh = torch.max(wh, torch.zeros_like(wh))
        inter = wh[..., 0] * wh[..., 1]
        area_1 = (boxes1[..., 2] - boxes1[..., 0] + 1) * (boxes1[..., 3] - boxes1[..., 1] + 1)
        area_2 = (boxes2[..., 2] - boxes2[..., 0] + 1) * (boxes2[..., 3] - boxes2[..., 1] + 1)

        return inter / (area_1 + area_2 - inter)

    def focal_loss(self, predict, target, alpha=1, gamma=2):
        bce_loss = F.binary_cross_entropy(predict, target, reduction='none')
        pt = torch.exp(-bce_loss)
        return alpha * (1 - pt) ** gamma * bce_loss
