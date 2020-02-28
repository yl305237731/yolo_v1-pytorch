import numpy as np
import cv2
import torchvision.transforms as transforms
import torch


def adjust_learning_rate(initial_lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_class_index(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_class_index()]

        return self.score


def decode_netout(netout, confidence_thresh=0.8, net_h=448, net_w=448, nb_box=2):
    grid_h, grid_w = netout.shape[:2]
    nb_box = nb_box
    boxes = []

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w
        if netout[row, col, 0] >= netout[row, col, 5]:
            box = netout[row, col, :5]
        else:
            box = netout[row, col, 5:10]
        if box[0] < confidence_thresh:
            continue
        x, y, w, h = box[1], box[2], box[3], box[4]
        x = (x + col) / grid_w * net_w
        y = (y + row) / grid_h * net_h
        w = w ** 2 * net_w
        h = h ** 2 * net_h
        classes = netout[row, col, nb_box * 5:]
        box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, box[0], classes)
        boxes.append(box)

    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):

    for i in range(len(boxes)):

        boxes[i].xmin = int(boxes[i].xmin * image_w / net_w)
        boxes[i].xmax = int(boxes[i].xmax * image_w / net_w)
        boxes[i].ymin = int(boxes[i].ymin * image_h / net_h)
        boxes[i].ymax = int(boxes[i].ymax * image_h / net_h)


def iou(box_a, box_b):
    """
    :param box_a: [x1, y1, x2, y2]
    :param box_b: [x1, y1, x2, y2]
    :return: iou
    """
    area_boxa = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    area_boxb = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)

    def intersection(box1, box2):
        x_lt = max(box1.xmin, box2.xmin)
        y_lt = max(box1.ymin, box2.ymin)
        x_br = min(box1.xmax, box2.xmax)
        y_br = min(box1.ymax, box2.ymax)
        inter_w = max(x_br - x_lt, 0)
        inter_h = max(y_br - y_lt, 0)
        return float(inter_w * inter_h)
    area_inter = intersection(box_a, box_b)
    return area_inter / (area_boxa + area_boxb - area_inter)


def do_nms(boxes, nms_thresh=0.4):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def preprocess_input(image, net_w, net_h):
    resized = cv2.resize(image, (net_w, net_h))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return torch.unsqueeze(transform(resized), dim=0)


def draw_boxes(image, boxes, labels, obj_thresh=0.8):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '':
                    label_str += ', '
                label_str += (labels[i] + ' ' + str(box.get_score().numpy()))
                label = i
        if label >= 0:
            cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=(255, 0, 255),
                          thickness=5)
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin + 13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0, 0, 0),
                        thickness=2)
    return image

