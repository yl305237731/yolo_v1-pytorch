import random
import cv2
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure


class DataAugmentation:

    def __init__(self, rotation_rate=0.5, max_rotation_angle=15, crop_rate=0.3, shift_rate=0.3, change_light_rate=0.3,
                 add_noise_rate=0.3):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate

    def noise(self, img):
        return random_noise(img, mode='gaussian', clip=True) * 255

    def light(self, img):
        flag = random.uniform(0.5, 1.5)
        return exposure.adjust_gamma(img, flag)

    def rotate(self, img, bboxs, angle=5, scale=1.):
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]

        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        rot_bboxes = list()
        for bbox in bboxs:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))

            concat = np.vstack((point1, point2, point3, point4))

            concat = concat.astype(np.int32)

            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh

            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    def crop(self, img, bboxs):
        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxs:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        crop_x_min = int(x_min - random.uniform(d_to_left // 2, d_to_left))
        crop_y_min = int(y_min - random.uniform(d_to_top // 2, d_to_top))
        crop_x_max = int(x_max + random.uniform(d_to_right // 2, d_to_right))
        crop_y_max = int(y_max + random.uniform(d_to_bottom // 2, d_to_bottom))

        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        crop_bboxes = list()
        for bbox in bboxs:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    def shift(self, img, bboxs):
        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxs:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        shift_bboxes = list()
        for bbox in bboxs:
            shift_bboxes.append([int(bbox[0] + x), int(bbox[1] + y), int(bbox[2] + x), int(bbox[3] + y)])

        return shift_img, shift_bboxes

    def augment(self, img, bboxs, crop=True, rotate=True, shift=True, light=True, noise=True):
        if crop and random.random() < self.crop_rate:
            img, bboxs = self.crop(img, bboxs)

        if rotate and random.random() < self.rotation_rate:
            angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            scale = random.uniform(0.7, 0.8)
            img, bboxs = self.rotate(img, bboxs, angle, scale)

        if shift and random.random() < self.shift_rate:
            img, bboxs = self.shift(img, bboxs)

        if light and random.random() < self.change_light_rate:
            img = self.light(img)

        if noise and random.random() < self.add_noise_rate:
            img = self.noise(img)

        return img, bboxs