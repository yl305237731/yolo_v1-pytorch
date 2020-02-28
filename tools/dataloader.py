import torch
import cv2
import os
import xml.etree.ElementTree as ET
import random
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, img_dir, xml_dir, target_size, S, B, name_list, shuffle=True, augmentation=None, transform=None):
        """
        :param img_dir: images root dir
        :param xml_dir: xml file root dir
        :param target_size: target image size: tuple(w, h)
        :param S: grid number S * S
        :param B: bounding box number
        :param name_list: like ['cat', 'dog'...]
        :param shuffle:
        :param augmentation: data augmentation
        :param transform:
        """
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.target_size = target_size
        self.S = S
        self.B = B
        self.name_list = name_list
        self.class_num = len(self.name_list)
        self.transform = transform
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.img_names, self.img_bboxs, self.img_clas = self.parse_xml()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_bbox = self.img_bboxs[idx]
        img_cla = self.img_clas[idx]
        img_ori = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            img, img_bbox = self.augmentation.augment(img, img_bbox, noise=False)
        height, width, _ = img.shape
        img = cv2.resize(img, self.target_size)
        img_bbox = torch.floor(torch.Tensor(img_bbox) * torch.Tensor([self.target_size[0] / width, self.target_size[1] / height,
                                                                      self.target_size[0] / width, self.target_size[1] / height]))
        img_cla = self.name2label(img_cla)
        if self.transform:
            img = self.transform(img)

        target = self.encode_target(img_bbox, img_cla)
        return img, target

    def encode_target(self, bboxs, img_clas):
        """
        :param bboxs: [[x1, y1, x2, y2]...]
        :param img_clas: [[0]...]
        :return: S * S * （B * 5 + class_num）
        """
        target = torch.zeros((self.S, self.S, self.B * 5 + self.class_num))
        n_box = len(bboxs)
        clas = torch.zeros((n_box, self.class_num))
        for i in range(n_box):
            clas[i, img_clas[i]] = 1
        # to [xc, yc , w, h]
        bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
        bboxs[:, 0] = bboxs[:, 0] + torch.floor(bboxs[:, 2] / 2)
        bboxs[:, 1] = bboxs[:, 1] + torch.floor(bboxs[:, 3] / 2)

        # x = xc / w * s - col
        # y = yc / h * s - row
        col = torch.floor(bboxs[:, 0] / (self.target_size[0] / self.S))
        row = torch.floor(bboxs[:, 1] / (self.target_size[1] / self.S))
        # offset value of current grid, not absolute value
        x = bboxs[:, 0] / self.target_size[0] * self.S - col
        y = bboxs[:, 1] / self.target_size[1] * self.S - row

        w_sqrt = torch.sqrt(bboxs[:, 2] / self.target_size[0])
        h_sqrt = torch.sqrt(bboxs[:, 3] / self.target_size[1])

        conf = torch.ones_like(col)
        # [[conf1, x, y, w_sqrt, h_sqrt, conf2, x, y, w_sqrt, h_sqrt, c1, c2 ...],
        # [....]]
        grid_info = torch.cat([conf.view(-1, 1), x.view(-1, 1), y.view(-1, 1), w_sqrt.view(-1, 1), h_sqrt.view(-1, 1)],
                              dim=1).repeat(1, self.B)
        grid_info = torch.cat([grid_info, clas], dim=1)

        for i in range(n_box):
            row_index = row[i].numpy()
            col_index = col[i].numpy()
            target[row_index, col_index] = grid_info[i].clone()

        return target

    def name2label(self, cla_name):
        clas_index = []
        for i, name in enumerate(cla_name):
            clas_index.append(self.name_list.index(name))
        return clas_index

    def parse_xml(self):
        img_names = []
        img_bboxs = []
        img_clas = []
        xml_dir = os.listdir(self.xml_dir)

        if self.shuffle:
            random.shuffle(xml_dir)

        for xml_name in xml_dir:
            xml_path = os.path.join(self.xml_dir, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_name = tree.find('filename').text
            if not os.path.exists(os.path.join(self.img_dir, img_name)):
                continue
            img_names.append(img_name)
            objs = root.findall('object')
            coords = list()
            clas = list()
            for ix, obj in enumerate(objs):
                name = obj.find('name').text
                if name in self.name_list:
                    box = obj.find('bndbox')
                    x_min = int(box[0].text)
                    y_min = int(box[1].text)
                    x_max = int(box[2].text)
                    y_max = int(box[3].text)
                    coords.append([x_min, y_min, x_max, y_max])
                    clas.append(name)
            img_bboxs.append(coords)
            img_clas.append(clas)
        return img_names, img_bboxs, img_clas

    def imshow(self, img, bboxs, widname=' '):
        for idx, bbox in enumerate(bboxs):
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 10)
        cv2.imshow(widname, img)
        cv2.waitKey(0)

