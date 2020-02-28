from __future__ import print_function
import os
import argparse
import torch
from vision.network import Yolo_v1
import torch.backends.cudnn as cudnn
import time
from tools.util import *


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./result', type=str, help='Dir to save img')
parser.add_argument('--cpu', default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.8, type=float, help='confidence_threshold')
parser.add_argument('--class_threshold', default=0.8, type=float, help='class threshold')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--net_w', default=448, type=int)
parser.add_argument('--net_h', default=448, type=int)
parser.add_argument('--input_path', default='test.jpg', type=str, help="image or images dir")
args = parser.parse_args()

labels = ["aby", "dazongdianping", "douying", "fangtianxia", "lashou", "weixin", "xiaozhu", "yilong", "youtianxia"]


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    net_w = args.net_w
    net_h = args.net_h

    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cpu = args.cpu
    confidence_threshold = args.confidence_threshold
    class_threshold = args.class_threshold
    nms_thresh = args.nms_threshold
    class_num = len(labels)
    device = torch.device("cpu" if cpu else "cuda")

    net = Yolo_v1(class_num=class_num)
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device(device)))
    net.eval()

    cudnn.benchmark = True
    net = net.to(device)

    input_path = args.input_path
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    for img_path in image_paths:
        begin = time.time()
        print("Detect {}".format(img_path))
        image = cv2.imread(img_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = img.shape
        img = preprocess_input(img, net_h, net_w).to(device)
        net_out = net(img)
        boxes = decode_netout(torch.squeeze(net_out, dim=0), confidence_thresh=confidence_threshold, net_w=net_w, net_h=net_h)
        correct_yolo_boxes(boxes, img_h, img_w, net_h, net_w)
        do_nms(boxes, nms_thresh)
        image = draw_boxes(image, boxes, labels, class_threshold)
        cv2.imwrite(os.path.join(save_folder, img_path.split('/')[-1]), image)
        end = time.time()
        print("per image tiem: {}".format(end - begin))

    print("Done!!!")
