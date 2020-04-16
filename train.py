import torch.backends.cudnn as cudnn
import math
import time
import datetime
import argparse
import os
import torch
from torch.autograd import Variable
from vision.network import Yolo_v1
from vision.yolov1_loss import YoloV1Loss
from tools.dataloader import TrainDataset

from tools.augment import DataAugmentation
from torch.utils.data import DataLoader
from tools.util import *


parser = argparse.ArgumentParser("--------Train yolo-v1--------")
parser.add_argument('--weights_save_folder', default='./weights', type=str, help='Dir to save weights')
parser.add_argument('--imgs_dir', default='/data/img/', help='train images dir')
parser.add_argument('--annos_dir', default='/data/annotation/', type=str, help='annotation xml dir')
parser.add_argument('--batch_size', default=16, type=int, help="batch size")
parser.add_argument('--net_w', default=448, type=int, help="input image width")
parser.add_argument('--net_h', default=448, type=int, help="input image height")
parser.add_argument('--max_epoch', default=30, type=int, help="max training epoch")
parser.add_argument('--initial_lr', default='1e-3', type=float, help="initial learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="gamma for adjust lr")
parser.add_argument('--weight_decay', default=5e-4, type=float, help="weights decay")
parser.add_argument('--decay1', default=15, type=int)
parser.add_argument('--decay2', default=25, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--num_gpu', default=1, type=int, help="gpu number")
parser.add_argument('--pre_train', default=False, type=bool, help="whether use pre-train weights for change class number")
args = parser.parse_args()

class_name = ["aby", "dazongdianping", "douying", "fangtianxia", "lashou", "weixin", "xiaozhu", "yilong", "youtianxia"]


def train(net, optimizer, trainSet, use_gpu):
    net.train()
    epoch = 0
    print('Loading Dataset...')

    epoch_size = math.ceil(len(trainSet) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    stepvalues = (args.decay1 * epoch_size, args.decay2 * epoch_size)
    step_index = 0
    start_iter = 0

    print("Begin training...")
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            epoch += 1

            batch_iterator = iter(DataLoader(trainSet, args.batch_size, shuffle=True, num_workers=args.num_workers))
            if epoch % 10 == 0 and epoch > 0:
                if args.num_gpu > 1:
                    torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
                else:
                    torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(args.initial_lr, optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)

        images, targets = Variable(images), Variable(targets)
        if use_gpu:
            images = images.cuda()
            targets = targets.cuda()

        out = net(images)
        optimizer.zero_grad()
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f}|| LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, args.max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss, lr, batch_time, str(datetime.timedelta(seconds=eta))))
    if args.num_gpu > 1:
        torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'Final.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'Final.pth'))
    print('Finished Training')


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    net = Yolo_v1(class_num=len(class_name))

    if args.pre_train:

        pretrained_dict = torch.load(os.path.join(args.weights_save_folder, "Final.pth"))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if args.num_gpu > 1 and use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    elif use_gpu:
        net = net.cuda()

    cudnn.benchmark = True
    criterion = YoloV1Loss(B=2, use_gpu=use_gpu)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    augmentation = DataAugmentation()
    trainSet = TrainDataset(img_dir=args.imgs_dir, xml_dir=args.annos_dir, target_size=(args.net_w, args.net_h),
                            S=7, B=2, name_list=class_name, augmentation=augmentation, transform=transform)

    train(net, optimizer, trainSet, use_gpu)
