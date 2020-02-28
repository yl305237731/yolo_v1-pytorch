import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SeparableConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depth_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    groups=self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.point_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Yolo_v1(nn.Module):
    def __init__(self, class_num, box_num=2):
        super(Yolo_v1, self).__init__()
        self.C = class_num
        self.box_num = box_num
        self.out_channel = self.box_num * 5 + self.C
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7//2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            SeparableConv2D(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3//2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            SeparableConv2D(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer4 = nn.Sequential(
            SeparableConv2D(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer5 = nn.Sequential(
            SeparableConv2D(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            SeparableConv2D(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3//2),
        )
        self.conv_layer6 = nn.Sequential(
            SeparableConv2D(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            SeparableConv2D(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
        )
        # self.flatten = Flatten()
        # self.fc1 = nn.Sequential(
        #     nn.Linear(in_features=7*7*1024, out_features=4096),
        #     nn.Dropout(),
        #     nn.LeakyReLU(0.1)
        # )
        # self.fc2 = nn.Sequential(nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + self.C)),
        #                          nn.Sigmoid())

        self.conv_out = nn.Sequential(
            SeparableConv2D(in_channels=1024, out_channels=self.out_channel, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_layer1 = self.conv_layer1(x)
        conv_layer2 = self.conv_layer2(conv_layer1)
        conv_layer3 = self.conv_layer3(conv_layer2)
        conv_layer4 = self.conv_layer4(conv_layer3)
        conv_layer5 = self.conv_layer5(conv_layer4)
        conv_layer6 = self.conv_layer6(conv_layer5)
        # flatten = self.flatten(conv_layer6)
        # fc1 = self.fc1(flatten)
        # fc2 = self.fc2(fc1)
        # output = fc2.reshape([-1, 7, 7, 2 * 5 + self.C])
        output = self.conv_out(conv_layer6)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output
