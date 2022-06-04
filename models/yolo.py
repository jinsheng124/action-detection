import torch
import torch.nn as nn
from models.common import BasicConv, DropBlock, SPP, CAM_Module, SE, CBAM
from models.fuse import Fuse_Module
import math


# ppyolo是在第二个卷积块加入dropblock,我这里相当于第一个，如果没作用可以改
def UpSample(p1, p2):
    m = nn.Sequential(
        BasicConv(p1, p2, 1),
        nn.Upsample(scale_factor=2, mode='nearest'))
    return m


class YoloHead(nn.Module):
    def __init__(self, in_c, c_c, out_c, use_channel_attention=True):
        super(YoloHead, self).__init__()
        # 可以考虑用SE模块
        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attention = SE(in_c, rotio=4)
        self.cv1 = BasicConv(in_c, c_c, 3, 1, 1)
        self.detect = nn.Conv2d(c_c, out_c, 1)

    def forward(self, x):
        if self.use_channel_attention:
            x = self.channel_attention(x)
        x = self.cv1(x)
        x = self.detect(x)
        return x


class YoloBody(nn.Module):
    def __init__(self,
                 num_anchors,
                 num_classes,
                 clip=None,
                 c2d="cspdarknet53",
                 c3d="ResNext101"):
        super(YoloBody, self).__init__()
        # dropblock参数
        self.drop_prob = 0.1
        # self.drop_prob = 0
        self.drop_block_size = 3

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.fuse = Fuse_Module(c2d, c3d, clip)

        # 此处可加CAM_Module试试
        self.conv1 = BasicConv(1024,512,1)
        self.spp = SPP(512, k=(3, 5, 7))
        self.conv2 = self._make_conv_block(512,num_layers=1,drop_seen=1)

        self.upsample_for_P4 = UpSample(512, 256)
        self.conv3 = BasicConv(768, 256, 1)
        self.conv4 = self._make_conv_block(256, num_layers=1, drop_seen=1)

        self.upsample_for_P3 = UpSample(256, 128)
        self.conv5 = BasicConv(384, 128, 1)
        self.conv6 = self._make_conv_block(128, num_layers=1, drop_seen=1)

        self.last_channel = num_anchors * (5 + num_classes)
        self.yolo_head0 = YoloHead(512, 1024, self.last_channel)
        self.yolo_head1 = YoloHead(256, 512, self.last_channel)
        self.yolo_head2 = YoloHead(128, 256, self.last_channel)
        self._init_classifier_bias()

    def _make_conv_block(self, in_channel, num_layers=1, drop_seen=None):
        layers = []
        for i in range(num_layers):
            layers.append(BasicConv(in_channel, in_channel * 2, 3, 1, 1))
            if drop_seen:
                if i + 1 == drop_seen:
                    layers.append(
                        DropBlock(drop_prob=self.drop_prob, block_size=self.drop_block_size, scale=True, use_step=True))
            layers.append(BasicConv(in_channel * 2, in_channel, 1))
        return nn.Sequential(*layers)

    def _init_classifier_bias(self):
        # ----------weight初始化-----------#
        # self.detect1.weight.data.fill_(0)
        # self.detect2.weight.data.fill_(0)
        # self.detect3.weight.data.fill_(0)
        # ------bias初始化-----------------#
        prior = 0.01
        mask = torch.zeros(self.num_anchors, 5 + self.num_classes)
        mask[:, 4:] = 1
        mask = mask.view(-1)
        self.yolo_head0.detect.bias.data.masked_fill_(mask == 1, -math.log((1.0 - prior) / prior))
        self.yolo_head1.detect.bias.data.masked_fill_(mask == 1, -math.log((1.0 - prior) / prior))
        self.yolo_head2.detect.bias.data.masked_fill_(mask == 1, -math.log((1.0 - prior) / prior))
        # --------------------------------#

    def forward(self, x_3d,x_2d):
        # 特征融合
        x1, x2, x3 = self.fuse(x_3d, x_2d)
        # 金字塔
        # 第三层
        P5 = self.conv1(x3)
        P5 = self.spp(P5)
        P5 = self.conv2(P5)

        # 第二层
        P5_Upsample = self.upsample_for_P4(P5)
        P4 = torch.cat([x2, P5_Upsample], dim=1)
        P4 = self.conv3(P4)
        P4 = self.conv4(P4)
        # 第一层
        P4_Upsample = self.upsample_for_P3(P4)
        P3 = torch.cat([x1, P4_Upsample], dim=1)
        P3 = self.conv5(P3)
        P3 = self.conv6(P3)
        # 检测头.
        out2 = self.yolo_head2(P3)
        out1 = self.yolo_head1(P4)
        out0 = self.yolo_head0(P5)

        return out0, out1, out2