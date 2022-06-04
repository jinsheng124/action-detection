import torch
import torch.nn as nn
from models.backbones_2d import CSPdarknet, mobilenetv3
from models.backbones_3d import resnext, resnet, mobilenet, mobilenetv2, shufflenet, shufflenetv2
from models.common import BasicConv, CBAM, SE, CAM_Module,Focus
from utils.utils import load_focus_state_dict
import math

pretrained_3d_model = {"ResNext101": "resnext-101-kinetics.pth",
                       "ResNet50": "resnet-50-kinetics.pth",
                       "ResNet18": "resnet-18-kinetics.pth",
                       "ResNet101": "resnet-101-kinetics.pth"}


class STM(nn.Module):
    def __init__(self, in_c, k=1):
        super(STM, self).__init__()
        self.in_c = in_c // (k * k)
        self.k = k

    def forward(self, x):
        # -----特征重排,feature map扩张k倍，通道压缩k*k倍-----#
        N, _, H, W = x.size()
        x = x.view(N, self.in_c, self.k, self.k, H, W).permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(N, self.in_c, self.k * H, self.k * W)
        # ---------------------------------------------------#
        return x


class Fuse_Module(nn.Module):
    def __init__(self, c2d, c3d, clip):
        super(Fuse_Module, self).__init__()
        if c3d == "ResNext50":
            self.backbone_3d = resnext.resnext50()
            in_channel = 2048
        elif c3d == "ResNext101":
            self.backbone_3d = resnext.resnext101()
            in_channel = 2048
        elif c3d == "ResNet18":
            self.backbone_3d = resnet.resnet18()
            in_channel = 512
        elif c3d == "ResNet50":
            self.backbone_3d = resnet.resnet50()
            in_channel = 2048
        elif c3d == "ResNet101":
            self.backbone_3d = resnet.resnet101()
            in_channel = 2048
        elif c3d == "mobilenet_2x":
            self.backbone_3d = mobilenet.get_model(width_mult=2.0)
            in_channel = 2048  # Number of output channels for backbone_3d
        elif c3d == "mobilenetv2_1x":
            self.backbone_3d = mobilenetv2.get_model(width_mult=1.0)
            in_channel = 1280  # Number of output channels for backbone_3d
        elif c3d == "shufflenet_2x":
            self.backbone_3d = shufflenet.get_model(groups=3, width_mult=2.0)
            in_channel = 1920  # Number of output channels for backbone_3d
        elif c3d == "shufflenetv2_2x":
            self.backbone_3d = shufflenetv2.get_model(width_mult=2.0)
            in_channel = 2048  # Number of output channels for backbone_3d
        else:
            raise ValueError("Wrong backbone_3d model is requested")
        if c2d == "cspdarknet53":
            self.backbone_2d = CSPdarknet.darknet53(pretrained="pretrained/cspdarknet.pth")
        elif c2d == "mobilenetv3":
            self.backbone_2d = mobilenetv3.MobilenetV3(weight_path="pretrained/mobilenetv3.pth")
        else:
            raise ValueError("Wrong backbone_2d model is requested")
        self.feature_2d_channels = self.backbone_2d.feature_channels[-3:]
        assert math.ceil(clip / 16) == 1, "clips out of boundary"
        # ---加载3d网络预训练权重-----#
        if c3d in pretrained_3d_model.keys():
            model_path = "pretrained/" + pretrained_3d_model[c3d]
            self.backbone_3d = load_focus_state_dict(self.backbone_3d,model_path,device="cuda")
            # self.backbone_3d = self.backbone_3d.cuda()
            # self.backbone_3d = nn.DataParallel(self.backbone_3d, device_ids=None)
            # pretrained_3d_backbone = torch.load("pretrained/" + pretrained_3d_model[c3d])
            # backbone_3d_dict = self.backbone_3d.state_dict()
            # pretrained_3d_backbone_dict = {k: v for k, v in pretrained_3d_backbone['state_dict'].items() if
            #                                k in backbone_3d_dict}  # 1. filter out unnecessary keys
            # backbone_3d_dict.update(pretrained_3d_backbone_dict)  # 2. overwrite entries in the existing state dict
            # self.backbone_3d.load_state_dict(backbone_3d_dict)  # 3. load the new state dict
            # self.backbone_3d = self.backbone_3d.module  # remove the dataparallel wrapper
            print("load 3d weight done!!!")
        # ------------------------#

        self.stdn1 = STM(in_channel, k=2)
        self.stdn2 = STM(in_channel, k=4)
        downsample = [16, 4, 1]
        self.out_feature_channels = [in_channel // downsample[i] + self.feature_2d_channels[i] for i in range(3)]
        self.conv3 = nn.Sequential(BasicConv(self.out_feature_channels[2], 512, 1), BasicConv(512, 1024, 3, 1, 1))
        self.conv2 = nn.Sequential(BasicConv(self.out_feature_channels[1], 256, 1), BasicConv(256, 512, 3, 1, 1))
        self.conv1 = nn.Sequential(BasicConv(self.out_feature_channels[0], 128, 1), BasicConv(128, 256, 3, 1, 1))
    def forward(self, x_3d, x_2d):
        y = torch.squeeze(self.backbone_3d(x_3d), dim=2)
        x1, x2, x3 = self.backbone_2d(x_2d)
        # 融合
        y1 = self.stdn2(y)
        y2 = self.stdn1(y)
        y3 = y
        out1 = self.conv1(torch.cat([y1, x1], dim=1))
        out2 = self.conv2(torch.cat([y2, x2], dim=1))
        out3 = self.conv3(torch.cat([y3, x3], dim=1))
        return out1, out2, out3