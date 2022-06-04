from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size, K_sample = 0):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.K_sample = K_sample
    def _decode_one_layer(self,input,anchors):
        num_anchors = len(anchors)
        batch_size = input.size(0)
        # 13，13
        input_height = input.size(2)
        input_width = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        # 416/13 = 32
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                          for anchor_width, anchor_height in anchors]

        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(batch_size, num_anchors, 5 + self.num_classes, input_height, input_width).\
            permute(0, 1,3,4,2).contiguous()

        # 先验框的中心位置的调整参数
        if self.K_sample == 0:
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
        else:
            x = torch.tanh(prediction[..., 0])
            y = torch.tanh(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角 batch_size,3,13,13
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(
            1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(
            1, 1, input_height * input_width).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        if self.K_sample==0:
            pred_boxes[..., 0] = x.data * 1.05 + grid_x - 0.025
            pred_boxes[..., 1] = y.data * 1.05 + grid_y - 0.025
        else:
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)),
                           -1)
        return output
    def forward(self, inputs):
        outputs = []
        for layer,input in enumerate(inputs):
            layer_anchors = self.anchors[layer]
            output = self._decode_one_layer(input,layer_anchors)
            outputs.append(output)
        outputs = torch.cat(outputs,dim=1)
        return outputs.data
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image
def correct_boxes(boxes, model_image_size,image_shape):
    w,h = image_shape
    wi,hi=model_image_size
    boxes/=np.array([[wi,hi,wi,hi]])
    shape = np.array([w, h, w, h])
    offset = (shape - np.max(shape)) / np.max(shape) / 2.0
    offset = np.expand_dims(offset, axis=0)
    boxes = (boxes + offset) / (1.0 + 2 * offset) * np.expand_dims(shape, axis=0)
    return boxes
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0]=np.clip(boxes[:, 0],0, img_shape[0]) # x1
    boxes[:, 1]=np.clip(boxes[:, 1],0, img_shape[1])  # y1
    boxes[:, 2]=np.clip(boxes[:, 2],0, img_shape[0])  # x2
    boxes[:, 3]=np.clip(boxes[:, 3],0, img_shape[1])  # y2
    return boxes
def bbox_iou(box1, box2, x1y1x2y2=True,DIoU=False):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    if DIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + 1e-16
        rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
        return iou - rho2 / c2  # DIoU

    return iou
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    x[...,:4] = y[...,:4]
    return x
def nms(prediction,conf_thres=0.5,nms_thres=0.4,xywh = True,only_objection=False,nms_link_classes = True):
    if xywh:
        prediction = xywh2xyxy(prediction)
    output = [None] * prediction.shape[0]
    max_wh = 4096
    xc = prediction[..., 4] > conf_thres
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[xc[image_i]]
        if not image_pred.size(0):
            continue
        if only_objection:
            box= image_pred[:, :4]
            conf=image_pred[:,4:5]
            j = torch.argmax(image_pred[:, 5:],dim=1,keepdim=True)
            image_pred = torch.cat((box, conf, j.float()), 1)
        else:
            image_pred[:, 5:] *= image_pred[:, 4:5]
            box= image_pred[:, :4]
            conf, j = image_pred[:, 5:].max(1, keepdim=True)
            image_pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        #找比框中参数最大的那个，加1为偏移单位，乘以类别作为偏移量
        offset = image_pred[:,5:6] * max_wh if nms_link_classes else 0
        # offset = image_pred[:,5:6] * (box.max()-box.min()+1)
        #偏移不影响iou计算
        boxes, scores = image_pred[:, :4] + offset, image_pred[:, 4]
        keep=torchvision.ops.nms(boxes, scores,nms_thres)
        output[image_i] = image_pred[keep]
    return output
def video_nms(prediction,conf_thres=0.5,nms_thres=0.4):
    max_wh = 4096
    prediction = xywh2xyxy(prediction)
    output = [None] * prediction.shape[0]
    xc = prediction[..., 4] > conf_thres
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[xc[image_i]]
        if not image_pred.size(0):
            continue
        image_pred[:, 5:] *= image_pred[:, 4:5]
        box = image_pred[:, :4]
        conf, j = image_pred[:, 5:].max(1, keepdim=True)
        mask = conf.view(-1) > conf_thres
        image_pred = image_pred[mask]

        #偏移不影响iou计算
        offset = j[mask].float()*max_wh
        boxes, scores = image_pred[:, :4]+offset, conf[mask].view(-1)
        keep=torchvision.ops.nms(boxes, scores,nms_thres)
        output[image_i] = image_pred[keep]
    return output

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.6,only_objection=True):
    # 求左上角和右下角
    prediction = xywh2xyxy(prediction)
    output = [None] * prediction.shape[0]
    xc = prediction[..., 4] > conf_thres
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[xc[image_i]]
        if not image_pred.size(0):
            continue
        if only_objection:
            box= image_pred[:, :4]
            conf=image_pred[:,4:5]
            j = torch.argmax(image_pred[:, 5:],dim=1,keepdim=True)
            detections = torch.cat((box, conf, j.float()), 1)
        else:
            image_pred[:, 5:] *= image_pred[:, 4:5]
            box= image_pred[:, :4]
            conf, j = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # 获得种类
        unique_labels = detections[:, -1].unique()
        all_classes_detection = []
        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]
            # 按照存在物体的置信度排序
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # 进行非极大抑制
            max_detections = []
            while detections_class.size(0):
                # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:],DIoU=True)
                detections_class = detections_class[1:][ious < nms_thres]
            # 堆叠
            max_detections = torch.cat(max_detections)
            all_classes_detection.append(max_detections)
        if len(all_classes_detection)!=0:
            all_classes_detection = torch.cat(all_classes_detection)
            output[image_i] = all_classes_detection
    return output
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]
def shuffle_net(model_path):
    net = torch.load(model_path)
    net["optimizer"] = None
    torch.save(net, model_path)
def load_focus_state_dict(model,pretrain_path,device = "cuda"):
    if device == "cuda":
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
    pretrain_dict = torch.load(pretrain_path)
    model_dict = model.state_dict()
    gain_dict = {}
    for k,v in pretrain_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) ==  np.shape(v):
            gain_dict[k] = v
    model_dict.update(gain_dict)
    model.load_state_dict(model_dict)
    return (model.module if device=="cuda" else model)