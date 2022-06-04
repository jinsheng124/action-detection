import cv2
import numpy as np
import torch
import torch.nn as nn
import cv2
import random
from models.yolo import YoloBody
from utils.utils import nms, DecodeBox, letterbox_image, correct_boxes, clip_coords,get_classes, get_anchors,xywh2xyxy

class VideoDetect(object):
    def __init__(self,
                 image_size=(224, 224),
                 model_path="logs/best.pt",
                 anchors_path='model_data/jhmdb_21_anchors.txt',
                 classes_path='model_data/jhmdb_21_classes.txt',
                 conf_thres=0.5,
                 iou_thres=0.3,
                 num_clip=8,
                 K_sample = 0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.num_clip = num_clip
        self.anchors = get_anchors(anchors_path)
        self.class_names = get_classes(classes_path)

        self.net = self.load_model(model_path, self.anchors, self.class_names,self.num_clip)
        self.yolo_decodes = DecodeBox(self.anchors, len(self.class_names),
                                    (self.image_size[1], self.image_size[0]),K_sample=K_sample)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]

    def load_model(self, path, anchors, class_names, num_clip):
        net = YoloBody(num_anchors=len(anchors[0]),num_classes=len(class_names),clip=num_clip,c2d="cspdarknet53",c3d="ResNext101").eval()
        net = nn.DataParallel(net)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(path, map_location=device)
        net.load_state_dict(state_dict["model"])
        # net.load_state_dict(state_dict)
        net = net.to(device)
        del state_dict
        print("load model done!!!")
        return net

    def plot_one_box(self,box,img,color=None,label=None,line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) /2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 2, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3,thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img,label, (c1[0], c1[1] - 2),0,tl / 4, [225, 255, 255],thickness=tf*2,lineType=cv2.LINE_AA)

    def detect(self, images,key_frame=-1):
        image = images[key_frame]
        image_shape = image.size
        #加灰条，防失真，推荐
        batch_img = [np.array(letterbox_image(img, self.image_size)) for img in images]
        batch_img = np.stack(batch_img) / 255.0
        batch_img = batch_img.transpose((3, 0, 1, 2))
        batch_img = torch.from_numpy(batch_img).unsqueeze(0).float()
        batch_key_frame = batch_img[:,:,key_frame,:,:]
        with torch.no_grad():
            batch_img = batch_img.to(self.device)
            batch_key_frame = batch_key_frame.to(self.device)
            outputs = self.net(batch_img,batch_key_frame)
        output = self.yolo_decodes(outputs)
        #非极大值抑制
        batch_detections = nms(output,conf_thres=self.conf_thres,nms_thres=self.iou_thres)
        #转为cv2格式
        image = np.array(image)
        # RGBtoBGR满足opencv显示格式
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #取出结果
        try:
            batch_detections = batch_detections[0].data.cpu()
        except:
            return image
        #置信度，标签，和框参数
        top_conf = np.array(batch_detections[:, 4])
        top_label = np.array(batch_detections[:, -1], np.int32)
        top_bboxes = np.array(batch_detections[:, :4])
        #去掉灰条
        top_bboxes = correct_boxes(top_bboxes,self.image_size,image_shape)
        #截断，取整
        boxes=clip_coords(top_bboxes, image_shape)
        boxes=np.round(boxes).astype('int32')
        s = ""
        for c in np.unique(top_label):
            n = (top_label == c).sum()
            s += '%g %ss, ' % (n,self.class_names[c])  # add to string
        if s:
            print(s[:-2])
        for i, c in enumerate(top_label):
            label = '{} {:.2f}'.format(self.class_names[c],top_conf[i])
            box = boxes[i]
            self.plot_one_box(box,image,color=self.colors[c],label=label,line_thickness=3)
        return image
