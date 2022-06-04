import torch
import numpy as np
import os
from utils.utils import nms, bbox_iou, DecodeBox, correct_boxes,non_max_suppression
from utils.region_loss import compute_loss
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
def test(net,
         dataloader,
         class_names,
         anchors,
         epoch=0,
         critical_iou=0.5,
         model_image_size=(224, 224),
         image_shape = (320, 240),
         K_sample = 0,
         ignore_threshold = 0.5,
         conf_thres=0.005,
         nms_thres=0.4,
         save_json = True):
    yolo_decodes = DecodeBox(anchors, len(class_names),(model_image_size[1], model_image_size[0]),K_sample=K_sample)
    bounding_boxes = []
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(devices)
    val_loss = 0.
    for iteration,batch in enumerate(tqdm(dataloader)):
        key_frames,idx, images, labels = batch[0], batch[1], batch[2],batch[3]
        with torch.no_grad():
            images = torch.from_numpy(images).float().to(devices)
            key_image = [images[i,:,key,:,:] for i,key in enumerate(key_frames)]
            key_image = torch.stack(key_image,dim=0)
            labels = [torch.from_numpy(ann).float() for ann in labels]
            outputs = net(images,key_image)
            loss_item = compute_loss(outputs, labels, np.reshape(anchors, [-1, 2]), len(class_names),K_sample=K_sample,ignore_threshold=ignore_threshold)
            val_loss += loss_item[0].item()
            output = yolo_decodes(outputs)
            # 非极大值抑制
            batch_detections = nms(output,conf_thres=conf_thres,nms_thres=nms_thres,only_objection=False,nms_link_classes=True)
            # batch_detections = non_max_suppression(output,conf_thres=conf_thres,nms_thres=nms_thres,only_objection=False)
            for i, o in enumerate(batch_detections):
                #存在预测结果就保存，否则continue
                if o is None:
                    continue
                o = o.data.cpu()
                top_conf = np.array(o[:, 4])
                top_label = np.array(o[:, -1], np.int32)
                top_bboxes = np.array(o[:, :4])
                #去掉灰条
                boxes = correct_boxes(top_bboxes,model_image_size,image_shape)
                #截断，取整
                #boxes=clip_coords(boxes, image_shape)
                #boxes=np.round(boxes).astype('int32')
                for c, l, b in zip(top_conf, top_label, boxes):
                    b = np.around(b, decimals=1)
                    bounding_boxes.append({
                        "conf": "%.6f" % c,
                        "class": int(l),
                        "bbox": b.tolist(),
                        "gt_path": idx[i]
                    })
    val_loss /= (iteration+1)
    print("all boxes catched...")
    # 将所有结果按类别存成.json文件格式
    # [{"confident":0-1,"gt_box":坐标,"truthbox":txt_path,"class":0 or 1 or 2},...]
    current_dir = 'logs/detections/detections_' + str(epoch)
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('logs/detections'):
        os.mkdir('logs/detections')
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)
    pre_bbox_mmap = [None]*len(class_names)
    for c in range(len(class_names)):
        #筛选出其中一类
        bounding_box = list(filter(lambda x: int(x["class"]) == c, bounding_boxes))
        if len(bounding_box) == 0:
            #如果未检测到该类，就从列表中剔除
            # count_class.remove(c)
            continue
        #按置信度排序，从大到小
        bounding_box.sort(key=lambda x: float(x['conf']), reverse=True)
        pre_bbox_mmap[c] = bounding_box
        if save_json:
            with open(current_dir +"/" +class_names[c] + "_dr.json", 'w') as outfile:
                json.dump(bounding_box, outfile)
    #所有真实框个数初始化
    truth_bbox_mmap = {}
    gt_class_list=[]
    #获得真实框
    base_path = dataloader.dataset.base_path
    lines = dataloader.dataset.lines
    split_image_label = dataloader.dataset.split_image_label
    for line in lines:
        line=line.rstrip()
        if split_image_label:
            line=line.replace("/","_")
            gt_box = np.loadtxt(base_path+"/labels/"+line)
        else:
            gt_box = np.loadtxt(base_path+"/labels/"+line)
            line = line.replace("/", "_")
        gt_box = np.reshape(gt_box, (-1, 5)).astype(int)
        #特别注意
        gt_box[:, 0] = gt_box[:, 0] - 1
        gt_class_list.extend(gt_box[:,0].tolist())
        truth_bbox_mmap[line]=gt_box.tolist()
    #所有真实框个数
    truth_num_box = len(gt_class_list)
    #所有预测框个数
    pre_num_box = len(bounding_boxes)
    print(f"truth_num_box: {truth_num_box}, pre_num_box: {pre_num_box}")
    #真实框类别
    count_class = np.unique(gt_class_list)
    #真阳性样本初始化
    tp = 0
    #计算ap
    mmAP = []
    for c in count_class:
        t_box_length = gt_class_list.count(c)
        bounding_box = pre_bbox_mmap[c]
        if bounding_box is None:
            mmAP.append(0.0)
            continue
        #初始化真阳性序列
        pred_match = np.zeros(len(bounding_box))
        for i, obj in enumerate(bounding_box):
            #取出一个预测框
            pre_box = torch.tensor(obj["bbox"])
            #读取真实框
            gt_box = np.array(truth_bbox_mmap[obj["gt_path"]])
            if len(gt_box) == 0:
                continue
            gt_box_t = torch.from_numpy(gt_box[:, 1:]).float()
            pre_box = pre_box.expand_as(gt_box_t)
            #计算预测框与所有真实框iou
            overlaps = bbox_iou(pre_box, gt_box_t).numpy()
            #iou按从大到小排序
            sorted_ixs = np.argsort(-overlaps)
            for s in sorted_ixs:
                #依次判断iou是否大于0.5，小于则说明是假阳性样本，直接退出
                if overlaps[s] < critical_iou:
                    break
                #大于就判断预测类别和真实框类别是否一致，一致则真阳性，将序列位置置一，直接退出循环
                if obj["class"] == int(gt_box[s, 0]):
                    tp += 1
                    pred_match[i] = 1
                    #匹配到便去除真实框
                    truth_bbox_mmap[obj["gt_path"]].pop(s)
                    break
        #累加
        precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
        #此时召回率逐渐上升
        recalls = np.cumsum(pred_match).astype(np.float32) / t_box_length
        # np.savetxt(f"logs/results/{class_names[c]}.txt",np.stack((recalls,precisions),axis=0),delimiter=',')
        # plt.plot(recalls,precisions)
        # plt.xlabel("recall")
        # plt.ylabel("precision")
        # plt.savefig(f"logs/results/{c}.png")
        # plt.show()

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])
        #保证准确率取每个召回率最大值
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        mmAP.append(ap)
    APs={class_names[c]:round(mmAP[i],3) for i, c in enumerate(count_class)}
    mAP = sum(mmAP) / len(mmAP)
    recall = tp / truth_num_box
    precision = tp / pre_num_box if pre_num_box != 0 else 0.0
    return APs, mAP, recall, precision,val_loss
