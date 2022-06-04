import torch
import torch.nn as nn
import math
from utils.utils import bbox_iou
def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]
#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_BCE(y_true, eps,num_classes):
    return y_true * (1.0 - eps) + eps / num_classes

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou
def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result
def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output
#二分类facalloss
def FocalLoss(preds,targets,gamma=2,alpha=0.25):
    epsilon = 1e-7
    preds = clip_by_tensor(preds, epsilon, 1.0 - epsilon)
    loss = -targets * torch.log(preds) - (1.0 - targets) * torch.log(1.0 - preds)
    p_t = targets * preds + (1 - targets) * (1 - preds)
    alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= alpha_factor * modulating_factor
    return loss
# 多分类损失
class SmoothLabelCrossEntropy(nn.Module):
    def __init__(self, smooth = 0.1 ,gamma = 0, epsilon=1e-7):
        super(SmoothLabelCrossEntropy,self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.smooth = smooth
    def forward(self,preds,targets):
        preds = clip_by_tensor(preds, self.epsilon, 1.0 - self.epsilon)
        n = preds.size()[-1]
        log_preds = torch.log(preds)
        if self.gamma != 0:
            log_preds *= (1 - preds) ** self.gamma
        loss = - log_preds * targets
        if self.smooth != 0:
            loss = -log_preds / n * self.smooth + loss * (1 - self.smooth)
        return loss
class KIOUloss(nn.Module):
    def __init__(self,div = 0.1,gamma=2):
        super(KIOUloss, self).__init__()
        self.div = div
        self.kk = div-math.log(div)*(1-div)**gamma
        self.gamma = gamma

    def forward(self, pred, target):
        #计算iou
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)
        #---------------------------------------------------#
        loss_high = -(1-iou[iou>self.div])**self.gamma*torch.log(iou[iou>self.div])
        loss_low = self.kk - iou[iou<=self.div]
        loss = loss_low.sum()+loss_high.sum()
        return loss
#yolo3损失
def compute_loss(preds,targets,anchors,num_classes,
                 strides=[32, 16, 8],
                 label_smooth=0,
                 ignore_threshold = 0.5,
                 lambda_mask={"conf":1.0,"cls":1.0,"loc":1.0},
                 K_sample = 0):
    #yolo5中超参数，可借鉴
    # lambda_mask={"conf":1.0,"cls":0.5,"loc":0.05},
    # feature_mask={8:4.0,16:1.0,32:0.4}
    anchors = torch.FloatTensor(anchors)
    losses, lconf, lcls, lloc = [], 0, 0, 0
    for layer, pred in enumerate(preds):
        device = pred.device
        scaled_anchors = anchors / strides[layer]
        #batchsize
        bs = pred.size(0)
        #特征图宽高
        in_h = pred.size(2)
        in_w = pred.size(3)
        # 对prediction预测进行调整
        pred = pred.view(bs, scaled_anchors.shape[0]//3,5+num_classes, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        conf = torch.sigmoid(pred[..., 4])  # Conf
        pred_cls = torch.sigmoid(pred[..., 5:])  # Cls pred.
        # pred_cls = torch.softmax(pred[...,5:],dim=-1)
        #获得正例掩模，负例掩模，类别掩模，框缩放掩模，预测结果，实际结果
        obj_mask,noobj_mask,tcls_mask,box_loss_scale,pred_boxes,t_box = \
                                    build_target(pred,targets,layer,scaled_anchors,num_classes,in_w,in_h,ignore_threshold,K_sample)
        #.......#
        obj_mask, noobj_mask = obj_mask.to(device), noobj_mask.to(device)
        tcls_mask, box_loss_scale = tcls_mask.to(device), box_loss_scale.to(device)
        pred_boxes, t_box = pred_boxes.to(device), t_box.to(device)
        #.......#
        #计算框ciou损失
        ciou = (1 -box_ciou(pred_boxes[obj_mask.bool()], t_box[obj_mask.bool()])) * box_loss_scale[obj_mask.bool()]
        loss_loc = torch.sum(ciou / bs)
        # #正例样本和负例样本的目标损失
        # loss_conf = torch.sum(BCELoss(conf, obj_mask) * obj_mask / bs) + \
        #             torch.sum(BCELoss(conf, obj_mask) * noobj_mask / bs)
        # #二分回归损失
        # loss_cls = torch.sum(BCELoss(pred_cls[obj_mask == 1],smooth_BCE(tcls_mask[obj_mask == 1], label_smooth,num_classes)) / bs)

        loss_conf = torch.sum(FocalLoss(conf, obj_mask,alpha=0.5) * obj_mask / bs) + \
                    torch.sum(FocalLoss(conf, obj_mask,alpha=0.5) * noobj_mask / bs)
        loss_cls = torch.sum(FocalLoss(pred_cls[obj_mask == 1],smooth_BCE(tcls_mask[obj_mask == 1], label_smooth,num_classes)) / bs)
        
        #相加
        loss = loss_conf * lambda_mask["conf"] + loss_cls * lambda_mask["cls"] + loss_loc * lambda_mask["loc"]
        #总损失，对应每个特征图权重
        losses.append(loss)
        lcls += loss_cls.item()
        lconf += loss_conf.item()
        lloc += loss_loc.item()
    losses = sum(losses)
    return losses, lconf, lcls, lloc


def build_target(pred,targets,layer,scaled_anchors,num_classes,in_w,in_h,ignore_threshold,K_sample):
    FloatTensor = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor
    bs = len(targets)
    #每个特征层分配先验框个数
    num_anchors = scaled_anchors.size(0)//3
    #分配先验框的索引
    anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][layer]
    #生成掩模,用于损失计算
    #目标
    obj_mask = torch.zeros(bs, num_anchors, in_w, in_h, requires_grad=False)
    #框参数
    t_box = torch.zeros(bs, num_anchors, in_h, in_w, 4, requires_grad=False)
    #缩放比例
    box_loss_scale = torch.zeros(bs,num_anchors,in_h,in_w,requires_grad=False)
    #类别
    tcls_mask = torch.zeros(bs,num_anchors,in_h,in_w,num_classes,requires_grad=False)
    for b in range(bs):
        costs = {}
        for t in range(targets[b].shape[0]):
            gx = targets[b][t, 0] * in_w
            gy = targets[b][t, 1] * in_h
            gw = targets[b][t, 2] * in_w
            gh = targets[b][t, 3] * in_h
            left = int(gx)
            top = int(gy)
            gt_box = torch.tensor([0, 0, gw, gh]).unsqueeze(0)
            anchor_shapes = torch.cat([torch.zeros(scaled_anchors.size(0), 2), scaled_anchors], 1)
            # 计算与9个先验框重合程度
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            #如果不在当前特征图分配的先验框中，则不进行匹配
            # best_n = torch.argmax(anch_ious).item()
            # if best_n not in anchor_index:
            #     continue
            #------------或许可以增加召回率，又能避免标签重写------------#
            best_n = -1
            topk = torch.topk(anch_ious,k=3,sorted=True)[1].tolist()
            for k,index in enumerate(topk):
                if index in anchor_index:
                    best_n = index
                    break
            if (best_n == -1) or (k > layer):
                continue
            #-----------------------------------------------------------#
            best_n = best_n - anchor_index[0]
            for gi in range(left,left+K_sample+1):
                for gj in range(top,top+K_sample+1):
                    if (gi < in_h) and (gj < in_w):
                        #与真实框的欧式距离作为代价值
                        cost = ((gx.item() - gi) ** 2 + (gy.item() - gj) ** 2) ** 0.5
                        #如果当前[gi,gj]代价大于已经存在的代价值，不进行更新，即防止一个特征点预测两个真实框
                        if (best_n,gi,gj) in costs.keys() and costs[(best_n,gi,gj)]<cost:
                            continue
                        costs[(best_n,gi,gj)] = cost
                        obj_mask[b, best_n, gj, gi] = 1
                        # 计算先验框中心调整参数
                        t_box[b, best_n, gj, gi, 0] = gx
                        t_box[b, best_n, gj, gi, 1] = gy
                        # 计算先验框宽高调整参数
                        t_box[b, best_n, gj, gi, 2] = gw
                        t_box[b, best_n, gj, gi, 3] = gh
                        # 这个参数是该框损失的缩放比例，为了给大框进行惩罚
                        box_loss_scale[b, best_n, gj, gi] = 2 - targets[b][t, 3] * targets[b][t, 2]
                        # 属于哪一类
                        tcls_mask[b, best_n, gj, gi, int(targets[b][t, 4])] = 1
    #负例样本掩模
    noobj_mask = 1 - obj_mask
    #获得实际分配的先验框
    scaled_anchors = scaled_anchors[anchor_index]
    # 先验框的中心位置的调整参数
    if K_sample == 0:
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
    elif K_sample==1:
        x = torch.tanh(pred[..., 0])
        y = torch.tanh(pred[..., 1])
    else:
        raise ValueError("K_sample must be 0 or 1")
    # 先验框的宽高调整参数
    w = pred[..., 2]  # Width
    h = pred[..., 3]  # Height
    #生成所有网格点对应的先验框x,y,w,在特征图坐标
    grid_x = torch.arange(0, in_w).repeat(in_w, 1).repeat(bs * num_anchors, 1, 1).view_as(x).type(FloatTensor)
    grid_y = torch.arange(0, in_h).repeat(in_h, 1).t().repeat(bs * num_anchors, 1, 1).view_as(y).type(FloatTensor)
    anchor_w = scaled_anchors.type(FloatTensor).index_select(1, LongTensor([0]))
    anchor_h = scaled_anchors.type(FloatTensor).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view_as(w)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view_as(h)
    # 计算调整后的先验框中心与宽高
    pred_boxes = FloatTensor(pred[..., :4].shape)
    if K_sample == 0:
        pred_boxes[..., 0] = x * 1.05 + grid_x - 0.025
        pred_boxes[..., 1] = y * 1.05 + grid_y - 0.025
    else:
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h
    #计算负例样本该忽略的，以更新负例样本掩模
    for i in range(bs):
        #所有预测框
        pred_boxes_for_ignore = pred_boxes[i]
        pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
        if len(targets[i]) > 0:
            #真实框
            gt_box = targets[i][:,:4] * torch.FloatTensor([[in_w, in_h, in_w, in_h]])
            gt_box = gt_box.type(FloatTensor)
            #计算iou
            anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
            for t in range(targets[i].shape[0]):
                anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
                #大于阈值的负例样本忽略
                noobj_mask[i][anch_iou > ignore_threshold] = 0

    return obj_mask, noobj_mask, tcls_mask, box_loss_scale, pred_boxes, t_box
