import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast,GradScaler
import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from utils.dataset import listDataset, dataset_collate_train, dataset_collate_test
from utils.utils import get_classes, get_anchors
# from utils.region_loss import compute_loss
from utils.region_loss import compute_loss
from torch.utils.data import DataLoader
from models.yolo import YoloBody
from models.common import DropBlock
from test import test
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_lr(optimizer,epoch,init_lr=0.0001,milestones=[10,18,25,35],gamma=[0.4,0.5,0.5,0.5]):
    reduce = 1.0
    for i,m in enumerate(milestones):
        if epoch>=m:
            if isinstance(gamma,list):
                reduce *= gamma[i]
            else:
                reduce *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*reduce
# def get_optimizer(net,lr):
#     #采用不同学习率
#     _2d_param = list(map(id,net.module.c2d.backbone_2d.parameters()))
#     base_params = filter(lambda p: id(p) not in _2d_param,net.parameters())
#     optimizer = torch.optim.SGD([
#             {'params': base_params},
#             {'params': net.module.c2d.backbone_2d.parameters(), 'lr': lr*10}],
#             lr=lr, momentum=0.937,weight_decay=5e-4)
#     return optimizer

def fit_one_epoch(epoch, Epoch, gen, genval):
    total_loss,loss_conf,loss_cls,loss_loc = 0,0,0,0
    epoch_size = max(1, len(gen.dataset) // batch_size)
    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            key_frames,images, labels = batch[0], batch[1] , batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).float().to(device)
                #--------数据增强，cutout------------#
                if cutout:
                    c_rt = random.choice([0.05,0.1])
                    c_ae = random.choice([3,5,7])
                    cl = random.randint(0,num_clip//2)
                    mask = [c_rt]*cl+[0]*(num_clip-cl)
                    random.shuffle(mask)
                    images = [DropBlock(mask[k],c_ae,scale=False,use_step=False)(images[:,:,k,:,:]) for k in range(num_clip)]
                    images = torch.stack(images,dim=2)
                #-------多尺度训练,每10个batch--------#
                if multitrain:
                    if iteration % 10 ==0:
                        gz = 32
                        sl = random.uniform(0.7,1.5)
                        img_sz = [int(x*sl//gz*gz) for x in model_image_size]
                    images = [F.interpolate(images[:,:,k,:,:],size=img_sz,mode='bilinear',align_corners=False) for k in range(num_clip)]
                    images = torch.stack(images,dim=2)
                #-------------获得关键帧-------------#
                key_image = [images[i,:,key,:,:] for i,key in enumerate(key_frames)]
                key_image = torch.stack(key_image,dim=0)
                labels = [torch.from_numpy(ann).float() for ann in labels]
            if not amp:
                optimizer.zero_grad()
                outputs = net(images,key_image)
                loss_item = compute_loss(outputs,labels,np.reshape(anchors, [-1, 2]),len(class_names),
                                         label_smooth =smooth_label,ignore_threshold=ignore_threshold,K_sample=K_sample)
                loss = loss_item[0]
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                with autocast():
                    outputs = net(images,key_image)
                    loss_item = compute_loss(outputs,labels,np.reshape(anchors, [-1, 2]),len(class_names),
                                             label_smooth = smooth_label,ignore_threshold=ignore_threshold,K_sample=K_sample)
                    loss = loss_item[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            #保留损失信息
            loss_conf += loss_item[1]
            loss_cls += loss_item[2]
            loss_loc += loss_item[3]
            total_loss += loss.item()
            del loss
            pbar.set_postfix(**{'loss': total_loss / (iteration + 1),
                    'lr': get_lr(optimizer)})
            pbar.update(1)
    train_loss = total_loss / (epoch_size + 1)
    conf_loss = loss_conf / (epoch_size + 1)
    cls_loss = loss_cls / (epoch_size + 1)
    loc_loss = loss_loc / (epoch_size + 1)
    print('Start Validation')
    net.eval()
    ap_dict, Map, recall, precision,valloss = test(net,genval,class_names,anchors,epoch=epoch,model_image_size=model_image_size,K_sample=K_sample,ignore_threshold=ignore_threshold)
    print("each class ap:")
    print(str(ap_dict)[1:-1])
    print("valloss:{:.3f} recall:{:.3f} precision:{:.3f} mAP:{:.3f}".format(valloss,recall, precision, Map))
    return train_loss,conf_loss,cls_loss,loc_loss,ap_dict,valloss, Map, recall, precision

if __name__ == "__main__":
    #设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_mAP,start_epoch = 0,0

    #超参数
    #------------tricks--------------#
    #是否使用Adam优化器
    adam = False
    use_scheduler = False
    #是否多尺度训练
    multitrain = False
    #cutout
    cutout = False
    #是否标签平滑，可设为0-0.1的小值
    #dropblock的值设为了0.1
    smooth_label = 0.1
    ignore_threshold = 0.45
    K_sample = 0
    #混合精度
    amp = True
    if amp:
        scaler = GradScaler()

    #---------------------------------#
    # anchors变过，在PC端
    #-----------------训练参数--------------------#
    #是否恢复现场继续训练及其恢复权重路径
    init_seed = False
    # torch.backends.cudnn.benchmark = True
    if init_seed:
        seed = 42
        #初始化种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #使用卷积固定算法
        torch.backends.cudnn.determinsitic = True
    Train_Next = False
    use_tb_writer = False
    weight_path = "logs/last.pt"
    #总迭代次数
    end_epoch = 40
    #冻结次数，可选择关闭冻结
    freeze_2d = True
    freeze_3d = True
    freeze_3d_epoch,freeze_2d_epoch= 10,25
    #学习率
    lr = 0.0001
    # 模型size
    model_image_size = (224,224)
    #batch_size
    batch_size = 14
    batch_size_val = 16
    #抓取图片数
    num_clip = 8
    #加载类别和先验框参数
    anchors_path = 'model_data/jhmdb_21_anchors.txt'
    classes_path = 'model_data/jhmdb_21_classes.txt'
    #-----------------------------------------------#

    #dataset、dataloader build
    base_path = 'jhmdb-21'
    split_image_label = True
    num_workers = 2
    train_dataset = listDataset(base_path,shape=model_image_size,train=True,clip_duration=num_clip,split_image_label=split_image_label)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,
                                  pin_memory=True,shuffle=True,drop_last=True,
                                  collate_fn=dataset_collate_train)
    test_dataset = listDataset(base_path,shape=model_image_size,train=False,clip_duration=num_clip,split_image_label=split_image_label)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size_val,
                                 num_workers=num_workers,pin_memory=True,
                                 drop_last=False,collate_fn=dataset_collate_test)
    #加载类别,先验框,模型,多GPU
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    hyp = {"class_names":class_names,"anchors":anchors}

    model = YoloBody(num_anchors=len(anchors[0]),num_classes=len(class_names),clip=num_clip,c2d="cspdarknet53",c3d="ResNext101")
    # net = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(model,device_ids=[0])
    net = net.to(device) 
    #冻结参数
    if freeze_2d:
        for param in net.module.fuse.backbone_2d.parameters():
            param.requires_grad = False
    if freeze_3d:
        for param in net.module.fuse.backbone_3d.parameters():
            param.requires_grad = False

    #优化器
    if adam:
        optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=5e-4)
        # lr_scheduler = None
    else:
        optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.937,weight_decay=5e-4)
    #断点训练
    if Train_Next and os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        net.load_state_dict(checkpoint["model"])
        if checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_mAP = checkpoint["score"]
        del checkpoint
        torch.cuda.empty_cache()
        if start_epoch >= end_epoch:
            end_epoch += start_epoch
    if use_scheduler:
        #余弦退火
        pr = 2
        lf = lambda x: (((1 + math.cos(x * math.pi / end_epoch)) / 2) ** pr) * 0.99 + 0.01  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1  # do not move
        lr *= lf(start_epoch)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,25,45],gamma=0.1)
    #-----------tensorboard----------------#
    if use_tb_writer:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir="logs/results")
        graph_inputs_3d = torch.randn(1,3,num_clip,*model_image_size).to(device)
        graph_inputs_2d = graph_inputs_3d[:,:,-1,:,:]
        tb_writer.add_graph(model,(graph_inputs_3d,graph_inputs_2d))
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('logs/results'):
        os.mkdir('logs/results')
    print('Start training...')
    #--------------------------------------#
    #开始训练
    for epoch in range(start_epoch, end_epoch):
        if not use_scheduler:
            fit_lr(optimizer, epoch, init_lr=lr)
        # fit_lr(optimizer,epoch,init_lr=lr,milestones=[8,15],gamma=[0.2,0.2])
        #判断是否解冻3d网络,并减少batch
        if epoch>= freeze_3d_epoch and freeze_3d:
            freeze_3d = False        
            for param in net.module.fuse.backbone_3d.parameters():
                param.requires_grad = True
            del train_dataloader
            torch.cuda.empty_cache()
            batch_size = 10
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=2,
                                  pin_memory=True,shuffle=True,drop_last=True,collate_fn=dataset_collate_train)
        # 判断是否解冻2d网络，并减少batch
        if epoch >= freeze_2d_epoch and freeze_2d:
            freeze_2d = False
            for param in net.module.fuse.backbone_2d.parameters():
                param.requires_grad = True
            del train_dataloader
            torch.cuda.empty_cache()
            batch_size = 6
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2,
                                          pin_memory=True, shuffle=True, drop_last=True,
                                          collate_fn=dataset_collate_train)
        train_loss,conf_loss,cls_loss,loc_loss,APs,valloss, mAP, recall, precision = fit_one_epoch(epoch, end_epoch, train_dataloader, test_dataloader)
        #更新学习率
        if use_scheduler:
            scheduler.step()
        #保存结果
        #-----用tensorboard保存结果以便可视化---------#
        if use_tb_writer:
            tags = ["train_loss","conf_loss","cls_loss","loc_loss","valloss","recall","precision","mAP@0.5"]
            for x,tag in zip([train_loss,conf_loss,cls_loss,loc_loss, valloss,recall, precision,mAP],tags):
                tb_writer.add_scalar(tag,x,epoch)
        #-------------------------------------------#
        if epoch == 0:
            s="{:<10s}"*9+"\n"
            s=s.format('epoch','loss','conf_loss','cls_loss','loc_loss','valloss','recall','precise','mAP')
            with open("logs/results/mAP.txt", "w") as f:
                f.write(s)
        s ="{:<10d}"+"{:<10.3f}"*8+"\n"
        s = s.format(epoch,train_loss,conf_loss,cls_loss,loc_loss,valloss,recall,precision,mAP)
        with open("logs/results/mAP.txt", "a+") as f:
            f.write(s)
        with open("logs/results/AP.txt", "a+") as f:
            ap = {"epoch": epoch, **APs}
            ap = str(ap) + "\n"
            f.write(ap)

        #保存模型
        state = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'score': mAP,
            'hyp':hyp}
        print('Saving state, iter:', str(epoch + 1))
        torch.save(state, 'logs/last.pt')
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(state, 'logs/best.pt')
            #------------最佳结果绘图--------------#
            paras = {'figure.figsize':'10,10'}
            plt.rcParams.update(paras)
            plt.clf()
            xi = list(APs.keys())
            yi = list(APs.values())
            plt.bar(xi,yi,align="center",color = "b",alpha=0.6)
            plt.xticks(xi,xi,rotation=60)
            for xn,yn in zip(xi,yi):
                plt.text(xn,yn+0.01,"%.2f"%yn,ha = "center",va = "bottom",fontsize =10)
            plt.text(0,1.1,f"mAP = {mAP:.3f}",fontsize =15)
            plt.ylim(0,1)
            plt.ylabel("AP")
            plt.savefig("logs/results/best_AP.png")
            #------------------------------------#