import random
import os
from PIL import Image
import numpy as np
import cv2
from utils.utils import letterbox_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a
def read_label(labpath):
    bs = []
    if not os.path.exists(labpath):
        return []
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return []
        bs = np.reshape(bs, (-1, 5))
        bs[:, 0] = bs[:, 0] - 1
    return bs
def get_rand_para(input_shape, jitter=.2, hue=.1, sat=1.5, val=1.5):
    #调整图片大小参数
    h, w = input_shape
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.5, 1.5)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    # 放置图片参数
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    #翻转参数
    flip = rand() < 0.5
    #色域变换参数
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    return (nw, nh, dx, dy, flip, hue, sat, val)


def get_random_image(image, input_shape, nw, nh, dx, dy, flip, hue, sat, val):
    # 实时数据增强的随机预处理
    h, w = input_shape
    # 调整图片大小
    image = image.resize((nw, nh), Image.BICUBIC)

    # 放置图片
    new_image = Image.new('RGB', (w, h),(np.random.randint(0, 255), np.random.randint(0, 255),
                         np.random.randint(0, 255)))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 是否翻转图片
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域变换
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
    return image_data


def get_random_box(box, input_shape, image_size,nw, nh, dx, dy, flip):
    h, w = input_shape
    # 调整目标框坐标
    box_data = np.zeros_like(box)
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [1, 3]] = box[:, [1, 3]] * nw / image_size[0] + dx
        box[:, [2, 4]] = box[:, [2, 4]] * nh / image_size[1] + dy
        if flip:
            box[:, [1, 3]] = w - box[:, [3, 1]]
        box[:, 1:3][box[:, 1:3] < 0] = 0
        box[:, 3][box[:, 3] > w] = w
        box[:, 4][box[:, 4] > h] = h
        box_w = box[:, 3] - box[:, 1]
        box_h = box[:, 4] - box[:, 2]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
        box_data[:len(box)] = box
    if len(box) == 0:
        return []
    if (box_data[:, :4] > 0).any():
        return box_data
    else:
        return []

def load_images_boxes(base_path,imgpath,train_dur,train,split_image_label):
    im_split = imgpath.split('/')
    num_parts = len(im_split)
    im_ind = int(im_split[num_parts - 1][0:5])
    if split_image_label:
        labpath = os.path.join(base_path, 'labels',im_split[0] + '_' + im_split[1] + '_' + '{:05d}.txt'.format(im_ind))
    else:
        #for 'ucf101-24',为了节省空间,不把标签放一起
        labpath = base_path+'/'+'labels'+'/'+imgpath
    img_folder = os.path.join(base_path, 'rgb-images', im_split[0],im_split[1])

    max_num = len(os.listdir(img_folder))
    if train_dur<=max_num:
        if train:
            #最大间隔采样为3
            step = min(random.randint(1, max_num//train_dur),3)
        else:
            step = min(max_num//train_dur,2)
            # step = 1

        key_min = train_dur-1-(max_num-im_ind)//step
        key_max = (im_ind - 1)//step
        key_min = max(0,key_min)
        key_max = min(key_max,train_dur-1)
        if train:
            key_frame = random.randint(key_min,key_max)
        else:
            key_frame = key_max
        temp = list(range(im_ind-key_frame*step,im_ind+(train_dur-key_frame)*step, step))
    else:
        #小于就padding

        num_pading = train_dur-max_num
        temp = [1]*num_pading+list(range(1,max_num+1))
        key_frame = im_ind-1 + num_pading
        # temp = list(map(lambda x: (x - 1) % max_num + 1, temp))
    try:
        path_tmp = [os.path.join(img_folder, '{:05d}.png'.format(i)) for i in temp]
        clip = [Image.open(p).convert('RGB') for p in path_tmp]
    except Exception:
        path_tmp = [os.path.join(img_folder, '{:05d}.jpg'.format(i)) for i in temp]
        clip = [Image.open(p).convert('RGB') for p in path_tmp]
    #读取框参数
    box = read_label(labpath)
    box_path = im_split[0] + '_' + im_split[1] + '_' + im_split[2]
    return clip,box,box_path,key_frame

def load_data_detection(base_path,imgpath,train,train_dur,shape,split_image_label=True,
                        jitter=.3,hue=.1,sat=1.5,val=1.5):
    
    clip,box,box_path,key_frame = load_images_boxes(base_path,imgpath,train_dur,train,split_image_label)

    if train:
        # 数据增强
        #获得随机参数
        image_size = clip[-1].size
        parameter = get_rand_para(shape, jitter, hue, sat, val)
        #使用参数对图片进行数据增强
        clip = [get_random_image(img, shape, *parameter) for img in clip]
        #调整相应的框参数
        box = get_random_box(box,  shape, image_size, *parameter[:5])

        return key_frame,clip, box
    else:
        #测试集处理
        #直接resize
        iw, ih = clip[-1].size
        w, h = shape
        clip = [np.array(letterbox_image(img, shape)) for img in clip]
        if len(box) == 0:
            return key_frame,box_path, clip, []
        else:

            scale = min(w / iw, h / ih)
            nw, nh = iw * scale, ih * scale
            box[:, [1, 3]] = box[:, [1, 3]] * scale + (w - nw) // 2
            box[:, [2, 4]] = box[:, [2, 4]] * scale + (h - nh) // 2
            box[:, 1:3][box[:, 1:3] < 0] = 0
            box[:, 3][box[:, 3] > w] = w
            box[:, 4][box[:, 4] > h] = h
            
            return key_frame,box_path, clip, box
