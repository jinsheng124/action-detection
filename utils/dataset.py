import random
import numpy as np
from torch.utils.data import Dataset
from utils.clip import load_data_detection
class listDataset(Dataset):
    # clip duration = 8, i.e, for each time 8 frames are considered together
    def __init__(self,
                 base,
                 shape=None,
                 train=False,
                 clip_duration=8,
                 split_image_label = True):

        self.clip_duration = clip_duration
        if train:
            with open(base+'/trainlist.txt', 'r') as file:
                self.lines = file.readlines()
            random.shuffle(self.lines)
        else:
            with open(base+'/testlist.txt', 'r') as file:
                self.lines = file.readlines()

        self.base_path = base
        self.nSamples = len(self.lines)
        self.train = train
        self.shape = shape
        self.split_image_label = split_image_label
    def xyxy2normal(self,y,normal = True):
        boxes = np.array(y[:, 1:5], dtype=np.float32)
        boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
        boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2
        if normal:
            _scale = np.array([self.shape[0],self.shape[1],self.shape[0],self.shape[1]])
            boxes/=_scale
            boxes = np.maximum(np.minimum(boxes, 1), 0)
        # angle = y[:,5:6]
        y = np.concatenate([boxes, y[:, :1]], axis=-1)
        return y
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        jitter = 0.3
        hue = 0.1
        sat = 1.5
        val = 1.5
        if self.train:  # For Training
            keyframe,clip, y = load_data_detection(self.base_path, imgpath, self.train,self.clip_duration, self.shape, self.split_image_label,jitter, hue, sat,val)

        else:  # For Testing
            keyframe,box_path,clip, y = load_data_detection(self.base_path, imgpath, self.train,self.clip_duration, self.shape,self.split_image_label)
        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            y = self.xyxy2normal(y,normal=True)
        label = np.array(y, dtype=np.float32)
        clip = np.stack(clip) / 255.0
        clip = clip.transpose((3, 0, 1, 2))
        if self.train:
            return (keyframe,clip, label)
        else:
            return (keyframe,box_path,clip, label)


def dataset_collate_train(batch):
    images = []
    bboxes = []
    key_frames = []
    for key,img, box in batch:
        images.append(img)
        bboxes.append(box)
        key_frames.append(key)
    images = np.stack(images)
    return key_frames,images, bboxes
def dataset_collate_test(batch):
    images = []
    bboxes = []
    frame_id=[]
    key_frames = []
    for key,idx,img, box in batch:
        images.append(img)
        bboxes.append(box)
        frame_id.append(idx)
        key_frames.append(key)
    images = np.stack(images)
    return key_frames,frame_id,images, bboxes
