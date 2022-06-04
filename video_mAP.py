import glob
from torch.utils.data import Dataset
from core.eval_results import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.yolo import YoloBody
from utils.utils import get_classes, get_anchors
import utils.utils as tools
from utils.clip import load_data_detection,read_label
from tqdm import tqdm
from scipy.io import loadmat
import joblib
# ---------------------------------------------------------------
#K_sample参数
dataset_use = "jhmdb-21"

if dataset_use=="jhmdb-21":
    args = {
        "model_path": 'logs/best_jhmdb_21.pt',
        "anchors_path": 'model_data/jhmdb_21_anchors.txt',
        "classes_path": 'model_data/jhmdb_21_classes.txt',
        "num_clip":16,
        "batch_size":16,
        "K_sample":0,
        "model_image_size":(224,224)
    }
elif dataset_use=="ucf-24":
    args = {
        "model_path": 'logs/best_ucf_24.pt',
        "anchors_path": 'model_data/ucf_24_anchors.txt',
        "classes_path": 'model_data/ucf_24_classes.txt',
        "K_sample": 0,
        "num_clip": 16,
        "batch_size": 16,
        "model_image_size": (224, 224)
    }
# Test parameters
conf_thresh   = 0.005
nms_thresh    = 0.4
eps           = 1e-5
def load_model(model_path,class_names,anchors,num_clip):
    # 模型加载
    model = YoloBody(num_anchors=len(anchors[0]), num_classes=len(class_names), clip=num_clip, c3d="ResNext101").eval()
    model = nn.DataParallel(model)
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    del state_dict
    model = model.cuda()
    return model
def dataset_collate(batch):
    images = []
    bboxes = []
    key_frames = []
    img_idx = []
    for key,img,box,img_name in batch:
        images.append(img)
        bboxes.append(box)
        key_frames.append(key)
        img_idx.append(img_name)
    images = np.stack(images)
    return key_frames,images, bboxes,img_idx
class testData(Dataset):
    def __init__(self, root, shape=None, clip_duration=16,dataset_use = "jhmdb-21"):

        self.root = root
        if dataset_use=="jhmdb-21":
            self.label_paths = sorted(glob.glob(os.path.join(root, '*.png')))
        elif dataset_use=="ucf-24":
            self.label_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
        else:
            raise ValueError("none type dataset!!!")

        self.shape = shape
        self.clip_duration = clip_duration
        self.dataset_use = dataset_use

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label_path = self.label_paths[index].replace("\\",'/')
        #------------#
        im_split = label_path.split('/')
        num_parts = len(im_split)
        class_name = im_split[-3]
        file_name = im_split[-2]
        im_ind = int(im_split[num_parts - 1][0:5])

        #---------------------#
        if self.dataset_use=="jhmdb-21":
            img_name = os.path.join(class_name, file_name, '{:05d}.png'.format(im_ind)).replace("\\", "/")
            keyframe,box_path,clip, _ = load_data_detection("jhmdb-21",img_name,train =False,
                                train_dur=self.clip_duration,shape = self.shape,split_image_label=True)
            txt_path = "jhmdb-21/labels/"+box_path.split('.')[0]+".txt"
            label = read_label(txt_path)
        elif self.dataset_use=="ucf-24":
            img_name = os.path.join(class_name, file_name, '{:05d}.jpg'.format(im_ind)).replace("\\", "/")
            keyframe,box_path,clip, _ = load_data_detection("ucf-24",img_name,train =False,
                                train_dur=self.clip_duration,shape = self.shape,split_image_label=False)
            txt_path = label_path.split('.')[0]+".txt"
            txt_path = txt_path.replace("rgb-images","labels")
            label = read_label(txt_path)
        else:
            raise ValueError("none type dataset!!!")
        clip = np.stack(clip) / 255.0
        clip = clip.transpose((3, 0, 1, 2))
        return (keyframe, clip, label,img_name)

####### Create model
# ---------------------------------------------------------------

def get_video_mAP(root_path = "jhmdb-21",save_result = False):
    """
    Calculate video_mAP over the test set
    """
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    detected_boxes = {}
    gt_videos = {}
    if root_path=="jhmdb-21":
        CLASSES = ('brush_hair', 'catch', 'clap', 'climb_stairs', 'golf',
                        'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                        'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                        'stand', 'swing_baseball', 'throw', 'walk', 'wave')
        with open("jhmdb-21/testlist_video.txt", 'r') as file:
            lines = file.readlines()
    elif root_path=="ucf-24":
        CLASSES = ('Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
                   'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
                   'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
                   'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
                   'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')
        with open("ucf-24/testlist_video.txt", 'r') as file:
            lines = file.readlines()
        lines = list(map(lambda x:x.rstrip(),lines))
        gt_data = loadmat('model_data/ucf24_finalAnnots.mat')['annot']
        n_videos = gt_data.shape[1]
        for i in range(n_videos):
            video_name = gt_data[0][i][1][0]
            if video_name in lines:
                n_tubes = len(gt_data[0][i][2][0])
                v_annotation = {}
                all_gt_boxes = []
                for j in range(n_tubes):
                    gt_one_tube = []
                    tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                    tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                    tube_class = gt_data[0][i][2][0][j][2][0][0]
                    tube_data = gt_data[0][i][2][0][j][3]
                    tube_length = tube_end_frame - tube_start_frame + 1

                    for k in range(tube_length):
                        gt_boxes = []
                        # gt_boxes.append(int(tube_start_frame + k))
                        gt_boxes.append(int(k+1))
                        gt_boxes.append(float(tube_data[k][0]))
                        gt_boxes.append(float(tube_data[k][1]))
                        gt_boxes.append(float(tube_data[k][0]) + float(tube_data[k][2]))
                        gt_boxes.append(float(tube_data[k][1]) + float(tube_data[k][3]))
                        gt_one_tube.append(gt_boxes)
                    all_gt_boxes.append(gt_one_tube)

                v_annotation['gt_classes'] = tube_class
                v_annotation['tubes'] = np.array(all_gt_boxes)
                gt_videos[video_name] = v_annotation
    class_names = get_classes(args["classes_path"])
    anchors = get_anchors(args["anchors_path"])
    num_classes = len(CLASSES)
    model = load_model(args["model_path"],class_names,anchors,args["num_clip"])
    yolo_decodes = tools.DecodeBox(anchors, len(class_names),args["model_image_size"],K_sample=args["K_sample"])
    pbar = tqdm(lines)
    for line in pbar:
        pbar.set_postfix(**{"cur_video":line})
        line = line.rstrip()
        test_loader = torch.utils.data.DataLoader(
            testData(os.path.join(root_path, 'rgb-images', line),shape=args["model_image_size"], clip_duration=args["num_clip"],dataset_use=root_path),
            batch_size=args["batch_size"],shuffle=False,num_workers=0,collate_fn=dataset_collate)
        video_name = ''
        v_annotation = {}
        all_gt_boxes = []
        t_label = -1
        for batch_idx, (key_frames, images, labels,img_name) in enumerate(test_loader):
            if video_name == '':
                path_split = img_name[0].split('/')
                video_name = os.path.join(path_split[0], path_split[1]).replace("\\",'/')
            with torch.no_grad():
                images = torch.from_numpy(images).float().to(devices)
                key_image = [images[i, :, key, :, :] for i, key in enumerate(key_frames)]
                key_image = torch.stack(key_image, dim=0)
                output = model(images, key_image)
                all_boxes = yolo_decodes(output)
                all_boxes = tools.video_nms(all_boxes, conf_thres=conf_thresh, nms_thres=nms_thresh)
                for i,bboxes in enumerate(all_boxes):
                    if bboxes is None or not len(labels[i]):
                        continue
                    # if bboxes is None:
                    #     continue
                    bboxes = bboxes.data.cpu()
                    # o_conf = np.array(bboxes[:, 4:5])
                    c_conf = np.array(bboxes[:, 5:])
                    boxes = bboxes[:, :4]
                    boxes = tools.correct_boxes(boxes, args["model_image_size"], (320,240))
                    img_annotation = {}
                    # generate detected tubes for all classes
                    # save format: {img_name: {cls_ind: array[[x1,y1,x2,y2, cls_score], [], ...]}}
                    for c in range(num_classes):
                        cls_boxes = np.concatenate((boxes,c_conf[:,c:1+c]),axis=1)
                        img_annotation[c+1] = cls_boxes
                    detected_boxes[img_name[i]] = img_annotation
                    if root_path == "jhmdb-21":
                        target = labels[i]
                        gt_boxes = []
                        num_gts = len(target)
                        if t_label == -1:
                            t_label = int(target[0][0])+1
                        for g in range(num_gts):
                            path_split = img_name[i].split('/')
                            gt_boxes.append(int(path_split[2][:5]))
                            gt_boxes.append(target[g][1])
                            gt_boxes.append(target[g][2])
                            gt_boxes.append(target[g][3])
                            gt_boxes.append(target[g][4])
                            all_gt_boxes.append(gt_boxes)
        # generate corresponding gts
        # save format: {v_name: {tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}}
        if root_path =="jhmdb-21":
            v_annotation['gt_classes'] = t_label
            v_annotation['tubes'] = np.expand_dims(np.array(all_gt_boxes), axis=0)
            gt_videos[video_name] = v_annotation

    iou_list = [0.2, 0.5, 0.75]
    if save_result:
        with open('logs/video_mAP_result.pkl','wb') as f:
            bags = {"gt_videos":gt_videos,"detected_boxes":detected_boxes,"classes":CLASSES}
            joblib.dump(bags,f)
    for iou_th in iou_list:
        ap_list = evaluate_videoAP(gt_videos, detected_boxes, CLASSES, iou_th, True)
        print("each AP: ",[round(k,3) for k in ap_list])
        print(f'video mAP@{iou_th}: ',sum(ap_list)/len(ap_list))
        print()

def compute_pickle(path):
    if path.endswith("pkl"):
        with open(path,'rb') as f:
            data = joblib.load(f)
        iou_list = [0.2, 0.5, 0.75]
        for iou_th in iou_list:
            ap_list = evaluate_videoAP(data["gt_videos"], data["detected_boxes"], data["classes"], iou_th, True)
            print("each AP: ",[round(k,3) for k in ap_list])
            print(f'video mAP@{iou_th}: ', sum(ap_list) / len(ap_list))
            print()

if __name__ == '__main__':
    use_pickle = False
    path = 'logs/video_mAP_result.pkl'
    if use_pickle and os.path.isfile(path):
        compute_pickle(path)
    else:
        get_video_mAP(root_path = dataset_use,save_result=False)

