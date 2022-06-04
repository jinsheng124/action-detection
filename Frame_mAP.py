import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import listDataset, dataset_collate_test
from models.yolo import YoloBody
from utils.utils import get_classes, get_anchors
from test import test
if __name__ == "__main__":
    args = {
        "model_path": 'logs/best_jhmdb_21.pt',
        "anchors_path": 'model_data/jhmdb_21_anchors.txt',
        "classes_path": 'model_data/jhmdb_21_classes.txt',
    }
    torch.cuda.empty_cache()
    num_clip = 16
    Cuda = True
    batch_size = 10
    model_image_size=(224,224)
    test_dataset = listDataset('fight',shape=model_image_size,train=False,clip_duration=num_clip,split_image_label=True)
    genval = DataLoader(test_dataset,batch_size=batch_size,
                        num_workers=2,pin_memory=True,
                        collate_fn=dataset_collate_test)
    class_names = get_classes(args["classes_path"])
    anchors = get_anchors(args["anchors_path"])
    net = YoloBody(num_anchors=len(anchors[0]),num_classes=len(class_names),clip=num_clip,c3d="ResNext101").eval()
    net = nn.DataParallel(net)
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args["model_path"], map_location=device)
    net.load_state_dict(state_dict["model"])
    del state_dict
    # net.load_state_dict(state_dict)
    if Cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        net = net.cuda()

    Maplist, Map, recall, precision,valloss = test(net, genval, class_names, anchors,model_image_size=model_image_size,conf_thres=0.5,K_sample=0)
    print("each class ap:")
    print(str(Maplist)[1:-1])
    print("valloss:{:.3f} recall:{:.3f} precision:{:.3f} Map:{:.3f}".format(valloss,recall, precision, Map))
