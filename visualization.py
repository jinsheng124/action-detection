import os
import random
import cv2
from detection import VideoDetect
#初始化检测器
from PIL import Image
import math

def cat_image(path="temp",w_num=4):
    img_path = os.listdir(path)
    img_path = list(filter(lambda x:(x.split(".")[-1] in ["jpg","png"]),img_path))
    w_hum = math.ceil(len(img_path)/w_num)
    unit_w = 320*w_num
    unit_h = 240*w_hum
    target = Image.new("RGB",(unit_w,unit_h))
    for i,p in enumerate(img_path):
        x,y = i%w_num*320,i//w_num*240
        img = Image.open(path+"/"+p)
        target.paste(img,(x,y))
    target.save(path+"/target.png")

def visulization(test_dir ="jhmdb-21/testlist.txt", num_clip = 16,topk = 100):
    detect = VideoDetect(image_size=(224, 224),
                         model_path="logs/best_jhmdb_21.pt",
                         anchors_path='model_data/jhmdb_21_anchors.txt',
                         classes_path='model_data/jhmdb_21_classes.txt',
                         num_clip=num_clip,
                         conf_thres=0.3)
    with open(test_dir, 'r') as file:
        lines = file.readlines()
    less_lines = random.choices(lines,k=topk)

    results = []
    for line in less_lines:
        im_split = line.split('/')
        num_parts = len(im_split)
        im_ind = int(im_split[num_parts - 1][0:5])
        img_folder = os.path.join(test_dir.split('/')[0], 'rgb-images', im_split[0],im_split[1])
        max_num = len(os.listdir(img_folder))
        if num_clip<=max_num:
            # step = 1
            step = min(max_num // num_clip, 2)
            key_min = num_clip-1-(max_num-im_ind)//step
            key_max = (im_ind - 1)//step
            key_min = max(0,key_min)
            key_max = min(key_max,num_clip-1)
            key_frame = key_max
            temp = list(range(im_ind-key_frame*step,im_ind+(num_clip-key_frame)*step, step))
        else:
            #小于就padding
            num_pading = num_clip-max_num
            temp = [1]*num_pading+list(range(1,max_num+1))
            key_frame = im_ind-1 + num_pading
            # temp = list(map(lambda x: (x - 1) % max_num + 1, temp))
        path_tmp = [os.path.join(img_folder, '{:05d}.png'.format(i)) for i in temp]
        clip = [Image.open(p).convert('RGB') for p in path_tmp]
        frame = detect.detect(clip, key_frame=key_frame)
        results.append(frame)
    if not os.path.exists("temp"):
        os.mkdir("temp")
    count = 1
    for result in results:
        cv2.imshow("action", result)
        cv2.waitKey(100)
        cv2.imwrite(f"temp/{count}.png",result,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        count+=1
    cv2.destroyAllWindows()
#-----------------------------#
cat = False
if __name__=="__main__":
    visulization(topk=100)
    if cat:
        cat_image(w_num=4)