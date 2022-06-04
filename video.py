import time
from PIL import Image
import cv2
import numpy as np
from detection import VideoDetect
#初始化检测器
num_clip = 16
detect = VideoDetect(image_size=(224, 224),
                     model_path="logs/best_jhmdb_21.pt",
                     anchors_path='model_data/jhmdb_21_anchors.txt',
                     classes_path='model_data/jhmdb_21_classes.txt',
                     num_clip=16,
                     conf_thres=0.3,
                     K_sample=0)
num_sample = 1
stride = 2
vs = cv2.VideoCapture("fight_video/fi145.mp4")
save_file = True
vid_writer = None
# W = round(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
# H = round(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
#FPS=vs.get(cv2.CAP_PROP_FPS)
#print(FPS)
stack = []
fps = 0.0
count_frame = 0
escape_time = 0
if __name__ == "__main__":
    #检测
    t1 = time.time()
    index = 1
    while True:
        t0 = time.time()
        ret, frame = vs.read()
        read_time = time.time() - t0
        escape_time += read_time
        if frame is None:
            break
        # frame = cv2.resize(frame,(0,0),fx=0.6,fy=0.6,interpolation=cv2.INTER_AREA)
        # frame = cv2.GaussianBlur(frame,(3,3),0)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        image = Image.fromarray(np.uint8(image))
        if count_frame%stride == 0:
            stack.append(image)
        else:
            time.sleep(0.02)
            escape_time += 0.02
        if len(stack) == num_clip:
            frame = detect.detect(stack,key_frame=-1)
            fps = (num_sample*stride)/(time.time() - t1 - escape_time)
            for i in range(num_sample):
                stack.pop(0)
            t1 = time.time()
            escape_time = 0.
        text = "fps= %.2f" % (fps)
        frame = cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
        if save_file:
            cv2.imwrite(f"video_demo/{count_frame}.jpg",frame)
            if not vid_writer:
                vid_writer = cv2.VideoWriter("video_demo/action.avi",
                                             cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 10,
                                             (frame.shape[1], frame.shape[0]))
            vid_writer.write(frame.astype(np.uint8))
        cv2.imshow("action", frame)
        count_frame += 1
        if cv2.waitKey(1) == 27:
            break
