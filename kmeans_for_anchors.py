import numpy as np
import glob
import os


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean(
        [np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # 取出一共有多少框
    row = box.shape[0]

    # 每个框各个点的位置
    distance = np.empty((row, k))

    # 最后的聚类位置
    last_clu = np.zeros((row, ))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # 取出最小点
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near

    return cluster


def read_label(labpath):
    box = []
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return box
        bs = np.reshape(bs, (-1, 5))
        num = bs.shape[0]
        box = np.zeros((num, 2))
        box[:, 0] = (bs[:, 3] - bs[:, 1]) / 320
        box[:, 1] = (bs[:, 4] - bs[:, 2]) / 240
    return box


def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for txt_file in glob.glob('{}/*txt'.format(path)):
        box = read_label(txt_file)
        if len(box) != 0:
            data.append(box)
    return np.concatenate(data, axis=0)

def load_data_ucf(path):
    data = []
    with open("ucf-24/testlist.txt", 'r') as file:
        test_lines = file.readlines()
    with open("ucf-24/trainlist.txt", 'r') as file:
        train_lines = file.readlines()
    for lines in [test_lines,train_lines]:
        for l in lines:
            label_path = path+"/"+l.rstrip()
            box = read_label(label_path)
            if len(box) != 0:
                data.append(box)
    return np.concatenate(data, axis=0)


    
if __name__ == '__main__':
    # 会生成yolo_anchors.txt
    SIZE = 224
    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    path = r'./fight/labels'

    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data(path)
    print(data.shape)
    # 使用k聚类算法
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out * SIZE)
    data = out * SIZE
    f = open("model_data/fight_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()