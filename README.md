#**This is an improved network of Spatio-temporal Action Location based on YOWO(基于YOWO改进的时空动作定位网络)**

Note： 

1, train.py中的K_sample = 0 和 K_sample=1的正负样本分配策略差别很大,训练、测试和预测时此参数要保持一致。

2,  [[3D预训练权重下载]](https://github.com/okankop/Efficient-3DCNNs)

4, model_data文件夹下的jhmdb_21_anchors.txt和ucf_24_anchors.txt(先验框)
最好用kmeans_for_anchors.py生成

5,自建数据集格式最好和jhmdb-21文件夹下数据集格式一致(或者自己写数据预处理代码)

6, jhmdb-21和ucf-24数据集标签都是从1开始的，如果你的数据集从0开始，请注释掉
utils/clip.py中的第20行以及test.py的104行。

