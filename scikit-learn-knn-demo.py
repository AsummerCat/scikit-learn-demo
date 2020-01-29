# -*- coding: utf-8 -*-
'''
近邻算法
'''
from sklearn import neighbors
import numpy as np

'''
knn预测算法
'''


def knn_sk_learn_predict():
    # 加载  算法 knn分类算法
    knn = neighbors.KNeighborsClassifier()
    # 获取数据集 类标签
    dataset, labels = create_dataset()
    # 训练模型
    knn.fit(dataset, labels)
    # 分类预测
    predictRes = knn.predict(np.array([[2, 4, 0]]))
    print("输出预测结果", predictRes)


# 创建数据集 返回特征向量 和类标签
def create_dataset():
    # 特征向量 数据集合 一般情况下 是经过数据预处理后生成的特征向量
    # 冰淇淋数量 喝水数量 户外活动时长
    dataset = np.array([[8, 4, 2], [7, 1, 1], [1, 4, 4], [3, 0, 5], [1, 5, 5]])
    # 类标签
    # 冷热程度
    labels = ['非常热', '非常热', '一般热', '一般热', '一般热']
    return dataset, labels


if __name__ == '__main__':
    knn_sk_learn_predict()
