# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

'''
决策树
'''
'''
数据预处理
数据建模
数据验证
'''


def test():
    # 数据预处理
    '''
    这边鸢尾已经在scikit库中 直接调用
    '''
    # 直接引入鸢尾iris数据集
    from sklearn.datasets import load_iris
    # data:属性 target:标注
    iris = load_iris()

    # 获取属性
    # print(len(iris["data"]))
    # 因为这边数据是比较规则的 不用预处理

    # 预处理
    ## 需要导入模块
    from sklearn.model_selection import train_test_split
    ## 将数据分为测试数据 和验证数据   test_size表示测试数据占比20% random_state=1 随机选择30个数据
    train_data, test_data, tarin_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2,
                                                                        random_state=1)

    # 建立模型
    ## 需要导入模块
    from sklearn import tree
    ## 建立决策树 分类器 (回归器:DecisionTreeRegressor)    criterion:标准选择 信息熵
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    # 建立模型 ->训练集训练
    clf.fit(train_data, tarin_target)
    # 进行预测
    y_pred = clf.predict(test_data)

    # 验证
    ## 需要导入模块
    from sklearn import metrics
    ## 两种方式

    ### 准确率
    #### y_true 验证的数据  y_pred:预测值
    ##### 输出结果就是 准确率
    print(metrics.accuracy_score(y_true=test_target, y_pred=y_pred))

    ### 混淆矩阵->误差矩阵
    print(metrics.confusion_matrix(y_true=test_target, y_pred=y_pred))

    ## 决策树输出结构文件
    with open("./data/tree.dot", "w") as fw:
        tree.export_graphviz(clf, out_file=fw)


if __name__ == '__main__':
    test()
