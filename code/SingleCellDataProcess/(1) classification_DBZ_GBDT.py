# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:13:24 2023

@author: yutah
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import sys
import pandas as pd
import math

def adjust_train_test(y_train, y_test, train_index, test_index):
    '''
    Adjust training and testing data to ensure there are at least 5 of each in the train and 3 of each in test data
    5 * the average number of samples of each class is sampled
    :param y_train: 训练集标签
    :param y_test: 测试集标签
    :param train_index: 训练集id
    :param test_index: 测试集id
    :return: 新的被调整的训练集和测试集
    '''
    np.random.seed(1)

    # 查找训练集测试集标签的交集
    unique_labels_temp = np.intersect1d(y_train, y_test)
    unique_labels_temp.sort()

    unique_labels  = []
    counter = []
    new_test_index = []
    new_y_test = []
    new_y_train = []

    for l in unique_labels_temp:

        # 根据交集得出的值，查找特定元素取该值的索引
        l_train = np.where(l == y_train)[0]
        l_test = np.where(l == y_test)[0]

        # 从y_test中剔除一部分（很少，一般只有四五个），后面计入训练集里
        if l_train.shape[0] > 5 and l_test.shape[0] > 3:
            unique_labels.append(l)
            new_test_index.append(l_test)   # 获取满足条件的 y_test 的索引
            counter.append(l_train.shape[0])
    new_test_index = np.concatenate((new_test_index))
    new_test_index.sort()

    # 报错点,把new_test_index打印出来看看是啥
    # print(new_test_index,len(new_test_index))
    # print(y_test,len(y_test))

    # 使用 pd.Series.isin() 方法检查每个元素是否在 test_index 中
    new_test_index = test_index[new_test_index]
    mask = y_test.index.isin(new_test_index)

    # 使用布尔索引筛选 y_test，保留在 test_index 中的元素
    new_y_test = y_test[mask]

    new_train_index = []
    avgCount = int(np.ceil(np.mean(counter)))   #sample 5x avgCount
    for l in unique_labels:
        l_train = np.where(l == y_train)[0]
        index = np.random.choice(l_train, 5*avgCount)
        new_train_index.append(index)
    new_train_index = list(set(np.concatenate(new_train_index)))
    new_train_index.sort()
    # print(new_train_index,len(new_train_index))

    new_train_index = train_index[new_train_index]
    mask = y_train.index.isin(new_train_index)
    new_y_train = y_train[mask]

    return new_y_train, new_y_test, new_train_index, new_test_index

def computeSVC(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mySVC = GradientBoostingClassifier(random_state =1)
    mySVC.fit(X_train, y_train)
    y_pred = mySVC.predict(X_test)
    return y_pred

def balanced_accuarcy(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    return ba

# 下面的函数嵌套了前面三个函数
def compute5foldClassification(X_ccp, y_true, max_state = 10):
    BA_ccp = np.zeros(max_state)

    # 10次5折交叉验证
    for state in range( max_state):
        kf = KFold(n_splits=5, shuffle=True, random_state=state)
        ba_ccp = np.zeros(5)
        print('state:',state + 1)

        # 进行一次五折交叉验证
        for i, (train_index, test_index) in enumerate(kf.split(X_ccp)):
            y_train = y_true[train_index]; y_test = y_true[test_index]

            # 训练集测试集调整
            # print(f'调整前：\ny_train:{y_train}\ny_test:{y_test}\ntrain_index:{train_index}\ntest_index:{test_index}\n')
            y_train, y_test, train_index, test_index = adjust_train_test(y_train, y_test, train_index, test_index)
            # print(f'调整后：\ny_train:{y_train}\ny_test:{y_test}\ntrain_index:{train_index}\ntest_index:{test_index}')
            print(f'调整数据集后训练集和测试集的数目：state:{state},fold:{i + 1},train_num:{len(train_index)},test:{len(test_index)}')

            X_ccp_train = X_ccp[train_index]; X_ccp_test = X_ccp[test_index]

            # 训练模型，预测结果
            y_pred = computeSVC(X_ccp_train, X_ccp_test, y_train, y_test)
            # print(y_pred)

            # 计算BA值
            ba_ccp[i] = balanced_accuarcy(y_true[test_index], y_pred)

        BA_ccp[state] = np.mean(ba_ccp)

    print(BA_ccp)
    BA_value = np.mean(BA_ccp)

    # best_random_seed = list(BA_ccp).index(BA_value)

    return BA_value

def main():
    couple = float(sys.argv[1])
    dataset = sys.argv[2]
    print('couple:', couple, flush=True)

    # 特征矩阵
    path_feature = '/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/GSE/data/feature/%s_feature_combined_couple_%.3f.npy' % (
    dataset, couple)
    # path_feature = 'GSE84133human4_feature_combined_couple_1.000.npy'
    data_feature = np.load(path_feature)
    print('size of data_feature:', np.shape(data_feature), flush=True)  # (300, 30)

    # 标签
    path_label = '/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/GSE/data/%s_full_labels.csv' % dataset
    # path_label = r'D:\python\result\b因子系列\GSE\GSE84133\GSE84133human4/GSE84133human4_full_labels.csv'
    df = pd.read_csv(path_label)  # 不读取列名
    # 读取第三列标签数据
    data_label = df.iloc[:, 2]
    print('size of data_label:', np.shape(data_label), flush=True)

    balanced_acc = compute5foldClassification(data_feature,data_label)
    print(f'Balanced Accuracy: {balanced_acc}', flush=True)
    # print(f'best random seed:{best_random_seed}')

if __name__ == '__main__':
    main()
