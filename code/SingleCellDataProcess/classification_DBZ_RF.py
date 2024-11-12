# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:13:24 2023

@author: yutah
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import sys
import pandas as pd
import math
import copy

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

def adjust_xy(X,Y,label):
    np.set_printoptions(suppress=True, threshold=np.inf)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    new_X,new_Y = copy.deepcopy(X),copy.deepcopy(Y)
    for i in label:
        new_X = np.array([X[j] for j in range(len(X)) if Y[j] != i])
        new_Y = pd.Series([Y[j] for j in range(len(X)) if Y[j] != i])
        X,Y = copy.deepcopy(new_X),copy.deepcopy(new_Y)
    return new_X,new_Y

def computeSVC(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mySVC = RandomForestClassifier(random_state =1)
    mySVC.fit(X_train, y_train)
    y_pred = mySVC.predict(X_test)
    return y_pred

def balanced_accuarcy(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    return ba

def expand_pro_matrix(y_pred_pro,y_true,y_pred):
    y_pred_pro_new = [[] for _ in range(len(y_pred_pro))]
    pro_exist = {_: False for _ in range(max(y_true) + 1)}
    for sort in y_pred:
        pro_exist[sort] = True
    for sample in range(len(y_pred_pro_new)):
        exist_sort = 0
        for sort in range(max(y_true) + 1):
            if pro_exist[sort] == True:
                y_pred_pro_new[sample].append(y_pred_pro[sample][exist_sort])
                exist_sort += 1
            else:
                y_pred_pro_new[sample].append(0.0)
    return y_pred_pro_new

# 下面的函数嵌套了前面三个函数
def compute5foldClassification(X_ccp, y_true, dataset, couple,max_state = 10):
    BA_ccp = np.zeros(max_state)

    pro = np.zeros((len(y_true), max(y_true) + 1))

    # 10次5折交叉验证
    for state in range(max_state):
        kf = KFold(n_splits=5, shuffle=True, random_state=state)
        ba_ccp = np.zeros(5)
        y_pred_list = np.zeros(y_true.shape)
        y_pred_list_pro = np.zeros((len(y_true),max(y_true) + 1))

        # 进行一次五折交叉验证
        for i, (train_index, test_index) in enumerate(kf.split(X_ccp)):
            y_train = y_true[train_index]; y_test = y_true[test_index]
            y_train, y_test, train_index, test_index = adjust_train_test(y_train, y_test, train_index, test_index)
            # print(f'调整数据集后训练集和测试集的数目：state:{state},fold:{i + 1},train_num:{len(train_index)},test_num:{len(test_index)}')

            X_ccp_train = X_ccp[train_index]; X_ccp_test = X_ccp[test_index]

            # 训练模型，预测结果
            scaler = StandardScaler()
            scaler.fit(X_ccp_train)
            X_ccp_train = scaler.transform(X_ccp_train)
            X_ccp_test = scaler.transform(X_ccp_test)
            mySVC = RandomForestClassifier(random_state =1)
            mySVC.fit(X_ccp_train, y_train)

            y_pred = mySVC.predict(X_ccp_test)
            # y_pred_list[test_index] = mySVC.predict(X_ccp_test)
            y_pred_pro = mySVC.predict_proba(X_ccp_test)  # 预测验证集上每个样本属于每个类别的概率

            # 扩充概率矩阵
            y_pred_pro = expand_pro_matrix(y_pred_pro,y_true,y_pred)

            # 概率矩阵拼接
            y_pred_list_pro[test_index] = y_pred_pro

            # y_pred_list_pro[test_index] = mySVC.predict_proba(X_ccp_test)[:,1]  # 将模型对测试集的每个样本属于类别1（通常是正例类别）的概率存储在名为GBDT_results_pro的数组中的特定索引（由test_idx指定）
            # print(y_pred,len(y_pred))

            # 计算折内BA值
            # ba_ccp[i] = balanced_accuarcy(y_true[test_index], y_pred)

        # print(y_pred_list_pro)
        for i in range(len(pro)):
            for j in range(len(pro[i])):
                pro[i][j] += y_pred_list_pro[i][j]
        #print(pro)

        # 一次五折交叉验证的BA值
        # BA_ccp[state] = np.mean(ba_ccp)

        # BA_ccp[state] = balanced_accuarcy(y_true, y_pred_list)
        #print(BA_ccp[state])

    pro = [[round(float(x / 10),3) for x in row] for row in pro]
    # print(pro)

    # 十次五折交叉验证的BA值
    # BA_value = np.mean(BA_ccp)

    # 基于概率平均的BA值
    pred_label = np.zeros(len(y_true))
    for sample in range(len(pro)):
        max_proba = max(pro[sample])
        pred_label[sample] = pro[sample].index(max_proba)
    # print(pred_label,y_true)
    BA_value = balanced_accuarcy(y_true, pred_label)
    np.save(r'D:\python\result\RS及UMAP聚类\pred_label/%s_RF_pred_label_results_%.3f.npy' % (dataset, couple),
            pred_label)

    return BA_value

def main():
    couple = 10
    dataset = 'GSE84133mouse2'
    #print('couple:', couple, flush=True)

    # 特征矩阵
    # path_feature = '/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/GSE/data/feature/%s_feature_combined_couple_%.3f.npy' % (
    # dataset, couple)
    path_feature = r'D:\python\result\RS及UMAP聚类\npy/%s_feature_combined_couple_%.3f.npy'%(dataset,couple)
    data_feature = np.load(path_feature)
    # print(data_feature)
    # print('size of data_feature:', np.shape(data_feature), flush=True)  # (300, 30)

    # 标签
    path_label = r'D:\python\result\RS及UMAP聚类\label/%s_full_labels.csv' % dataset
    # path_label = r'D:\python\result\b因子系列\GSE\GSE84133\GSE84133human3/GSE84133human3_full_labels.csv'
    df = pd.read_csv(path_label)  # 不读取列名
    # 读取第三列标签数据
    data_label = df.iloc[:, 2]
    #print('size of data_label:', np.shape(data_label), flush=True)

    data_feature, data_label = adjust_xy(data_feature, data_label, [0,1,9,12])
    data_label = pd.Series(list(data_label))

    balanced_acc = compute5foldClassification(data_feature,data_label,dataset,couple)
    print(f'Balanced Accuracy: {balanced_acc}', flush=True)
    # print(f'best random seed:{best_random_seed}')


if __name__ == '__main__':
    main()