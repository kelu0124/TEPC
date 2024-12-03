import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import pandas as pd
import math
import copy
#
def knn_cla(X,Y):
    '''K-nearest neighbors classification'''

    knn = KNeighborsClassifier(n_neighbors=3)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    seeds = np.arange(0,10)
    scores = []

    # 进行十次五折交叉验证
    for seed in seeds:
        # 重新初始化交叉验证对象
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        # 执行交叉验证
        score = cross_val_score(knn, X, Y, cv=cv, scoring='accuracy').mean()
        scores.append(score)
    average_accuracy = np.mean(scores)
    print(f"KNN Average Accuracy: {average_accuracy:.2f}")

def SRG_cla(X,Y):
    seeds = np.arange(0, 10)
    svm_scores = []
    rf_scores = []
    gbdt_scores = []
    # 对每个随机种子重复交叉验证
    for seed in seeds:
        # 设置模型和随机状态
        svm_model = SVC(random_state=seed)
        rf_model = RandomForestClassifier(random_state=seed)
        gbdt_model = GradientBoostingClassifier(random_state=seed)

        # 执行5折交叉验证并添加分数到列表
        svm_scores.append(np.mean(cross_val_score(svm_model, X, Y, cv=5, scoring='accuracy', n_jobs=-1)))
        rf_scores.append(np.mean(cross_val_score(rf_model, X, Y, cv=5, scoring='accuracy', n_jobs=-1)))
        gbdt_scores.append(np.mean(cross_val_score(gbdt_model, X, Y, cv=5, scoring='accuracy', n_jobs=-1)))
    # 打印每个模型的平均分数
    print(f"SVM Average Accuracy: {np.mean(svm_scores):.2f}")
    print(f"RF Average Accuracy: {np.mean(rf_scores):.2f}")
    print(f"GBDT Average Accuracy: {np.mean(gbdt_scores):.2f}")


# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:13:24 2023

@author: yutah
"""

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

def compute5foldClassification(X_ccp, y_true, dataset,max_state = 10):
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
            # y_train, y_test, train_index, test_index = adjust_train_test(y_train, y_test, train_index, test_index)
            # print(f'调整数据集后训练集和测试集的数目：state:{state},fold:{i + 1},train_num:{len(train_index)},test_num:{len(test_index)}')

            X_ccp_train = X_ccp[train_index]; X_ccp_test = X_ccp[test_index]

            # 训练模型，预测结果
            scaler = StandardScaler()
            scaler.fit(X_ccp_train)
            X_ccp_train = scaler.transform(X_ccp_train)
            X_ccp_test = scaler.transform(X_ccp_test)
            mySVC = KNeighborsClassifier() # 设置SVC时要加参数probability=True
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

    accuracy = accuracy_score(y_true, pred_label)
    print(f'Accuracy: {accuracy}', flush=True)

    # np.save(r'D:\python\result\RS及UMAP聚类\pred_label/%s_RF_pred_label_results_%.3f.npy' % (dataset, couple,cutoff),
    #         pred_label)

    return BA_value

def main():
    # couples = np.arange(1,31,1)
    # couples = np.array([1, 2, 6])
    # for couple in couples:
    dataset = 'allaml'

    # 特征矩阵
    # path_feature = r'D:\python\result\allaml\feature\rossler/%s_feature_combined_couple_%.3f.npy'%(dataset,couple)
    #data_feature = np.load(r'D:\python\result\allaml\origin_data/ALLAML.npy')
    data_feature = np.load(r'/public/home/chenlong666/desktop/my_desk1/coil_20/feature/rossler/allaml_feature_combined_couple_1.000.npy')
    #data_feature = np.load(r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/ALLAML.npy')
    print(data_feature.shape)

    umap_reducer = UMAP(n_neighbors=10, min_dist=0.1, random_state=42, n_components=3)
    umap_data = umap_reducer.fit_transform(data_feature)
    X = np.array(umap_data)
    print('umap降维后分类精度：')
    #np.save(r'C:\Users\administered\Desktop\coil_20',X)
    np.save(r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/coil_20_reduced.npy', X)

    # 标签
    #path_label = r'D:\python\result\allaml\origin_data/ALLAML_labels.npy'
    #path_label = r'C:\Users\administered\Desktop\coil_20\origin_data\ALLAML_labels.npy'
    path_label = r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/ALLAML_labels.npy'
    data_label = np.load(path_label)
    print(data_label)
    print('size of data_label:', np.shape(data_label), flush=True)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, data_label, test_size=0.3, random_state=42)
    # 创建随机森林分类器实例
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # 训练模型
    rf_clf.fit(X_train, y_train)
    # 使用模型进行预测
    y_pred = rf_clf.predict(X_test)
    # 计算并打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    balanced_acc = balanced_accuarcy(y_test, y_pred)

    #balanced_acc = compute5foldClassification(X,data_label,dataset)
    print(f'Balanced Accuracy: {balanced_acc}', flush=True)
    # print(f'best random seed:{best_random_seed}')

if __name__ == '__main__':
    main()
