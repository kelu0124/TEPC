# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:30:36 2022

@author: yutah
"""

import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
import copy
    
def rs_score(X, y, metric = 'euclidean'):
    '''
    This is rs-score for clustering, or full data
        Input:
            X: np.array data matrix. Rows are the samples, columns are the features
            y: np.array true labels
            metric: used in computing distance
    '''
    distance = pairwise_distances(X, metric = metric)  #compute distance
    distance = distance / np.max(distance)  #scale the distance0
    labels = list(set(list(y)))   #get the classes
    rs_score = np.zeros([X.shape[0], 2])  #initialize
    for label in labels:  #run through the labels
        label_idx = np.where(y == label)[0]  #find the current label's index
        not_label_idx = np.where(y != label)[0]  #find index of everything no in the current label
        
        #Computing residue scores
        residue = distance[label_idx, :].copy()  
        residue = residue[:, not_label_idx]
        residue = np.sum(residue, axis = 1)
        rs_score[label_idx, 0] = residue
        
        #Similarity score
        similarity = distance[label_idx, :].copy()
        similarity = similarity[:, label_idx]
        similarity = 1-similarity
        rs_score[label_idx, 1] = np.sum(similarity, axis = 1)/(label_idx.shape[0]-1)
    # print(y)
    rs_score[:, 0] = rs_score[:, 0] / np.max(rs_score[:, 0])
    return rs_score
    
    
def rs_score_train_test(X_train, X_test, y_train, y_test, metric = 'euclidean'):
    '''
    This version is used for rs-score for classification data.
    This computes the rs-score for the testing data 
    y_train, y_test should be true labels, if you are plotting each class
        Input:
            X_train: training data
            X_test: testing data
            y_train: training label 
            y_test: testing label
        Output:
            rs_score: [residue, similarity] for the testing data
    '''
    
    distance = pairwise_distances(X_test, X_train)  #distance between trainig and testing
    distance = distance / np.max(distance)  #normalize
    labels = list(set(list(y_train)))  #get all the class
    rs_score = np.zeros([X_test.shape[0], 2])  
    
    for label in labels: 
        label_idx = np.where(y_train == label)[0]  #index for the y_train that is part of the current label
        label_test_idx = np.where(y_test == label)[0]  #index of y_test that is part of current label
        not_label_idx = np.where(y_train != label)[0]  #index of y_train not in the current label
        
        #reside score of testing data
        residue = distance[label_test_idx, :].copy()
        residue = residue[:, not_label_idx]
        residue = np.sum(residue, axis = 1)
        rs_score[label_test_idx, 0] = residue
        
        #similarity score of testing data
        similarity = distance[label_test_idx, :].copy()
        similarity = similarity[:, label_idx]
        similarity = 1-similarity
        rs_score[label_test_idx, 1] = np.mean(similarity, axis = 1)
    
    rs_score[:, 0] = rs_score[:, 0] / np.max(rs_score[:, 0]) #Normalize the residue score
    
    return rs_score

# path = '/mnt/ufs18/home-192/jiangj33/opium/hDAT/'

# feature_matrix_PL = np.load(path+'hDAT_uptake_feature_CHON_rips_T_6[38](10).npy',allow_pickle=True)
# feature_matrix_BT = np.load(path+'feature_matrix_bt_fps_hDAT_uptake_chembl27_pubchem_zinc_512_cla.npy',allow_pickle=True)

# temp1 = np.load(path + 'feature_CHON_rips_L_1_42.npy',allow_pickle=True)
# temp2 = np.load(path + 'feature_rips_L_Cl.npy',allow_pickle=True)
# temp3 = np.load(path + 'feature_rips_L_F.npy',allow_pickle=True)
# temp4 = np.load(path + 'feature_rips_L_S.npy',allow_pickle=True)
# feature_matrix_PL = np.concatenate((temp1,temp2,temp3,temp4),axis=1)
# feature_matrix_BT = np.load(path + 'all_DAT_uptake_cla(chembl_27_pubchem_zinc_512)_features.npy',allow_pickle=True)

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

dataset = 'GSE84133mouse2'
couple = 10
feature_metrix = np.load(r'D:\python\result\RS及UMAP聚类\npy/%s_feature_combined_couple_%.3f.npy'%(dataset,couple))

# y_val = np.load(path+'label_hDAT_uptake_cla.npy',allow_pickle=True)
# y_val = np.load(path+'label_all-DAT_uptake_cla.npy',allow_pickle=True)
y_true = pd.read_csv(r'D:\python\result\RS及UMAP聚类\label\%s_full_labels.csv'%dataset)

# feature_metrix, y_true = adjust_xy(feature_metrix, y_true, [5])
# y_true = pd.Series(list(y_true))

y_val = y_true['Label']

feature_metrix, y_val = adjust_xy(feature_metrix, y_val, [0,1,9,12])
y_val = pd.Series(list(y_val))

y_val = np.array(y_val)

rs_score_matrix = rs_score(feature_metrix,y_val)
print('rs_score_matrix:',rs_score_matrix)
print(rs_score_matrix.shape)

# rs_score_matrix_PL = rs_score(feature_matrix_PL,y_val)
# rs_score_matrix_BT = rs_score(feature_matrix_BT,y_val)
# print('rs_score_matrix_PL:',rs_score_matrix_PL)
# print('rs_score_matrix_BT:',rs_score_matrix_BT)
# np.save('hDAT_uptake_cla_rs_PL.npy',rs_score_matrix_PL)
# np.save('hDAT_uptake_cla_rs_BT.npy',rs_score_matrix_BT)
# np.save('all_DAT_uptake_cla_rs_PL.npy',rs_score_matrix_PL)
# np.save('all_DAT_uptake_cla_rs_BT.npy',rs_score_matrix_BT)
np.save(r'D:\python\result\RS及UMAP聚类\rs_npy/rs_%s_feature_combined_couple_%.3f.npy'%(dataset,couple),rs_score_matrix)
