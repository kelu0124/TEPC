import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from umap import UMAP
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

color = px.colors.qualitative.Light24   #color map
color_discrete_map= { str(idx): color[idx-1] for idx in range(1, 25)}   #map label to new color
symbol_discrete_map = {'1': 'circle-open', '2': 'diamond-open', '3': 'square-open', '4': 'cross-open',
                       '5': 'x-open', '6': 'triangle-up-open', '7': 'triangle-down-open', '8': 'triangle-left-open',
                       '9': 'triangle-right-open', '10': 'triangle-ne-open', '11': 'triangle-se-open',
                       '12': 'triangle-sw-open', '13': 'triangle-nw-open', '14': 'pentagon-open',
                       '15': 'star-open', '16':'hexagram-open', '17': 'star-square-open',
                       '18': 'star-diamond-open', '19': 'diamond-tall-open', '20': 'diamond-wide-open'}    #map label to symbols

def constructFigure(rs_score, y, color_discrete_map = color_discrete_map,
                    symbol_discrete_map = symbol_discrete_map):
    '''
        Input:
            rs_score: residue-similarity score. If you are plotting all the classes, make sure you run adjustCoordinate first
            y: the color you want to use to color the rs-plot. (I used y_pred to color the points)
    '''
    df = {'x': rs_score[:, 0], 'y': rs_score[:, 1], 'label':y.astype(int)}   #creat dictionary of points
    df['label'] = df['label'].astype('str')   #convert preicted labels to stirng -> allows discrete colors

    fig = px.scatter(df, x = 'x', y = 'y', color = 'label', symbol = 'label',
                    color_discrete_map=color_discrete_map, symbol_map=symbol_discrete_map,
                    )  #Construct figure with discrete color and custom symbols
    fig.update_traces(marker=dict(size=13,
                              line=dict(width=3,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))   #size and line width of markers
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')  # make background white
    max_x = int(np.ceil( np.max(rs_score[:, 0]))); max_y = int(np.ceil( np.max(rs_score[:, 1])) )  #get the boundaries
    min_x = int(np.ceil( np.min(rs_score[:, 0]))); min_y = int(np.ceil( np.min(rs_score[:, 1])) )  #get the boundaries

    fig.add_vline(x=min_x-2, line_width=3, line_color = 'black')  #add the borders
    fig.add_vline(x=max_x + 1, line_width=3, line_color = 'black')   #add the borders
    fig.add_hline(y=min_y-2, line_width=3, line_color = 'black')  #add the borders
    fig.add_hline(y=max_y + 1, line_width=3, line_color = 'black')#add the borders

    # for idx in range(1,max_x):
    #     fig.add_vline(x = idx+0.01, line_width=2, line_color = 'black')
    #
    # for idx in range(1,max_y):
    #     fig.add_hline(y = idx + 0.01, line_width=2, line_color = 'black')

    fig.update_xaxes(range=[min_x - 2, max_x + 1])  #fix the range of x
    fig.update_yaxes(range=[min_y - 2, max_y + 1])  #fix the range of y
    fig.update_layout({ax: {"showticklabels": False} for ax in ['xaxis', 'yaxis']}) #remove ticks
    # fig.update_layout(xaxis_title='UMAP 1', yaxis_title='UMAP 2',
    #                   xaxis_title_font = dict(color='black'),yaxis_title_font = dict(color='black'))
    fig.update_layout(height=600, width=1000)  #size of graph   You should change this, depending on your plot
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))  # margin

    return fig

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

def calculate_residue_similarity_index(original_data, umap_data):
    # 计算原始数据点对之间的距离
    original_distances = pairwise_distances(original_data)

    # 计算 UMAP 降维后的数据点对之间的距离
    umap_distances = pairwise_distances(umap_data)

    # 对两个距离矩阵进行排序
    original_sorted_indices = np.argsort(original_distances.flatten())
    umap_sorted_indices = np.argsort(umap_distances.flatten())

    # 计算排序的残差
    residue = np.sum(np.abs(original_sorted_indices - umap_sorted_indices))

    # 计算 RSI
    n = len(original_sorted_indices)
    rsi = 1 - (2 * residue) / (n * (n - 1))

    return rsi

#######控制台#########
number = 0 # 数据集编号
######################

path = r'D:\python\result\b因子系列\clusting'

datasets = ['GSE45719','GSE59114','GSE67835','GSE75748cell','GSE75748time','GSE82187','GSE84133human1'
            ,'GSE84133human2','GSE84133human4','GSE84133mouse1','GSE84133mouse2','GSE89232','GSE94820']
couples = [6,1,4,1,13,6,7,4.6,8,4,10,7,0.5]
labels = [[],[],[],[],[],[5],[10,12,13],[0,7,13],[0,7,10,11,12,13],[0,1,2,9,12],[0,1,9,12],[],[]]
dataset,couple,label = datasets[number],couples[number],labels[number]

data_feature = np.load(r'D:\python\result\RS及UMAP聚类\npy/%s_feature_combined_couple_%.3f.npy'%(dataset,couple))
y_true = pd.read_csv(r'D:\python\result\RS及UMAP聚类\label\%s_full_labels.csv'%dataset)
y_true = y_true['Label']
# 调整数据和标签以排除特定类别的点
data_feature, y_true = adjust_xy(data_feature, y_true, label) # label:要被删除的标签名
y_true = pd.Series(list(y_true))
y_true = np.array(y_true)
print(data_feature.shape,y_true.shape)

# 初始化UMAP实例
umap_reducer = UMAP(n_neighbors=15, min_dist=0.5, random_state=42)

# 拟合数据并转换
umap_data = umap_reducer.fit_transform(data_feature)

kmeans = KMeans(n_clusters=np.unique(y_true).size, random_state=42)
# kmeans = KMeans(n_clusters=16, random_state=42)
kmeans.fit(data_feature)
cluster_labels = kmeans.labels_
# rs_score = np.load(path + '/rs_feature_combined_couple_23.000.npy')
# print(rs_score, np.shape(rs_score))
# # y_pred = np.load('all_DAT_uptake_cla_y_pred_PL.npy')
# y_pred = np.load(path + '/RF_pred_label_results_23.000.npy')
# print(y_pred, np.shape(y_pred))
# y_true = pd.read_csv(path + '\GSE45719_full_labels.csv')
# y_true = np.array(y_true['Label'])
# print(y_true, np.shape(y_true))
# exit()
fig = constructFigure(umap_data, cluster_labels)

fig.show()
pio.write_image(fig, r'D:\python\result\RS及UMAP聚类\clusting_result\U-map/%s_umap_output.eps'%dataset)

rsi_score = calculate_residue_similarity_index(data_feature, umap_data)
print("Residue Similarity Index (RSI):", rsi_score)