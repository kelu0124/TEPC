# -*- coding: utf-8 -*-
import getpass
import random
from matplotlib.transforms import Bbox
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
import copy
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import pearsonr
import plotly.express as px
from umap import UMAP
import umap
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

# 将十六进制颜色码转换为RGB元组（0-1范围）
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # 移除'#'前缀
    rgb_int = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    rgb_float = tuple(val / 255.0 for val in rgb_int)
    return rgb_float

def adjustCoordinate(rs_score, y,y_pred, max_col = None):
    '''将所有类别分别分到对应的框内（分之前所有点是聚在一起的）'''
    rs_score_new = rs_score.copy()
    label_ls = list(set(list(y)))
    total_labels = len(label_ls)

    if max_col == None:
        max_col = int(np.ceil(np.sqrt(total_labels)))
    max_row = int(np.ceil(total_labels / max_col))

    current_col = 0
    current_row = max_row - 1
    label_ls_new = [_ for _ in range(max(label_ls) + 1)]

    for label in label_ls_new:

        index = np.where(y == label)[0]
        # print(label,index)
        rs_score_new[index, 0] += current_col
        rs_score_new[index, 1] += current_row

        current_col += 1
        if current_col == max_col:
            current_col = 0
            current_row -= 1

    rs_score_right,y_right,y_pred_right,rs_score_false,y_false,y_pred_false = [],[],[],[],[],[]
    for i in range(len(rs_score)):
        if y[i] == y_pred[i]:
            rs_score_right.append(rs_score_new[i])
            y_right.append(y[i])
            y_pred_right.append(y_pred[i])
        else:
            rs_score_false.append(rs_score_new[i])
            y_false.append(y[i])
            y_pred_false.append(y_pred[i])
    rs_score_new = np.array(rs_score_right + rs_score_false)
    y = np.array(y_right + y_false)
    y_pred = np.array(y_pred_right + y_pred_false)
    return rs_score_new,y,y_pred

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

# def cal_confusion_mat(y_true,y_pred,xpos,ypos):

def draw_result(dataset,couple,adjust_label,xpos,ypos):
    # 获取并调整数据
    rs_score = np.load(r'C:\Users\HAHA\Desktop\rs_npy\rs_%s_feature_combined_couple_%.3f.npy'%(dataset,couple))
    y_pred = np.load(r'C:\Users\HAHA\Desktop\源文件\pred_label/%s_RF_pred_label_results_%.3f.npy'%(dataset,couple))
    y_true = pd.read_csv(r'C:\Users\HAHA\Desktop\源文件\label\%s_full_labels.csv'%dataset)
    y_true = y_true['Label']
    _ = np.load(r'C:\Users\HAHA\Desktop\源文件\npy\%s_feature_combined_couple_%.3f.npy'%(dataset,couple))
    _, y_true = adjust_xy(_, y_true, adjust_label)
    y_true = pd.Series(list(y_true))
    y_true = np.array(y_true)

    # rs_plot部分
    rs_new,y_true_new,y_pred_new = adjustCoordinate(rs_score, y_true,y_pred) # rs_plot中分错的点要处于上方，要转移位置
    color_discrete_map = {str(idx): color[idx-1] for idx in range(0, 25)}
    colors = [color_discrete_map[str(int(float(label)))] for label in map(str, y_pred_new)]
    symbols = [symbol_discrete_map[str(int(float(label)))] for label in map(str, y_pred_new)]
    handles = []
    for i, (x, y, symbol) in enumerate(zip(rs_new[:, 0], rs_new[:, 1], symbols)):
        label = str(y_true_new[i])
        scatter = axs[xpos, ypos].scatter(x, y, c=colors[i],marker=symbol, edgecolors='k', s=60, label=label)
        # 只添加第一个出现的标签到handles列表
        if label not in [h.get_label() for h in handles]:
            handles.append(scatter)
    if dataset == 'GSE75748time':
        dataset_ = 'GSE75748 time'
        axs[xpos, ypos].set_title(f'{dataset_}', size=16)
    elif dataset == 'GSE75748cell':
        dataset_ = 'GES75748 cell'
        axs[xpos, ypos].set_title(f'{dataset_}', size=16)
    else:
        dataset_ = dataset
        axs[xpos, ypos].set_title(f'{dataset}',size = 16)
    if xpos == 0 and ypos == 0:
        legend = axs[xpos, ypos].legend(handles=handles, bbox_to_anchor=(0.1, -4),loc='upper left',ncol=len(handles)) # 添加标签，bbox_to_anchor用于调整标签坐标
        legend.get_title().set_fontsize('18')  # 设置标题的字体大小
        for text in legend.get_texts():
            text.set_fontsize('18')  # 设置每个标签的字体大小

    # umap部分
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.5, random_state=42)
    umap_data = umap_reducer.fit_transform(rs_score)
    kmeans = KMeans(n_clusters=np.unique(y_true).size, random_state=42)
    # kmeans = KMeans(n_clusters=16, random_state=42)
    kmeans.fit(rs_score)
    cluster_labels = kmeans.labels_
    colors = [color_discrete_map[str(int(float(label)))] for label in map(str, cluster_labels)]
    symbols = [symbol_discrete_map[str(int(float(label)))] for label in map(str, cluster_labels)]
    handles = []
    for i, (x, y, symbol) in enumerate(zip(umap_data[:, 0], umap_data[:, 1], symbols)):
        label = str(cluster_labels[i])
        scatter = axs[xpos, ypos + 1].scatter(x, y, c=colors[i], marker=symbol, edgecolors='k', s=60, label=label)
        # 只添加第一个出现的标签到handles列表
        if label not in [h.get_label() for h in handles]:
            handles.append(scatter)
    axs[xpos, ypos ].set_title(f'{dataset_}', size=16)
    #axs[xpos, ypos + 1].legend(handles=handles, title='Labels', bbox_to_anchor=(1, 2))

    # 热力图部分
    conf_mat = confusion_matrix(y_true, y_pred)

    hex_left = '#ffff99'  # 黄色
    hex_right = '#CD4F39'  # 红色
    rgb_left = hex_to_rgb(hex_left)
    red_right = hex_to_rgb(hex_right)

    # 创建颜色映射
    colors = [rgb_left, red_right]
    cmap = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', colors, N=256)
    cax = axs[xpos, ypos + 2].imshow(conf_mat, cmap=cmap, interpolation='nearest')

    # Add colorbar
    fig.colorbar(cax, ax=axs[xpos, ypos+2], fraction=0.046, pad=0.04)
    # Set labels and title for confusion matrix subplot
    if xpos == 3:
        axs[xpos, ypos+2].set_xlabel('Predicted labels', size=16)
    axs[xpos, ypos+2].set_ylabel('The true labels', size=16)
    class_labels = list(set(y_true))
    axs[xpos, ypos+2].set_xticks(np.arange(len(class_labels)))
    axs[xpos, ypos+2].set_yticks(np.arange(len(class_labels)))
    axs[xpos, ypos+2].set_xticklabels(class_labels, size=14)
    axs[xpos, ypos+2].set_yticklabels(class_labels, size=14)
    axs[xpos, ypos+2].set_title(f'{dataset_}', size=16)
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            axs[xpos, ypos+2].text(j, i, str(conf_mat[i, j]), ha='center', va='center', color='black', size=12)

col,vol = 4,3
fig, axs = plt.subplots(col,vol, figsize=(12, 14))
symbol_discrete_map = {'1': 'o','2': 's','3': 'D','4': '+','5': 'x','6': '^','7': 'v','8': '<','9': '>',
                       '10': '>','11': 'v','12': '<','13': '^','14': 'p','15': '*','0': 'D'}
color = px.colors.qualitative.Light24 # color map
draw_result('GSE45719',6,[],0,0)
draw_result('GSE59114',1,[],1,0)
draw_result('GSE75748cell',1,[],2,0)
draw_result('GSE75748time',13,[],3,0)

# 去除所有刻度
for i in range(col):
    for j in [0,1]:
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        
#第一行第一列子图
axs[0, 0].axhline(y=1, color='grey', linestyle='-', linewidth=1)
axs[0, 0].axhline(y=2, color='grey', linestyle='-', linewidth=1)
axs[0, 0].axvline(x=1, color='grey', linestyle='-', linewidth=1)
axs[0, 0].axvline(x=2, color='grey', linestyle='-', linewidth=1)
axs[0, 0].set_xlim([0, 3])
axs[0, 0].set_ylim([0, 3])

#第二行第一列子图
axs[1, 0].axhline(y=1, color='grey', linestyle='-', linewidth=1)
axs[1, 0].axvline(x=1, color='grey', linestyle='-', linewidth=1)
axs[1, 0].axvline(x=2, color='grey', linestyle='-', linewidth=1)
axs[1, 0].set_xlim([0, 3])
axs[1, 0].set_ylim([0, 2])

#第三行第一列子图
axs[2, 0].axhline(y=1, color='grey', linestyle='-', linewidth=1)
axs[2, 0].axhline(y=2, color='grey', linestyle='-', linewidth=1)
axs[2, 0].axvline(x=1, color='grey', linestyle='-', linewidth=1)
axs[2, 0].axvline(x=2, color='grey', linestyle='-', linewidth=1)
axs[2, 0].set_xlim([0, 3])
axs[2, 0].set_ylim([0, 3])

#第四行第一列子图
axs[3, 0].axhline(y=1, color='grey', linestyle='-', linewidth=1)
axs[3, 0].axvline(x=1, color='grey', linestyle='-', linewidth=1)
axs[3, 0].axvline(x=2, color='grey', linestyle='-', linewidth=1)
axs[3, 0].set_xlim([0, 3])
axs[3, 0].set_ylim([0, 2])


axs[0 ,0].text(-0.3, 3.2, 'a',size=26,weight='bold')
#第一行第一列子图添加文本，内容为a，位置在x坐标为-0.3，y坐标为3.2。大小设置为26磅字体加粗。
axs[0 ,1].text(-7, 24, 'b',size=26,weight='bold')
axs[0 ,2].text(-1.0, -1, 'c',size=26,weight='bold')

plt.subplots_adjust(wspace=0.15, hspace=0.3,left=0.05, right=0.92, top=0.95, bottom=0.07)
#调整matplotlib图形中子图之间的间距以及整个图形的边距

for ax in axs.flat:  #去掉边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
plt.savefig('C:\\Users\\HAHA\\Desktop\\finally  code\\fig5.svg', bbox_inches='tight')  # 保存子图为PNG文件
plt.savefig('C:\\Users\\HAHA\\Desktop\\finally  code\\fig5.png', bbox_inches='tight')  # 保存子图为PNG文件
plt.show()
#plt.savefig('fig5.png')
