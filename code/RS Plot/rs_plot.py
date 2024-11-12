import getpass
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.io as pio
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
import copy
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import pearsonr

color = px.colors.qualitative.Light24   #color map
color_discrete_map= { str(idx): color[idx-1] for idx in range(1, 25)}   #map label to new color
symbol_discrete_map = {'1': 'circle-open', '2': 'diamond-open', '3': 'square-open', '4': 'cross-open',
                       '5': 'x-open', '6': 'triangle-up-open', '7': 'triangle-down-open', '8': 'triangle-left-open',
                       '9': 'triangle-right-open', '10': 'triangle-ne-open', '11': 'triangle-se-open',
                       '12': 'triangle-sw-open', '13': 'triangle-nw-open', '14': 'pentagon-open',
                       '15': 'star-open', '16':'hexagram-open', '17': 'star-square-open',
                       '18': 'star-diamond-open', '19': 'diamond-tall-open', '20': 'diamond-wide-open'}    #map label to symbols

def adjustCoordinate(rs_score, y, max_col = None):
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
    # print(rs_score_new)
    return rs_score_new

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

    max_x = int(np.ceil( np.max(rs_score[:, 0]))); max_y = int(np.ceil( np.max(rs_score[:, 1])) )  #get the boundaries
    fig.add_vline(x=0-0.01, line_width=3, line_color = 'black')  #add the borders
    fig.add_vline(x=max_x + 0.01, line_width=3, line_color = 'black')   #add the borders
    fig.add_hline(y=0-0.01, line_width=3, line_color = 'black')  #add the borders
    fig.add_hline(y=max_y + 0.01, line_width=3, line_color = 'black')#add the borders

    for idx in range(1,max_x):
        fig.add_vline(x = idx+0.01, line_width=2, line_color = 'black')

    for idx in range(1,max_y):
        fig.add_hline(y = idx + 0.01, line_width=2, line_color = 'black')

    fig.update_xaxes(range=[0 - 0.01, max_x + 0.01])  # fix the range of x
    fig.update_yaxes(range=[0 - 0.01, max_y + 0.01])  # fix the range of y
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')  #make background white
    fig.update_layout(legend={'title_text':''})  #remove title
    fig.update_layout({ax: {"visible": False, "matches": None}
                       for ax in fig.to_dict()["layout"] if "axis" in ax})  # remove ticks
    fig.update_layout(height=600, width=1000)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))   #margin

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

def cal_confusion_mat(y_true,y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    # 混淆矩阵热力图
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(confusion_mat, cmap='Reds', vmin=0, vmax=confusion_mat.max())
    # 添加坐标轴标签
    ax.xaxis.set_major_locator(MaxNLocator(nbins=confusion_mat.shape[1] + 1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=confusion_mat.shape[0] + 1))
    ax.set_xlabel('Predicted labels', size=18)
    ax.set_ylabel('True labels', size=18)
    # 添加类别标签
    class_labels = [0] + list(set(y_true))
    ax.xaxis.set_ticklabels(class_labels, size=12)
    ax.yaxis.set_ticklabels(class_labels, size=12)
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(j, i, format(confusion_mat[i, j], 'd'),
                    ha='center', va='center', color='black', size=12)
    # 显示热力图
    plt.savefig(r'D:\python\result\RS及UMAP聚类\clusting_result\heatmap/%s_heatmap.pdf' % dataset)
    # plt.show()
    return confusion_mat

path = r'D:\python\result\b因子系列\clusting'
dataset = 'GSE45719'
couple = 6

rs_score = np.load(r'D:\python\result\RS及UMAP聚类\rs_npy/rs_%s_feature_combined_couple_%.3f.npy'%(dataset,couple))
print('rs矩阵：\n',rs_score, np.shape(rs_score))

y_pred = np.load(r'D:\python\result\RS及UMAP聚类\pred_label/%s_RF_pred_label_results_%.3f.npy'%(dataset,couple))
print('预测值：\n',y_pred, np.shape(y_pred))

y_true = pd.read_csv(r'D:\python\result\RS及UMAP聚类\label\%s_full_labels.csv'%dataset)
y_true = y_true['Label']
_ = np.load(r'D:\python\result\RS及UMAP聚类\npy/%s_feature_combined_couple_%.3f.npy'%(dataset,couple))
_, y_true = adjust_xy(_, y_true, [])
y_true = pd.Series(list(y_true))
y_true = np.array(y_true)
print('真实值：\n',y_true,y_true.shape)

rs_new = adjustCoordinate(rs_score, y_true)
fig = constructFigure(rs_new, y_pred)

pio.write_image(fig, r'D:\python\result\RS及UMAP聚类\clusting_result\rs_plot/%s_rsplot_output.pdf'%dataset)
# fig.show()

# 计算混淆矩阵并生成热力图
print(f'混淆矩阵:\n{cal_confusion_mat(y_true,y_pred)}')

# 计算调整后的兰德指数（Adjusted Rand Score）
ars = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Score（ARI）: {ars}")

# 这里我们使用Adjusted Rand Score作为残基相似性指数的示例
residue_similarity_index = ars
print(f"Residue-Similarity Index（RSI）: {residue_similarity_index}")

# 计算残差索引（RI）
residual = y_true - y_pred
RI = np.sum(np.abs(residual)) / np.sum(np.abs(y_pred - np.mean(y_pred)))

# 计算相似性索引（SI）
SI = pearsonr(y_true,y_pred)[0]

print(f"residue index（RI）: {RI}")
print(f"similarity index（SI）: {SI}")

residue_similarity_index = 1 - abs(RI - SI)
print(f"Residue-Similarity Index（RSI）: {residue_similarity_index}")
