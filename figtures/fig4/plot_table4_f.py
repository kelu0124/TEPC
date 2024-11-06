# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams["font.family"] = "Times New Roman"  # 字体
#
# # 数据
# accession_ids = ['RF', 'GBDT','KNN'] #横坐标的名称
# results2 = [0.947,0.944,0.719]
# results1 = [0.992,0.982,0.978] #柱状图的纵坐标
# results3 = [0.913,0.903,0.880]
# results4 = [0.890,0.865,0.876]
# results5 = [0.889,0.890,0.876]
#
# x = range(len(accession_ids))  # 横坐标的位置
# width = 0.17  # 每个柱的宽度
#
# fig, ax = plt.subplots(figsize=(8, 5))  # 宽10 高6的图形fig，轴ax
#
# # colors_datasets = {
# #     '33 (small) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
# #     '36 (medium) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
# #     '35 (large) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9']
# # }
#
# # 绘制柱状图
# # bars1 = ax.barh(np.array(x) - width * 2, results1, width, label='TEMLPCN (Lorenz)', color=['#882a11'])
# # bars2 = ax.barh(np.array(x) - width, results2, width, label='TEMLPCN (Chen)', color=['#a1472b'])
# # bars3 = ax.barh(x, results3, width, label='Umap', color=['#b96346'])
# # bars4 = ax.barh(np.array(x) + width, results4, width, label='PCA', color=['#d08062'])
# # bars5 = ax.barh(np.array(x) + width * 2, results5, width, label='t-SNE', color=['#ffad8c'])
# bars1 = ax.barh(np.array(x) - width * 2, results1, width, label='TEMLPCN (Lorenz)', color=['#a1472b'])
# bars2 = ax.barh(np.array(x) - width, results2, width, label='TEMLPCN (Chen)', color=['#c56053'])
# bars3 = ax.barh(x, results3, width, label='Umap', color=['#c57d53'])
# bars4 = ax.barh(np.array(x) + width, results4, width, label='PCA', color=['#c59953'])
# bars5 = ax.barh(np.array(x) + width * 2, results5, width, label='t-SNE', color=['#e2daa9'])
#
# # 添加数值标签
# def add_value_labels(ax, bars):
#     for bar in bars:
#         width = bar.get_width()
#         ax.annotate('{}'.format(width),
#                     xy=(width, bar.get_y() + bar.get_height() / 2),
#                     xytext=(3, 0),
#                     textcoords="offset points",
#                     ha='left', va='center',
#                     fontsize=12)  # 设置字体大小为8
#
# add_value_labels(ax, bars1)  # 在每个柱子上添加数值标签
# add_value_labels(ax, bars2)
# add_value_labels(ax, bars3)
# add_value_labels(ax, bars4)
# add_value_labels(ax, bars5)
#
# # 设置图例的字体大小和位置
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), shadow=False, ncol=5, fontsize=10, frameon=False)
#
# # 设置 y 轴范围从0.6开始
# ax.set_xlim(0.7, 1.0)
# # 设置 x 轴范围
# plt.ylim(-0.5, len(accession_ids) - 0.5)  # 线的长度
# # 设置刻度字号
# plt.yticks(fontsize=15)
# plt.xticks(fontsize=18)
#
# # 添加标签、标题和图例
# ax.set_xlabel('Balanced Accuracy', fontsize=20)  # Y轴的标签和字号
# ax.set_yticks(x)  # 设置x轴的刻度位置
# ax.set_yticklabels(accession_ids, rotation=45, ha='right')
# ax.tick_params(axis='y', which='major', pad=5, length=5)
#
# plt.gca().spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的线
# plt.gca().spines['right'].set_visible(False)
# # 设置y轴刻度在内部显示，并增加粗细
# plt.gca().tick_params(axis='x',direction='in', width=1, length=5)
# plt.gca().tick_params(axis='y',  width=1, length=5)
#
# plt.tight_layout()
# # plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig3_2.png', dpi=800, bbox_inches='tight')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams["font.family"] = "Times New Roman"  # 字体
#
#
# # 数据
# accession_ids = ['TEMLPCN (Lorenz)', 'TEMLPCN (Chen)','Umap','PCA','t-SNE'] #横坐标的名称
#
# '''results2 = [0.947,0.944,0.719]
# results1 = [0.992,0.982,0.978] #柱状图的纵坐标
# results3 = [0.913,0.903,0.880]
# results4 = [0.890,0.865,0.876]
# results5 = [0.889,0.890,0.876]'''
# results1 = [0.992,0.947,0.913,0.890,0.889]
# results2 = [0.982,0.944,0.903,0.865,0.890]
# results3 = [0.978,0.719,0.880,0.876,0.876]
# #results4 = [0.890,0.865,0.876]
# #results5 = [0.889,0.890,0.876]
#
# x = range(len(accession_ids))  # 横坐标的位置
#
# width = 0.25  # 每个柱的宽度 0.2
#
# fig, ax = plt.subplots(figsize=(8, 5))  # 宽10 高6的图形fig，轴ax
#
# # colors_datasets = {
# #     '33 (small) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
# #     '36 (medium) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
# #     '35 (large) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9']
# # }
#
# # 绘制柱状图
# # bars1 = ax.barh(np.array(x) - width * 2, results1, width, label='TEMLPCN (Lorenz)', color=['#882a11'])
# # bars2 = ax.barh(np.array(x) - width, results2, width, label='TEMLPCN (Chen)', color=['#a1472b'])
# # bars3 = ax.barh(x, results3, width, label='Umap', color=['#b96346'])
# # bars4 = ax.barh(np.array(x) + width, results4, width, label='PCA', color=['#d08062'])
# # bars5 = ax.barh(np.array(x) + width * 2, results5, width, label='t-SNE', color=['#ffad8c'])
# bars1 = ax.barh(np.array(x) - width , results1, width, label='RF', color=['#c2e1c0'])
# bars2 = ax.barh(x, results2, width, label='GBDT', color=['#96d1c6'])
# bars3 = ax.barh(np.array(x) + width, results3, width, label='KNN', color=['#9cc399'])
# #bars4 = ax.barh(np.array(x) + width, results4, width, label='PCA', color=['#c59953'])
# #bars5 = ax.barh(np.array(x) + width * 2, results5, width, label='t-SNE', color=['#e2daa9'])
#
# # 添加数值标签
# def add_value_labels(ax, bars):
#     for bar in bars:
#         width = bar.get_width()
#         ax.annotate('{}'.format(width),
#                     xy=(width, bar.get_y() + bar.get_height() / 2),
#                     xytext=(3, 0),
#                     textcoords="offset points",
#                     ha='left', va='center',
#                     fontsize=12)  # 设置字体大小为8
#
# add_value_labels(ax, bars1)  # 在每个柱子上添加数值标签
# add_value_labels(ax, bars2)
# add_value_labels(ax, bars3)
# #add_value_labels(ax, bars4)
# #add_value_labels(ax, bars5)
#
# # 设置图例的字体大小和位置
# ax.legend(loc='center', bbox_to_anchor=(0.3, 1), shadow=False, ncol=5, fontsize=12, frameon=False)
# '''ncol=5: 指定图例列数'''
#
# # 设置 y 轴范围从0.6开始
# ax.set_xlim(0.7, 1.0)
# # 设置 x 轴范围
# plt.ylim(-0.5, len(accession_ids) - 0.5)  # 线的长度
# # 设置刻度字号
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=18)
#
# # 添加标签、标题和图例
# ax.set_xlabel('Balanced Accuracy', fontsize=20)  # Y轴的标签和字号
# ax.set_yticks(x)  # 设置x轴的刻度位置
# ax.set_yticklabels(accession_ids, rotation=45, ha='right')
# ax.tick_params(axis='y', which='major', pad=5, length=5)
#
# plt.gca().spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的线
# plt.gca().spines['right'].set_visible(False)
# # 设置y轴刻度在内部显示，并增加粗细
# plt.gca().tick_params(axis='x',direction='in', width=1, length=5)
# plt.gca().tick_params(axis='y',  width=1, length=5)
#
# plt.tight_layout()
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig3_1.png', dpi=800, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"  # 字体


# 数据
accession_ids = ['TEPC \n(Lorenz)', 'TEPC \n(Chen)','Umap','PCA','t-SNE'] #横坐标的名称

'''results2 = [0.947,0.944,0.719]
results1 = [0.992,0.982,0.978] #柱状图的纵坐标
results3 = [0.913,0.903,0.880]
results4 = [0.890,0.865,0.876]
results5 = [0.889,0.890,0.876]'''
results1 = [0.992,0.947,0.913,0.890,0.889]
results2 = [0.982,0.944,0.903,0.865,0.890]
results3 = [0.978,0.719,0.880,0.876,0.876]
#results4 = [0.890,0.865,0.876]
#results5 = [0.889,0.890,0.876]

x = range(len(accession_ids))  # 横坐标的位置

width = 0.27  # 每个柱的宽度 0.2

fig, ax = plt.subplots(figsize=(7.2, 7))  # 宽10 高6的图形fig，轴ax

# colors_datasets = {
#     '33 (small) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
#     '36 (medium) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
#     '35 (large) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9']
# }

bars1 = ax.barh(np.array(x) - width , results1, width, label='RF', color=['#68789e'])#4f8950  #237588  #a06f3f  #DFE1E2
bars2 = ax.barh(x, results2, width, label='GBDT', color=['#E1B6B5'])#75a674  #60a3b5  #b8895d  #B7DBE3
bars3 = ax.barh(np.array(x) + width, results3, width, label='KNN', color=['#F9F2C1'])#c2e1c0绿色  #97d3e5  #d0a47d  #F5E09B三色

# 添加数值标签
def add_value_labels(ax, bars):
    for bar in bars:
        width = bar.get_width()
        ax.annotate('{}'.format(width),
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=14)  # 设置字体大小为8

add_value_labels(ax, bars1)  # 在每个柱子上添加数值标签
add_value_labels(ax, bars2)
add_value_labels(ax, bars3)


# 设置图例的字体大小和位置
ax.legend(loc='center', bbox_to_anchor=(0.4, 1.02), shadow=False, ncol=5, fontsize=14, frameon=False)

# 设置 y 轴范围从0.6开始
ax.set_xlim(0.7, 1.0)
# 设置 x 轴范围
plt.ylim(-0.5, len(accession_ids) - 0.5)  # 线的长度
# 设置刻度字号

plt.xticks(fontsize=18)

# 添加标签、标题和图例
ax.set_xlabel('Balanced Accuracy', fontsize=24)  # Y轴的标签和字号
ax.set_yticks(x)  # 设置x轴的刻度位置
ax.set_yticklabels(accession_ids, rotation=20,ha='right',fontsize=16)
ax.tick_params(axis='y', which='major', width=1, pad=0, length=5)

plt.gca().spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的线
plt.gca().spines['right'].set_visible(False)
# 设置y轴刻度在内部显示，并增加粗细
plt.gca().tick_params(axis='x',direction='in', width=1, length=5)


plt.tight_layout()
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig3_1.png' , dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig3_1.svg' , dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig3_1.pdf' , dpi=800, bbox_inches='tight')
plt.show()

