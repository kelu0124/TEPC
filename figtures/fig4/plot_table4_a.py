# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams["font.family"] = "Times New Roman"
#
# # 数据
# datasets = ['33 (small) datasets', '36 (medium) datasets', '35 (large) datasets']
#
# # 方法和对应的值的字典
# # data_dict = {
# #     '300': {'TEMLPCN (Lorenz)': 0.817, 'TEMLPCN (Rossler)': 0.6, 'mANM': 0.546},
# #     '362': {'TEMLPCN (Lorenz)': 0.796, 'TEMLPCN (Rossler)': 0.721, 'mGNM': 0.642},
# #     '364': {'TEMLPCN (Lorenz)': 0.794, 'TEMLPCN (Rossler)': 0.712, 'EH (Rossler)': 0.698, 'EH (Lorenz)': 0.691,
# #             'mANM': 0.670, 'mGNM': 0.626, 'mFRI': 0.565}
# # }
# accession_ids = ['TEMLPCN(Lorenz)', 'TEMLPCN(Rossler)', 'EH(Lorenz)', 'EH(Rossler)', 'opFRI','pfFRI', 'MND', 'GNM', 'GNM','NMA']
# PCC_results_33_small = [0.940, 0.916, 0.746, 0.773, 0.667, 0.594, 0.580, 0.541, 0.541, 0.480]
# PCC_results_36_medium = [0.826, 0.728, 0.701, 0.729, 0.664, 0.605, 0.603, 0.555, 0.550, 0.482]
# PCC_results_35_large = [0.789, 0.664, 0.663, 0.665, 0.636, 0.591, 0.584, 0.530, 0.529, 0.494]
#
# data_dict = {}
#
# # Populate data_dict with values from PCC_results_33_small
# data_dict['33 (small) datasets'] = {
#     'TEMLPCN (Lorenz)': PCC_results_33_small[0],
#     'TEMLPCN (Rossler)': PCC_results_33_small[1],
#     'EH (Lorenz)': PCC_results_33_small[2],
#     'EH (Rossler)': PCC_results_33_small[3],
#     'opFRI': PCC_results_33_small[4],
#     'pfFRI': PCC_results_33_small[5],
#     'MND': PCC_results_33_small[6],
#     'GNM': PCC_results_33_small[7],
#     ' GNM': PCC_results_33_small[8],
#     'NMA': PCC_results_33_small[9]
# }
#
# # Populate data_dict with values from PCC_results_36_medium
# data_dict['36 (medium) datasets'] = {
#     'TEMLPCN (Lorenz)': PCC_results_36_medium[0],
#     'TEMLPCN (Rossler)': PCC_results_36_medium[1],
#     'EH (Lorenz)': PCC_results_36_medium[2],
#     'EH (Rossler)': PCC_results_36_medium[3],
#     'opFRI': PCC_results_36_medium[4],
#     'pfFRI': PCC_results_36_medium[5],
#     'MND': PCC_results_36_medium[6],
#     'GNM': PCC_results_36_medium[7],
#     ' GNM': PCC_results_36_medium[8],
#     'NMA': PCC_results_36_medium[9]
# }
#
# # Populate data_dict with values from PCC_results_35_large
# data_dict['35 (large) datasets'] = {
#     'TEMLPCN (Lorenz)': PCC_results_35_large[0],
#     'TEMLPCN (Rossler)': PCC_results_35_large[1],
#     'EH (Lorenz)': PCC_results_35_large[2],
#     'EH (Rossler)': PCC_results_35_large[3],
#     'opFRI': PCC_results_35_large[4],
#     'pfFRI': PCC_results_35_large[5],
#     'MND': PCC_results_35_large[6],
#     'GNM': PCC_results_35_large[7],
#     ' GNM': PCC_results_35_large[8],
#     'NMA': PCC_results_35_large[9]
# }
#
# print(data_dict)
# # 创建画布
# fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [0.33, 0.33, 0.33]})
#
# # 不显示右边和上边的线条
# for ax in axs:
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
# colors_datasets = {
#     '33 (small) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
#     '36 (medium) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9'],
#     '35 (large) datasets': ['#8a3a3a', '#c55353', '#c56053','#c56f53','#c57d53','#c58b53','#c59953','#c5a653','#c5b553','#e2daa9']
# }
# for i, dataset in enumerate(datasets):
#     ax = axs[i]
#     values_dict = data_dict[dataset]
#     methods = list(values_dict.keys())
#     values = list(values_dict.values())
#     num_methods = len(methods)
#
#     # 选择当前数据集的颜色
#     colors = colors_datasets[dataset]
#
#     bars = ax.bar(np.arange(num_methods), values, width=0.8, color=colors)  # 使用自定义颜色
#     ax.set_title(f'{dataset}',fontsize=18)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)  # 默认标签右对齐
#     ax.set_ylim(0.4, 0.9)  # 设置纵坐标范围
#     ax.set_yticks(np.arange(0.4, 0.91, 0.1))  # 设置纵坐标刻度
#     ax.yaxis.set_tick_params(width=1, length=5, pad=10, direction='in', labelright=False)  # 调整刻度参数位置
#     if i == 0:
#         ax.set_ylabel('Average PCCs', fontsize=19)  # 在第一个子图上标注坐标轴标签
#         ax.set_ylim(0.45, 0.95)  # 设置纵坐标范围
#         ax.set_yticks(np.arange(0.45, 0.96, 0.1))  # 设置纵坐标刻度、
#         ax.set_yticks([0.45,0.55,0.65,0.75,0.85,0.95])
#     if i == 1:
#         # ax.set_ylabel('Average PCCs', fontsize=18)  # 在第一个子图上标注坐标轴标签
#         ax.set_ylim(0.45, 0.85)  # 设置纵坐标范围
#         ax.set_yticks(np.arange(0.45, 0.86, 0.1))  # 设置纵坐标刻度
#     if i == 2:
#         # ax.set_ylabel('Average PCCs', fontsize=18)  # 在第一个子图上标注坐标轴标签
#         ax.set_ylim(0.45, 0.80)  # 设置纵坐标范围
#         ax.set_yticks([0.45,0.55,0.65,0.75,0.80])  # 设置纵坐标刻度
#
#     # 在每个柱子的顶端显示数值
#     for j, value in enumerate(values):
#         ax.text(j, value, f'{value:.3f}', ha='center', va='bottom' if value > 0.4 else 'top', fontsize=10)
#
#     # 设置x轴刻度位置和标签
#     ax.set_xticks(np.arange(num_methods))
#     ax.set_xticklabels(methods, ha='right', fontsize=12)  # "TEMLPCN (Lorenz)" 右对齐
#
# # 调整布局
# plt.subplots_adjust(wspace=0.15)
#
# # 保存和显示图形
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_1_1.png', dpi=800, bbox_inches='tight')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# 数据
datasets = ['33 (small) datasets', '36 (medium) datasets', '35 (large) datasets']

# 方法和对应的值的字典
accession_ids = ['TEPC (Lorenz)', 'TEPC (Rossler)', 'EH(Lorenz)', 'EH(Rossler)', 'opFRI', 'pfFRI', 'MND', 'GNM1',
                 'GNM2', 'NMA']
PCC_results_33_small = [0.940, 0.934, 0.746, 0.773, 0.667, 0.594, 0.580, 0.541, 0.541, 0.480]
PCC_results_36_medium = [0.826, 0.774, 0.701, 0.729, 0.664, 0.605, 0.603, 0.555, 0.550, 0.482]
PCC_results_35_large = [0.789, 0.701, 0.663, 0.665, 0.636, 0.591, 0.584, 0.530, 0.529, 0.494]

data_dict = {}

# Populate data_dict with values from PCC_results_33_small
data_dict['33 (small) datasets'] = {
    'TEPC (Lorenz)': PCC_results_33_small[0],
    'TEPC (Rossler)': PCC_results_33_small[1],
    'EH (Lorenz)': PCC_results_33_small[2],
    'EH (Rossler)': PCC_results_33_small[3],
    'opFRI': PCC_results_33_small[4],
    'pfFRI': PCC_results_33_small[5],
    'MND': PCC_results_33_small[6],
    'GNM1': PCC_results_33_small[7],
    'GNM2': PCC_results_33_small[8],
    'NMA': PCC_results_33_small[9]
}

# Populate data_dict with values from PCC_results_36_medium
data_dict['36 (medium) datasets'] = {
    'TEPC (Lorenz)': PCC_results_36_medium[0],
    'TEPC (Rossler)': PCC_results_36_medium[1],
    'EH (Lorenz)': PCC_results_36_medium[2],
    'EH (Rossler)': PCC_results_36_medium[3],
    'opFRI': PCC_results_36_medium[4],
    'pfFRI': PCC_results_36_medium[5],
    'MND': PCC_results_36_medium[6],
    'GNM1': PCC_results_36_medium[7],
    'GNM2': PCC_results_36_medium[8],
    'NMA': PCC_results_36_medium[9]
}

# Populate data_dict with values from PCC_results_35_large
data_dict['35 (large) datasets'] = {
    'TEPC (Lorenz)': PCC_results_35_large[0],
    'TEPC (Rossler)': PCC_results_35_large[1],
    'EH (Lorenz)': PCC_results_35_large[2],
    'EH (Rossler)': PCC_results_35_large[3],
    'opFRI': PCC_results_35_large[4],
    'pfFRI': PCC_results_35_large[5],
    'MND': PCC_results_35_large[6],
    'GNM1': PCC_results_35_large[7],
    'GNM2': PCC_results_35_large[8],
    'NMA': PCC_results_35_large[9]
}

# 创建画布
fig, axs = plt.subplots(1, 3, figsize=(8.5, 6))

# 不显示右边和上边的线条
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# 为每个数据集选择颜色
colors_datasets = {
    '33 (small) datasets': ['#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2',
                            '#74c476', '#74c476'],
    '36 (medium) datasets': ['#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2',
                            '#74c476', '#74c476'],
    '35 (large) datasets': ['#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2',
                            '#74c476', '#74c476'],
}
# colors_datasets = {
#     '33 (small) datasets': ['#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476',
#                             '#4f8950', '#4f8950'],
#     '36 (medium) datasets': ['#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476',
#                             '#75a674', '#75a674'],
#     '35 (large) datasets': ['#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476', '#74c476',
#                             '#75a674', '#4f8950'],
# }
for i, dataset in enumerate(datasets):
    ax = axs[i]
    values_dict = data_dict[dataset]
    methods = list(values_dict.keys())
    values = list(values_dict.values())
    num_methods = len(methods)

    # 选择当前数据集的颜色
    colors = colors_datasets[dataset]
    bars = ax.barh(np.arange(num_methods), values[::-1], height=0.65, color=colors)  # 使用自定义颜色，同时反转values列表以匹配y轴的顺序
    ax.set_title(f'{dataset} ', fontsize=18)
    ax.tick_params(axis='y', labelsize=10)  # 默认标签不旋转
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 不显示 x 轴刻度及其标签
    ax.set_xlim(0.4, 0.95)  # 设置横坐标范围
    ax.xaxis.set_tick_params(width=1, length=5, pad=0, direction='in', labeltop=False)  # 调整刻度参数位置

    ax.set_xlabel('Average PCCs', fontsize=20)  # 标注坐标轴标签
    ax.set_yticks(np.arange(num_methods))

    if i == 0:
        ax.set_yticklabels(accession_ids[::-1], rotation=45,fontsize=15)  #y轴
        ax.set_xlim(0.35, 0.940)
    if i == 1:
        ax.set_yticks([])  # 隐藏纵坐标标签
        ax.set_xlim(0.35, 0.826)
    if i == 2:
        ax.set_yticks([])  # 隐藏纵坐标标签
        ax.set_xlim(0.4, 0.789)

    for j, value in enumerate(values[::-1]):  # 反转values列表以匹配y轴的顺序
        ax.text(value, j, f'{value:.3f}', ha='right', va='center', fontsize=13, fontweight='bold')

# 调整布局
plt.subplots_adjust(wspace=0.1)

# 保存和显示图形
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_1_2.svg', dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_1_2.png', dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_1_2.pdf', dpi=800, bbox_inches='tight')
plt.show()