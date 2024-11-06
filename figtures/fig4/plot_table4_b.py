import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# 数据
datasets = ['300', '362', '364']

# 方法和对应的值的字典
data_dict = {
    '300': {'TEPC (Lorenz)': 0.817, 'TEPC (Rossler)': 0.748, 'mANM': 0.546},
    '362': {'TEPC (Lorenz)': 0.796, 'TEPC (Rossler)': 0.722, 'mGNM': 0.642},
    '364': {'TEPC (Lorenz)': 0.794, 'TEPC (Rossler)': 0.720, 'EH (Rossler)': 0.698, 'EH (Lorenz)': 0.691,
            'mFRI': 0.670, 'pfFRI': 0.626, 'GNM': 0.565}
}

# 创建画布
fig, axs = plt.subplots(1, 3, figsize=(8.5, 8), gridspec_kw={'width_ratios': [0.25, 0.25, 0.5]})

# 不显示右边和上边的线条
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# 为每个数据集选择颜色
# colors_datasets = {
#     '300': ['#8a3a3a', '#c55353', '#bb4f4f'],
#     '362': ['#8a3a3a', '#c55353', '#b14b4b'],
#     '364': ['#8a3a3a', '#c55353', '#c56053','#c56f53', '#bb844f', '#c58b53', '#b17d4b']
# }
# colors_datasets = {
#     '300': ['#4f8950', '#75a674', '#c2e1c0'],
#     '362': ['#4f8950', '#75a674', '#c2e1c0'],
#     '364': ['#4f8950', '#75a674', '#c2e1c0','#c2e1c0', '#c2e1c0', '#c2e1c0', '#c2e1c0']
# }
colors_datasets = {
    '300': ['#74c476', '#74c476', '#b9e3b2'],
    '362': ['#74c476', '#74c476', '#b9e3b2'],
    '364': ['#74c476', '#74c476', '#b9e3b2','#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2']
}
for i, dataset in enumerate(datasets):
    ax = axs[i]
    values_dict = data_dict[dataset]
    methods = list(values_dict.keys())
    values = list(values_dict.values())
    num_methods = len(methods)

    # 选择当前数据集的颜色
    colors = colors_datasets[dataset]
    bars = ax.bar(np.arange(num_methods), values, width=0.8, color=colors)  # 使用自定义颜色
    ax.set_title(f'{dataset} datasets',fontsize=18, pad=15)  #标题大小及位置

    ax.tick_params(axis='x', rotation=50, labelsize=10)  # 默认标签右对齐
    # ax.set_ylim(0.4, 0.9)  # 设置纵坐标范围
    # ax.set_yticks(np.arange(0.4, 0.91, 0.1))  # 设置纵坐标刻度
    ax.yaxis.set_tick_params(width=1, length=5, pad=0, direction='in', labelsize=16,labelright=False, rotation=0)  # 调整刻度参数位置
    if i == 0:
        ax.set_ylabel('Average PCCs', fontsize=24)  # 在第一个子图上标注坐标轴标签
        ax.set_ylim(0.5, 0.85)  # 设置纵坐标范围
        ax.set_yticks([0.5,0.6,0.7,0.8,0.85])  # 设置纵坐标刻度、
    if i == 1:
        #ax.set_ylabel('Average PCCs', fontsize=18)  # 在第一个子图上标注坐标轴标签
        ax.set_ylim(0.62, 0.8)  # 设置纵坐标范围
        ax.set_yticks([0.62,0.7,0.8])  # 设置纵坐标刻度
    if i == 2:
        #ax.set_ylabel('Average PCCs', fontsize=18)  # 在第一个子图上标注坐标轴标签
        ax.set_ylim(0.52, 0.8)  # 设置纵坐标范围
        ax.set_yticks([0.52,0.65,0.75,0.80])  # 设置纵坐标刻度

    # 在每个柱子的顶端显示数值
    for j, value in enumerate(values):
        ax.text(j, value+0.00, f'{value:.3f}', ha='center', va='bottom' , fontsize=13  ,rotation=0)#, fontweight='bold'
    # for j, value in enumerate(values[::-1]):  # 反转values列表以匹配y轴的顺序
        # ax.text(value, j, f'{value:.3f}', ha='right', va='center', fontsize=11, fontweight='bold')

    # 设置x轴刻度位置和标签
    ax.set_xticks(np.arange(num_methods))
    ax.set_xticklabels(methods, ha='right', fontsize=16)  # "TEMLPCN (Lorenz)" 右对齐

# 调整布局
plt.subplots_adjust(wspace=0.22)

# 保存和显示图形
plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_2_3.png', dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_2_3.pdf', dpi=800, bbox_inches='tight')
plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_2_3.svg', dpi=800, bbox_inches='tight')

plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams["font.family"] = "Times New Roman"
#
# # 数据
# datasets = ['300', '362', '364']
#
# # 方法和对应的值的字典
# data_dict = {
#     '300': {'TEMLPCN (Lorenz)': 0.817, 'TEMLPCN (Rossler)': 0.6, 'mANM': 0.546},
#     '362': {'TEMLPCN (Lorenz)': 0.796, 'TEMLPCN (Rossler)': 0.721, 'mGNM': 0.642},
#     '364': {'TEMLPCN (Lorenz)': 0.794, 'TEMLPCN (Rossler)': 0.712, 'EH (Rossler)': 0.698, 'EH (Lorenz)': 0.691,
#             'mFRI': 0.670, 'pfFRI': 0.626, 'GNM': 0.565}
# }
#
# # 创建画布
# fig, axs = plt.subplots(1, 3, figsize=(14, 4))
#
# # 不显示右边和上边的线条
# for ax in axs:
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
# # 为每个数据集选择颜色
# colors_datasets = {
#     '300': ['#b9e3b2', '#74c476', '#74c476'],
#     '362': ['#b9e3b2', '#74c476', '#74c476'],
#     '364': ['#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#b9e3b2', '#74c476', '#74c476']
# }
#
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
#     # 排序方法和对应的值
#     # sorted_values = sorted(zip(values, methods))  # 不再使用 reverse=True
#     # values, methods = zip(*sorted_values)
#
#     #bars = ax.barh(np.arange(num_methods), values[::-1], height=0.5, color=colors)  # 使用自定义颜色
#     ax.set_title(f'{dataset} datasets', fontsize=18)
#     ax.tick_params(axis='y', rotation=0, labelsize=10)  # 默认标签不旋转
#     ax.set_xlim(0.4, 0.9)  # 设置横坐标范围
#     ax.set_xticks(np.arange(0.4, 0.91, 0.1))  # 设置横坐标刻度
#     ax.xaxis.set_tick_params(width=1, length=5, pad=10, direction='in', labeltop=False)  # 调整刻度参数位置
#     ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 不显示 x 轴刻度及其标签
#
#     ax.set_xlabel('Average PCCs', fontsize=14)  # 标注坐标轴标签
#     ax.set_yticks(np.arange(num_methods))
#     ax.set_yticklabels(methods[::-1],rotation=45, fontsize=12)
#
#     if i == 0:
#         ax.set_xlim(0.5, 0.817)
#         bars = ax.barh(np.arange(num_methods), values[::-1], height=0.3, color=colors)  # 使用自定义颜色
#     if i == 1:
#         ax.set_xlim(0.6, 0.796)
#         bars = ax.barh(np.arange(num_methods), values[::-1], height=0.3, color=colors)  # 使用自定义颜色
#     if i == 2:
#         ax.set_xlim(0.5, 0.794)
#         bars = ax.barh(np.arange(num_methods), values[::-1], height=0.6, color=colors)  # 使用自定义颜色
#
#     # 在每个柱子的顶端显示数值
#     for j, value in enumerate(values[::-1]):
#         ax.text(value, j, f'{value:.3f}', ha='right', va='center', fontsize=10)
#
# # 调整布局
# plt.subplots_adjust(wspace=0.4)
#
# # 保存和显示图形
# # plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig1_2_2.png', dpi=800, bbox_inches='tight')
# plt.show()
