
# =============================== 版本一：折线图 ================================
'''
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time',
                 'GSE82187', 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2',
                 'GSE89232', 'GSE94820']
rf_ba_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
gbdt_ba_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
svm_ba_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

rf_ba_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
gbdt_ba_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
svm_ba_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制 RF 方法的结果
ax.plot(accession_ids, rf_ba_rk4, marker='o', linestyle='-', label='RF (RK-4)')
ax.plot(accession_ids, rf_ba_rk2, marker='o', linestyle='--', label='RF (RK-2)')

# 绘制 GBDT 方法的结果
ax.plot(accession_ids, gbdt_ba_rk4, marker='s', linestyle='-', label='GBDT (RK-4)')
ax.plot(accession_ids, gbdt_ba_rk2, marker='s', linestyle='--', label='GBDT (RK-2)')

# 绘制 SVM 方法的结果
ax.plot(accession_ids, svm_ba_rk4, marker='^', linestyle='-', label='SVM (RK-4)')
ax.plot(accession_ids, svm_ba_rk2, marker='^', linestyle='--', label='SVM (RK-2)')

# 设置图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# 设置 y 轴标签
ax.set_ylabel('Balanced Accuracy (BA)',fontsize=16)

# 旋转 x 轴刻度标签
plt.xticks(rotation=45, ha='right')
plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_1.png' , dpi=500, bbox_inches='tight')

# 显示图形
plt.tight_layout()
plt.show()
'''


'''
# =============================== 版本二：堆积柱状图 ================================

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time',
                 'GSE82187', 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2',
                 'GSE89232', 'GSE94820']
rf_ba_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
gbdt_ba_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
svm_ba_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

rf_ba_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
gbdt_ba_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
svm_ba_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, ax = plt.subplots()

# 计算柱状图的宽度
bar_width = 0.35

# 设置 x 轴位置
bar_positions1 = range(len(accession_ids))
bar_positions2 = [pos + bar_width for pos in bar_positions1]

# 绘制 RF 方法的结果
ax.bar(bar_positions1, rf_ba_rk4, width=bar_width, label='RF (RK-4)')
ax.bar(bar_positions2, rf_ba_rk2, width=bar_width, label='RF (RK-2)')

# 绘制 GBDT 方法的结果
ax.bar(bar_positions1, gbdt_ba_rk4, width=bar_width, label='GBDT (RK-4)')
ax.bar(bar_positions2, gbdt_ba_rk2, width=bar_width, label='GBDT (RK-2)')

# 绘制 SVM 方法的结果
ax.bar(bar_positions1, svm_ba_rk4, width=bar_width, label='SVM (RK-4)')
ax.bar(bar_positions2, svm_ba_rk2, width=bar_width, label='SVM (RK-2)')

# 设置 x 轴标签
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions1])
ax.set_xticklabels(accession_ids, rotation=45, ha='right')

# 设置 y 轴标签
ax.set_ylabel('Balanced Accuracy',fontsize=16)

# 设置 y 轴范围从0.2开始
ax.set_ylim(0.25, 1.0)

# 设置图例位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_2.png' , dpi=500, bbox_inches='tight')

# 显示图形
plt.tight_layout()
plt.show()
'''


'''
# =============================== 版本三：堆积柱状图 + 折线图 ================================
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用 Times New Roman 字体显示

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time',
                 'GSE82187', 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2',
                 'GSE89232', 'GSE94820']
rf_ba_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
gbdt_ba_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
svm_ba_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

rf_ba_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
gbdt_ba_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
svm_ba_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, ax = plt.subplots()

# 计算柱状图的宽度
bar_width = 0.35

# 设置 x 轴位置
bar_positions1 = range(len(accession_ids))
bar_positions2 = [pos + bar_width for pos in bar_positions1]

# 绘制 RF 方法的结果
ax.plot(bar_positions1, rf_ba_rk4,  marker='^', linestyle='-', label='RF (RK-4)')
ax.plot(bar_positions2, rf_ba_rk2, marker='o', linestyle='-', label='RF (RK-2)')

# 绘制 GBDT 方法的结果
ax.bar(bar_positions1, gbdt_ba_rk4, width=bar_width, label='GBDT (RK-4)')
ax.bar(bar_positions2, gbdt_ba_rk2, width=bar_width, label='GBDT (RK-2)')

# 绘制 SVM 方法的结果
ax.bar(bar_positions1, svm_ba_rk4, width=bar_width, label='SVM (RK-4)')
ax.bar(bar_positions2, svm_ba_rk2, width=bar_width, label='SVM (RK-2)')

# 设置 x 轴标签
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions1])
ax.set_xticklabels(accession_ids, rotation=45, ha='right')

# 设置 y 轴标签
ax.set_ylabel('Balanced Accuracy',fontsize=16)

# 设置 y 轴范围从0.2开始
ax.set_ylim(0.2, 1.0)

# 设置图例位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_3.png' , dpi=500, bbox_inches='tight')

# 显示图形
plt.tight_layout()
plt.show()

'''

'''
#   ================================== 将 RF 计算结果用折线图可视化
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用 Times New Roman 字体显示

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time',
                 'GSE82187', 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2',
                 'GSE89232', 'GSE94820']
rf_ba_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
gbdt_ba_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
svm_ba_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

rf_ba_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
gbdt_ba_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
svm_ba_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, ax = plt.subplots()

# 计算柱状图的宽度
bar_width = 0.35

# 设置 x 轴位置
bar_positions1 = range(len(accession_ids))
bar_positions2 = [pos + bar_width for pos in bar_positions1]

# 绘制 RF 方法的结果（折线）
ax.plot(bar_positions1, rf_ba_rk4, marker='^', linestyle='-',color='orangered',  label='RF (RK-4)')
ax.plot(bar_positions2, rf_ba_rk2, marker='o', linestyle='-',  color='darkorange',label='RF (RK-2)')

# 绘制 GBDT 方法的结果（柱状图）
ax.bar(bar_positions1, gbdt_ba_rk4, width=bar_width, color='rosybrown', label='GBDT (RK-4)')
ax.bar(bar_positions2, gbdt_ba_rk2, width=bar_width, color='lightsteelblue',label='GBDT (RK-2)')

# 绘制 SVM 方法的结果（柱状图）
ax.bar(bar_positions1, svm_ba_rk4, width=bar_width,  color='salmon',  label='SVM (RK-4)')
ax.bar(bar_positions2, svm_ba_rk2, width=bar_width, color='burlywood', label='SVM (RK-2)')

# 设置 x 轴标签
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions1])
ax.set_xticklabels(accession_ids, rotation=45, ha='right')

# 设置 y 轴标签
ax.set_ylabel('Balanced Accuracy ', fontsize=16)

# 设置 y 轴范围从0.2开始
ax.set_ylim(0.2, 1.0)

# 设置图例位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_4.png' , dpi=500, bbox_inches='tight')

# 显示图形
plt.tight_layout()
plt.show()
'''

'''
# =========================   将 RF、GBDT 和 SVM 方法的 RK-4 计算结果用折线图可视化
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time',
                 'GSE82187', 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2',
                 'GSE89232', 'GSE94820']

rf_ba_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
gbdt_ba_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
svm_ba_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

rf_ba_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
gbdt_ba_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
svm_ba_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制 RF 方法的 RK-4 计算结果
ax.plot(accession_ids, rf_ba_rk4, marker='o', linestyle='-',  color='darkorange',label='RF (RK-4)')

# 绘制 GBDT 方法的 RK-4 计算结果
ax.plot(accession_ids, gbdt_ba_rk4, marker='x', linestyle='-', color='lightseagreen', label='GBDT (RK-4)')

# 绘制 SVM 方法的 RK-4 计算结果
ax.plot(accession_ids, svm_ba_rk4, marker='^', linestyle='-' ,color='royalblue',label='SVM (RK-4)')

# 添加柱状图，表示 RF 方法的 RK-2 计算结果
bar_width = 0.2
bar_positions_rf = [i - bar_width for i in range(len(accession_ids))]
ax.bar(bar_positions_rf, rf_ba_rk2, width=bar_width, color='darkolivegreen', alpha=0.8, label='RF (RK-2)')

# 添加柱状图，表示 GBDT 方法的 RK-2 计算结果
bar_positions_gbdt = [i for i in range(len(accession_ids))]
ax.bar(bar_positions_gbdt, gbdt_ba_rk2, width=bar_width, color='darkorchid',alpha=0.6, label='GBDT (RK-2)')

# 添加柱状图，表示 SVM 方法的 RK-2 计算结果
bar_positions_svm = [i + bar_width for i in range(len(accession_ids))]
ax.bar(bar_positions_svm, svm_ba_rk2, width=bar_width,  color='darkgoldenrod', alpha=0.6, label='SVM (RK-2)')

# 设置图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# 设置 y 轴标签
ax.set_ylabel('Balanced Accuracy ', fontsize=16)

# 设置 y 轴范围从0.2开始
ax.set_ylim(0.2, 1.0)

# 旋转 x 轴刻度标签
plt.xticks(rotation=45, ha='right')
plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_5.png', dpi=500, bbox_inches='tight')
plt.show()

'''



'''
#创建一个两行三列的图，其中每个子图中都有水平条形图，表示每个Accession ID 的不同方法（RF、GBDT、SVM）的平衡准确度（Balanced Accuracy）。第一行是RK-4的结果，第二行是RK-2的结果
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time', 'GSE82187',
                 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2', 'GSE89232', 'GSE94820']

ba_rf_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
ba_gbdt_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
ba_svm_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

ba_rf_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
ba_gbdt_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
ba_svm_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# RK-4结果绘图
axs[0, 0].barh(accession_ids, ba_rf_rk4, color='blue', label='RF')
# axs[0, 0].barh(range(len(accession_ids)), ba_rf_rk4, color='blue', label='RF')
axs[0, 0].set_title('RF (RK-4)')
#axs[0, 0].set_xlabel('Balanced Accuracy ')
axs[0, 0].set_xlim(0.2, 1)
# axs[0, 0].tick_params(axis='y', which='both', length=0)  # 去除y轴刻度线

#axs[0, 1].barh(accession_ids, ba_gbdt_rk4, color='green', label='GBDT')
axs[0, 1].barh(range(len(accession_ids)), ba_gbdt_rk4, color='green', label='GBDT')
axs[0, 1].set_title('GBDT (RK-4)')
# axs[0, 1].set_xlabel('Balanced Accuracy')
axs[0, 1].set_xlim(0.2, 1)
axs[0, 1].set_yticklabels([])
axs[0, 1].tick_params(axis='y', which='both', length=0)  # 去除y轴刻度线

# axs[0, 2].barh(accession_ids, ba_svm_rk4, color='red', label='SVM')
axs[0, 2].barh(range(len(accession_ids)), ba_svm_rk4, color='red', label='SVM')
axs[0, 2].set_title('SVM (RK-4)')
# axs[0, 2].set_xlabel('Balanced Accuracy')
axs[0, 2].set_xlim(0.2, 1)
axs[0, 2].set_yticklabels([])
axs[0, 2].tick_params(axis='y', which='both', length=0)  # 去除y轴刻度线

# RK-2结果绘图
axs[1, 0].barh(accession_ids, ba_rf_rk2, color='blue', label='RF')
axs[1, 0].set_title('RF (RK-2)')
axs[1, 0].set_xlabel('Balanced Accuracy')
axs[1, 0].set_xlim(0.2, 1)
# axs[1, 0].tick_params(axis='y', which='both', length=0)  # 去除y轴刻度线

axs[1, 1].barh(accession_ids, ba_gbdt_rk2, color='green', label='GBDT')
axs[1, 1].set_title('GBDT (RK-2)')
axs[1, 1].set_xlabel('Balanced Accuracy')
axs[1, 1].set_xlim(0.2, 1)
axs[1, 1].set_yticklabels([])
axs[1, 1].tick_params(axis='y', which='both', length=0)  # 去除y轴刻度线

axs[1, 2].barh(accession_ids, ba_svm_rk2, color='red', label='SVM')
axs[1, 2].set_title('SVM (RK-2)')
axs[1, 2].set_xlabel('Balanced Accuracy')
axs[1, 2].set_xlim(0.2, 1)
axs[1, 2].set_yticklabels([])
axs[1, 2].tick_params(axis='y', which='both', length=0)  # 去除y轴刻度线

# 设置图例
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')

# 调整布局
plt.tight_layout()

# 设置刻度字号
plt.xticks(fontsize=14)
plt.yticks(fontsize=16)

# 显示图形
plt.show()

'''


'''
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time', 'GSE82187',
                 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h3', 'GSE84133 m1', 'GSE84133 m2', 'GSE89232', 'GSE94820']

# RK-4结果
ba_rf_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
ba_gbdt_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
ba_svm_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

# RK-2结果
ba_rf_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
ba_gbdt_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
ba_svm_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

# 柱状图宽度和间隙调整
bar_width = 0.35
bar_gap = 0.0

# RF结果绘图
axs[0].bar(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_rf_rk4, width=bar_width, color='#9281DD', label='RF (RK-4)')
axs[0].bar(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_rf_rk2, width=bar_width, color='lightblue', label='RF (RK-2)')
axs[0].set_xticks(np.arange(len(accession_ids)))
axs[0].set_xticklabels(accession_ids, rotation=45, fontsize=12, ha='right')  # 倾斜45度，字号12，对齐方式右对齐
axs[0].set_ylabel('Balanced Accuracy', fontsize=18)  # 设置y轴标签字体大小
axs[0].set_ylim(0.4, 1)
# axs[0].tick_params(axis='both', which='major', labelsize=12,  width=1, length=5, direction='in')  # 设置刻度字号、线的粗细和方向
axs[0].tick_params(axis='x', which='both', width=1, length=5, labelsize=12)  # 设置x轴刻度在内部显示，并增加粗细
axs[0].tick_params(axis='y', which='both', direction='in', width=1)  # 设置y轴粗细
axs[0].spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
axs[0].spines['right'].set_visible(False)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), shadow=True, ncol=2)

# GBDT结果绘图
# GBDT结果绘图'#D2ADAB'
axs[1].bar(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_gbdt_rk4, width=bar_width,  color='#96d1c6', label='GBDT (RK-4)')
axs[1].bar(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_gbdt_rk2, width=bar_width, color='#F5E09B', label='GBDT (RK-2)')
axs[1].set_xticks(np.arange(len(accession_ids)))
axs[1].set_xticklabels(accession_ids, rotation=45, fontsize=12, ha='right')  # 倾斜45度，字号12，对齐方式右对齐
# axs[1].set_ylabel('Balanced Accuracy (GBDT)', fontsize=14)  # 设置y轴标签字体大小
axs[1].set_ylim(0.45, 1)
# axs[1].tick_params(axis='both', which='major', labelsize=12,  width=1, length=5, direction='in')  # 设置刻度字号、线的粗细和方向
axs[1].tick_params(axis='x', which='both', width=1, length=5, labelsize=12)  # 设置x轴刻度在内部显示，并增加粗细
axs[1].tick_params(axis='y', which='both', direction='in', width=1)  # 设置y轴粗细
axs[1].spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
axs[1].spines['right'].set_visible(False)
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), shadow=True, ncol=2)

# SVM结果绘图
axs[2].bar(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_svm_rk4, width=bar_width, color='#82969D', label='SVM (RK-4)')
axs[2].bar(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_svm_rk2, width=bar_width, color='pink', label='SVM (RK-2)')
axs[2].set_xticks(np.arange(len(accession_ids)))
axs[2].set_xticklabels(accession_ids, rotation=45, fontsize=12, ha='right')  # 倾斜45度，字号12，对齐方式右对齐
# axs[2].set_ylabel('Balanced Accuracy (SVM)', fontsize=14)  # 设置y轴标签字体大小
axs[2].set_ylim(0.2, 1)
# axs[2].tick_params(axis='both', which='major', labelsize=12, width=1, length=5, direction='in')  # 设置刻度字号、线的粗细和方向
axs[2].tick_params(axis='x', which='both', width=1, length=5, labelsize=12)  # 设置x轴刻度在内部显示，并增加粗细
axs[2].tick_params(axis='y', which='both', direction='in', width=1)  # 设置y轴粗细
axs[2].spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
axs[2].spines['right'].set_visible(False)
axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), shadow=True, ncol=2)

# 添加数值标签
for ax in axs:
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 3), ha='center', va='bottom', fontsize=7, rotation=90)

# 调整布局
plt.tight_layout()
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_6.png', dpi=500, bbox_inches='tight')
plt.show()
'''


# #创建一个一行三列的大图，每个子图都绘制RF、GBDT和SVM三种方法的RK-4和RK-2结果，使用横向的双轴柱状图表示。每个子图添加图例以区分RK-4和RK-2的结果
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams["font.family"] = "Times New Roman"
#
# # 数据
# accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time', 'GSE82187',
#                  'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h4', 'GSE84133 m1', 'GSE84133 m2', 'GSE89232', 'GSE94820']
#
# # RK-4结果
# ba_rf_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
# ba_gbdt_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
# ba_svm_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]
#
# # RK-2结果
# ba_rf_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
# ba_gbdt_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
# ba_svm_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]
#
# # 创建画布和子图
# fig, axs = plt.subplots(1, 3, figsize=(16, 6))
#
# # 柱状图宽度和间隙调整
# bar_width = 0.4
# bar_gap = 0.
#
# # RF结果绘图,
# axs[0].barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_rf_rk4, height=bar_width, color='#9281DD', label='RF (RK-4)')
# axs[0].barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_rf_rk2, height=bar_width, color='lightblue', label='RF (RK-2)')
# axs[0].set_yticks(np.arange(len(accession_ids)))
# axs[0].set_yticklabels(accession_ids, rotation=45, fontsize=16, ha='right')  # 倾斜45度，字号12，对齐方式右对齐
# axs[0].set_xlabel('Balanced Accuracy', fontsize=20)  # 设置x轴标签字体大小
# axs[0].set_xlim(0.4, 1)
# axs[0].tick_params(axis='x', which='both', direction='in', width=1, length=5,labelsize=16)   # 设置x轴刻度在内部显示，并增加粗细
# axs[0].tick_params(axis='y', which='both', width=1)  # 设置y轴粗细
# axs[0].spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
# axs[0].spines['right'].set_visible(False)
# axs[0].legend(loc='upper center', bbox_to_anchor=(0.42, 1.03), shadow=True, ncol=2, fontsize=12, frameon=False)  # 去掉图例的边框
#
# # GBDT结果绘图,
# axs[1].barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_gbdt_rk4, height=bar_width, color='#96d1c6', label='GBDT (RK-4)')
# axs[1].barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_gbdt_rk2, height=bar_width, color='#F5E09B', label='GBDT (RK-2)')
# axs[1].set_yticks(axs[0].get_yticks())  # 与第一个子图保持一致的刻度线
# axs[1].set_yticklabels([])
# axs[1].set_xlabel('Balanced Accuracy', fontsize=20)  # 设置x轴标签字体大小
# axs[1].set_xlim(0.4, 1)
# axs[1].tick_params(axis='x', which='both', direction='in', width=1, length=5, labelsize=16)  # 设置x轴刻度在内部显示，并增加粗细
# axs[1].tick_params(axis='y', which='both', width=1)  # 设置y轴粗细
# axs[1].spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
# axs[1].spines['right'].set_visible(False)
# axs[1].legend(loc='upper center', bbox_to_anchor=(0.45, 1.03), shadow=True, ncol=2, fontsize=12,  frameon=False)  # 去掉图例的边框
#
# # SVM结果绘图
# axs[2].barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_svm_rk4, height=bar_width, color='#68789e', label='SVM (RK-4)')
# axs[2].barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_svm_rk2, height=bar_width, color='pink', label='SVM (RK-2)')
# axs[2].set_yticks(axs[0].get_yticks())  # 与第一个子图保持一致的刻度线
# axs[2].set_yticklabels([])
# axs[2].set_xlabel('Balanced Accuracy', fontsize=20)  # 设置x轴标签字体大小
# axs[2].set_xlim(0.2, 1)
# axs[2].tick_params(axis='x', which='both', direction='in', width=1, length=5, labelsize=16)  # 设置x轴刻度在内部显示，并增加粗细
# axs[2].tick_params(axis='y', which='both', width=1)  # 设置y轴粗细
# axs[2].spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
# axs[2].spines['right'].set_visible(False)
# axs[2].legend(loc='upper center', bbox_to_anchor=(0.43, 1.03), shadow=True, ncol=2, fontsize=12, frameon=False)  # 去掉图例的边框
#
# # 添加数字标签
# for ax in axs:
#     for i, bar in enumerate(ax.patches):
#         ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', ha='left', va='center', fontsize=10)
#
# # 调整布局
# plt.tight_layout()
# # plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig4_7.png', dpi=800, bbox_inches='tight')
# plt.show()


# ======================将RF结果、GBDT结果和SVM结果分别绘制成三个单独的图像的代码
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# 数据
accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time', 'GSE82187',
                 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h4', 'GSE84133 m1', 'GSE84133 m2', 'GSE89232', 'GSE94820']

# RK-4结果
ba_rf_rk4 = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
ba_gbdt_rk4 = [0.926, 0.954, 0.826, 0.949, 0.972, 0.773, 0.810, 0.830, 0.930, 0.877, 0.859, 0.925, 0.968]
ba_svm_rk4 = [0.505, 0.649, 0.613, 0.672, 0.808, 0.464, 0.586, 0.738, 0.712, 0.803, 0.687, 0.448, 0.611]

# RK-2结果
ba_rf_rk2 = [0.555, 0.443, 0.830, 0.852, 0.928, 0.768, 0.525, 0.873, 0.907, 0.868, 0.841, 0.814, 0.928]
ba_gbdt_rk2 = [0.533, 0.534, 0.798, 0.862, 0.924, 0.774, 0.521, 0.822, 0.888, 0.870, 0.821, 0.910, 0.931]
ba_svm_rk2 = [0.285, 0.406, 0.581, 0.432, 0.762, 0.366, 0.443, 0.739, 0.801, 0.764, 0.812, 0.285, 0.426]

# 创建画布和子图
fig, ax = plt.subplots(figsize=(8, 7))
# 柱状图宽度和间隙调整
bar_width = 0.41
bar_gap = 0.
# RF结果绘图,
# ax.barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_rf_rk4, height=bar_width, color='#9281DD', label='RF (RK-4)')
# ax.barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_rf_rk2, height=bar_width, color='lightblue', label='RF (RK-2)')

ax.barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_rf_rk4, height=bar_width, color='#68789e', label='RF (RK-4)')
ax.barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_rf_rk2, height=bar_width, color='pink', label='RF (RK-2)')
ax.set_yticks(np.arange(len(accession_ids)))
ax.set_yticklabels(accession_ids, rotation=30, fontsize=18, ha='right')  # 倾斜45度，字号12，对齐方式右对齐
ax.set_xlabel('Balanced Accuracy', fontsize=24) # 设置x轴标签字体大小
ax.set_xlim(0.4, 1)
ax.tick_params(axis='x', which='both', direction='in', width=1, length=5,labelsize=18)   # 设置x轴刻度在内部显示，并增加粗细
ax.tick_params(axis='y', which='both',  pad=0,width=1)  # 设置y轴粗细
ax.spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的边界线
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), shadow=True, ncol=2, fontsize=14, frameon=False)  # 去掉图例的边框

# 添加数字标签
for i, bar in enumerate(ax.patches):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', ha='left', va='center', fontsize=14)

# 调整布局
plt.tight_layout()

# 保存图像
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/RF_results.png', dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/RF_results.svg', dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/RF_results.pdf', dpi=800, bbox_inches='tight')

plt.show()

# GBDT结果绘图,
fig, ax = plt.subplots(figsize=(8, 7))
# ax.barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_gbdt_rk4, height=bar_width, color='#96d1c6', label='GBDT (RK-4)')
# ax.barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_gbdt_rk2, height=bar_width, color='#F5E09B', label='GBDT (RK-2)')
ax.barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_gbdt_rk4, height=bar_width, color='#68789e', label='GBDT (RK-4)')
ax.barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_gbdt_rk2, height=bar_width, color='pink', label='GBDT (RK-2)')
ax.set_yticks(np.arange(len(accession_ids)))
ax.set_yticklabels(accession_ids, rotation=0, fontsize=16, ha='right')
ax.set_xlabel('Balanced Accuracy',fontsize=20)
ax.set_xlim(0.5, 1)
ax.tick_params(axis='x', which='both', direction='in', width=1, length=5,labelsize=16)
ax.tick_params(axis='y', which='both', width=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), shadow=True, ncol=2, fontsize=14, frameon=False)

for i, bar in enumerate(ax.patches):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', ha='left', va='center', fontsize=14)

plt.tight_layout()
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/GBDT_results.png', dpi=800, bbox_inches='tight')
plt.show()

# SVM结果绘图
fig, ax = plt.subplots(figsize=(8, 7))

ax.barh(np.arange(len(accession_ids)) - (bar_width + bar_gap) / 2, ba_svm_rk4, height=bar_width, color='#68789e', label='SVM (RK-4)')
ax.barh(np.arange(len(accession_ids)) + (bar_width + bar_gap) / 2, ba_svm_rk2, height=bar_width, color='pink', label='SVM (RK-2)')
ax.set_yticks(np.arange(len(accession_ids)))
ax.set_yticklabels(accession_ids, rotation=0, fontsize=16, ha='right')
ax.set_xlabel('Balanced Accuracy', fontsize=20)
ax.set_xlim(0.25, 0.85)
ax.tick_params(axis='x', which='both', direction='in', width=1, length=5, labelsize=16)
ax.tick_params(axis='y', which='both', width=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), shadow=True, ncol=2, fontsize=14, frameon=False)

for i, bar in enumerate(ax.patches):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', ha='left', va='center', fontsize=14)

plt.tight_layout()
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/SVM_results.png', dpi=800, bbox_inches='tight')
plt.show()
