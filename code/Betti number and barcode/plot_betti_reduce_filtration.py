import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

betti0_npy,betti1_npy,betti2_npy = [],[],[]
for i in range(10):
    # filtration = round()
    betti_npy = np.load(fr'D:\python\result\EEG\npy/reduce_betti_{round((i+1)/10,1)}.npy')
    column0 = betti_npy[:, 0].tolist()
    column1 = betti_npy[:, 1].tolist()
    column2 = betti_npy[:, 2].tolist()
    betti0_npy.append(column0)
    betti1_npy.append(column1)
    betti2_npy.append(column2)
betti0_npy = np.array(betti0_npy).T
betti1_npy = np.array(betti1_npy).T
betti2_npy = np.array(betti2_npy).T

# 创建一个自定义的颜色映射，从蓝色（负值）到白色（零值附近），再到红色（正值）
colors = [(0, 0, 0.8), (0, 0.5, 1), (0.95, 1, 1), (1, 0.5, 0), (0.7, 0, 0)]  # RGBA tuples(r,g,b)
cmap_name = 'custom_blue_white_red'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# 创建一个图形和一个轴
plt.figure(figsize=(10, 8))  # 你可以根据需要调整图形大小

xticklabels = [str(i/10) for i in range(1,11)]  # x轴刻度标签
yticklabels = [str(i) for i in range(1,15)]  # y轴刻度标签
plt.gca().xaxis.set_tick_params(labelsize=28)  # x轴标签大小
plt.gca().yaxis.set_tick_params(labelsize=28)  # y轴标签大小

# 绘制热力图
ax1 = plt.subplot()
im1 = sns.heatmap(betti0_npy, cmap=cmap, center=0)  # 使用'viridis'颜色映射，但你可以选择其他颜色映射,cbar_kws={"label": "Betti 0 values"}
cb1 = im1.figure.axes[1]
cb1.tick_params(labelsize = 16)
ax1.set_xlabel('Filtration parameter',size = 20)
ax1.set_ylabel('Sample',size = 20)
ax1.set_xticklabels(xticklabels)  # 设置x轴刻度标签
ax1.set_yticklabels(yticklabels)  # 设置y轴刻度标签
# rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='k', lw=3, transform=ax1.transAxes)
# ax1.add_patch(rect)
ax1.tick_params(axis='x',labelsize = 18)
ax1.tick_params(axis='y',labelsize = 18)
plt.savefig(r'D:\python\result\EEG\npy/betti_reduce_0.png')
plt.show()

plt.figure(figsize=(10, 8))  # 你可以根据需要调整图形大小
ax2 = plt.subplot()
im2 = sns.heatmap(betti1_npy, cmap=cmap, center=0)  # 使用颜色映射，可以选择其他颜色映射,
cb2 = im2.figure.axes[1]
cb2.tick_params(labelsize = 16)
ax2.set_xlabel('Filtration parameter',size = 20)
ax2.set_ylabel('Sample',size = 20)
ax2.set_xticklabels(xticklabels)  # 设置x轴刻度标签
ax2.set_yticklabels(yticklabels)  # 设置y轴刻度标签
rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='k', lw=3, transform=ax2.transAxes)
ax2.add_patch(rect)
ax2.tick_params(axis='x',labelsize = 18)
ax2.tick_params(axis='y',labelsize = 18)
plt.savefig(r'D:\python\result\EEG\npy/betti_reduce_1.png')
plt.show()

plt.figure(figsize=(10, 8))  # 你可以根据需要调整图形大小
ax3 = plt.subplot()
im3 = sns.heatmap(betti2_npy, cmap=cmap, center=0)  # 使用'viridis'颜色映射，但你可以选择其他颜色映射,cbar_kws={"label": "Betti 0 values"}
cb3 = im3.figure.axes[1]
cb3.tick_params(labelsize = 16)
ax3.set_xlabel('Filtration parameter',size = 20)
ax3.set_ylabel('Sample',size = 20)
ax3.set_xticklabels(xticklabels)  # 设置x轴刻度标签
ax3.set_yticklabels(yticklabels)  # 设置y轴刻度标签
rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='k', lw=3, transform=ax3.transAxes)
ax3.add_patch(rect)
ax3.tick_params(axis='x',labelsize = 18)
ax3.tick_params(axis='y',labelsize = 18)
plt.savefig(r'D:\python\result\EEG\npy/betti_reduce_2.png')
plt.show()
