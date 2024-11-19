import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示

# cutoff = np.arange(0.05, 0.755, 0.018)
# cutoff = [0.211 ,0.246 ,0.281 ,0.316 ,0.351 ,0.387 ,0.422 ,0.457 ,0.492 ,
#           0.527 ,0.562 ,0.598 ,0.633 ,0.668 ,0.703 ]
cutoff = [0.1, 0.141, 0.176,
          0.211 , 0.246 ,0.281 ,    0.316 , 0.351 ,0.387 ,0.422 ,
          0.457 ,0.492 , 0.527 ,0.562 ,
          0.598 ,0.633 ,    0.668 ,0.703 ,
          0.8, 0.9, 1.0]
print('cutoff:', cutoff)
print(len(cutoff))
# interval_thrs_1 = [4.20 ,3.60 ,2.60 ,2.50 ,2.50 ,2.40 ,2.30 ,2.30 ,2.20 ,                    2.10 ,2.00 ,1.90 ,1.90 ,1.70 ,1.70 ]
# interval_thrs_2 = [5.00 ,4.60 ,3.80 ,3.80 ,3.50 ,3.30 ,3.00 ,2.80 ,2.80 ,                 2.60 ,2.50 ,2.30 ,2.30 ,2.30 ,2.30 ]
interval_thrs_1 = [ 4.7, 4.6 ,4.5,
                   3.9 ,3.4,3,        2.78 ,2.48 ,2.38 ,2.23 ,
                   2.13, 2.03, 1.93 ,1.85,
                   1.79,1.73,   1.7,1.69 ,
                   1.68,1.68,1.68]
interval_thrs_2 = [4.7, 4.6 ,4.5,
                   4.4,4.3 ,4.1 ,    3.8 ,3.5 ,3.25 , 3.05,
                   2.85, 2.65, 2.45, 2.3,
                   2.2,2.19,2.19,2.19,
                   2.18,2.18,2.18]
print(len(interval_thrs_1))
print(len(interval_thrs_2))
phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_0.png')
phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_1.png')
phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_2.png')

plt.figure(figsize=(8, 6))
plt.ylabel('coupling strength', fontsize=26, labelpad=10)
plt.xlabel('filtration radius', fontsize=26)
# 设置绘图的坐标轴范围
plt.axis([0.10, 1.0, 1, 5])
ax = plt.gca()
# 设置纵坐标刻度间隔为0.1
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# 设置纵坐标刻度间隔为0.1
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

# 使用自定义的浅灰色来设置y轴位置为0.35和0.4的虚线并标记刻度
custom_light_gray = (0.7, 0.7, 0.7)  # 自定义浅灰色的RGB值
ax.axhline(1.7, color=custom_light_gray, linestyle='dashed')
ax.axhline(2.2, color=custom_light_gray, linestyle='dashed')
ax.axhline(4.5, color=custom_light_gray, linestyle='dashed')
ax.axhline(4.7, color=custom_light_gray, linestyle='dashed')
yticks = [1.7,2.2,4.5,4.7]
ytick_labels = ['1.7','2.2','4.5','4.7']
for i, tick_label in enumerate(ytick_labels):
    if tick_label in ['1.7','2.2','4.5','4.7']:
        plt.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=17)
    else:
        plt.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=17)

plt.imshow(phase_1, extent=(0.42, 0.59,3.55, 4.4))
plt.imshow(phase_2, extent=(0.61, 0.78,3.55, 4.4))
plt.imshow(phase_3, extent=(0.8, 0.97,3.55, 4.4))

ax = plt.gca()
ax.set_aspect(0.2)  # 纵轴的单位长度是横轴单位长度的0.65倍
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 横排
plt.text(0.502, 3.9, 'I', fontsize=20)
plt.text(0.676, 3.9, 'II', fontsize=20)
plt.text(0.862, 3.902,  'III', fontsize=20)
# 纵排
plt.text(0.928, 1.3, 'I', fontsize=20)
plt.text(0.922, 1.85, 'II', fontsize=20)
plt.text(0.916, 2.5, 'III', fontsize=20)
#
plt.text(0.7, 3., 'healthy', fontsize=22)

ax.fill_between(cutoff, 0, interval_thrs_1, facecolor='green', alpha=0.2, linewidth=0)  # 区域的下边界为0，上边界为interval_thrs_1
ax.fill_between(cutoff, interval_thrs_1, interval_thrs_2, facecolor='red', alpha=0.2, edgecolor="k", linewidth=0)
ax.fill_between(cutoff, interval_thrs_2, 5, edgecolor="k", alpha=0.2, linewidth=0.0)

plt.tight_layout()
plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h_2.png',  dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_egg_s.png',  dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.pdf',  dpi=1200, bbox_inches='tight')

plt.show()

#======================1.5-7.5
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.ticker as ticker
#
# plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示
#
# # cutoff = np.arange(0.05, 0.755, 0.018)
# # cutoff = [0.211 ,0.246 ,0.281 ,0.316 ,0.351 ,0.387 ,0.422 ,0.457 ,0.492 ,
# #           0.527 ,0.562 ,0.598 ,0.633 ,0.668 ,0.703 ]
# cutoff = [0.1, 0.125, 0.141,0.156, 0.176,
#           0.211 , 0.246 , 0.265, 0.281 ,0.316 , 0.351 ,0.387 ,0.422 ,
#           0.457 ,0.492 ,    0.527 ,0.562 ,0.598 ,0.633 ,0.668 ,0.703 ,
#           0.8, 0.9, 1.0]
# # couple = np.concatenate((couple,[20]))
# print('cutoff:', cutoff)
# print(len(cutoff))
#
# # interval_thrs_1 = [4.20 ,3.60 ,2.60 ,2.50 ,2.50 ,2.40 ,2.30 ,2.30 ,2.20 ,
# #                    2.10 ,2.00 ,1.90 ,1.90 ,1.70 ,1.70 ]
# # interval_thrs_2 = [5.00 ,4.60 ,3.80 ,3.80 ,3.50 ,3.30 ,3.00 ,2.80 ,2.80 ,
# #                 2.60 ,2.50 ,2.30 ,2.30 ,2.30 ,2.30 ]
# interval_thrs_1 = [7.35,7.2, 7,6.85,6.65,
#                 4.50 ,3.60,3.2,2.90,2.65 ,2.50 ,2.40 ,2.30 ,
#                    2.25, 2.175, 2.10 ,2.02,1.95,1.90,1.80,1.75 ,
#                    1.7,1.65,1.60]
# interval_thrs_2 = [7.35, 7.2, 7 , 6.85,6.65,
#                    6.0,4.80 , 4.3 ,  4.1 ,3.80 ,3.50 ,3.30 ,3.10 ,
#                    2.9,2.80,2.60,2.50,2.35,2.31,2.30,2.29,      2.28 ,2.27,2.26]
# print(len(interval_thrs_1))
# print(len(interval_thrs_2))
#
# phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_0.png')
# phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_1.png')
# phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_2.png')
#
# plt.figure(figsize=(8, 6))
#
# plt.ylabel('coupling strength', fontsize=26, labelpad=10)
# plt.xlabel('filtration radius', fontsize=26)
#
# # 设置绘图的坐标轴范围
# plt.axis([0.10, 1.0, 1, 7.5])
# ax = plt.gca()
# # 设置纵坐标刻度间隔为0.1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# # 设置纵坐标刻度间隔为0.1
# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
#
# plt.imshow(phase_1, extent=( 0.33, 0.53,4.25, 6.15))
# plt.imshow(phase_2, extent=(0.55, 0.75,4.25, 6.15))
# plt.imshow(phase_3, extent=(0.77, 0.97,4.25, 6.15))
#
# ax = plt.gca()
# ax.set_aspect(0.1)  # 纵轴的单位长度是横轴单位长度的0.65倍
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
#
# # 横排
# plt.text(0.425, 5.1, 'I', fontsize=20)
# plt.text(0.64, 5.1, 'II', fontsize=20)
# plt.text(0.855, 5.05,  'III', fontsize=20)
# # 纵排
# plt.text(0.928, 1.2, 'I', fontsize=20)
# plt.text(0.922, 1.85, 'II', fontsize=20)
# plt.text(0.916, 2.6, 'III', fontsize=20)
#
# ax.fill_between(cutoff, 0, interval_thrs_1, facecolor='green', alpha=0.2, linewidth=0)  # 区域的下边界为0，上边界为interval_thrs_1
# ax.fill_between(cutoff, interval_thrs_1, interval_thrs_2, facecolor='red', alpha=0.2, edgecolor="k", linewidth=0)
# ax.fill_between(cutoff, interval_thrs_2, 7.5, edgecolor="k", alpha=0.2, linewidth=0.0)
#
# # 使用自定义的浅灰色来设置y轴位置为0.35和0.4的虚线并标记刻度
# custom_light_gray = (0.7, 0.7, 0.7)  # 自定义浅灰色的RGB值
# ax.axhline(1.7, color=custom_light_gray, linestyle='dashed')
# ax.axhline(2.2, color=custom_light_gray, linestyle='dashed')
# yticks = [1.7,2.2]
# ytick_labels = ['1.7','2.2']
# for i, tick_label in enumerate(ytick_labels):
#     if tick_label in ['1.7','2.2']:
#         plt.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=18)
#     else:
#         plt.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=16)
#
# plt.tight_layout()
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h_1.png',  dpi=1200, bbox_inches='tight')
# # plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_egg_s.png',  dpi=1200, bbox_inches='tight')
# # plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.pdf',  dpi=1200, bbox_inches='tight')
#
# plt.show()


# #=====================真实===================
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.ticker as ticker
#
# plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示
#
# # cutoff = np.arange(0.05, 0.755, 0.018)
# # cutoff = [0.211 ,0.246 ,0.281 ,0.316 ,0.351 ,0.387 ,0.422 ,0.457 ,0.492 ,
# #           0.527 ,0.562 ,0.598 ,0.633 ,0.668 ,0.703 ]
# cutoff = [0.1, 0.141, 0.176, 0.211 , 0.246 , 0.265, 0.281 ,0.316 , 0.351 ,0.387 ,0.422 ,
#           0.457 ,0.492 ,    0.527 ,0.562 ,0.598 ,0.633 ,0.668 ,0.703 ,     0.8, 0.9, 1.0]
# # couple = np.concatenate((couple,[20]))
# print('cutoff:', cutoff)
# print(len(cutoff))
#
# # interval_thrs_1 = [4.20 ,3.60 ,2.60 ,2.50 ,2.50 ,2.40 ,2.30 ,2.30 ,2.20 ,
# #                    2.10 ,2.00 ,1.90 ,1.90 ,1.70 ,1.70 ]
# # interval_thrs_2 = [5.00 ,4.60 ,3.80 ,3.80 ,3.50 ,3.30 ,3.00 ,2.80 ,2.80 ,
# #                 2.60 ,2.50 ,2.30 ,2.30 ,2.30 ,2.30 ]
# interval_thrs_1 = [14, 9, 6, 4.50 ,  3.60 ,3.2 ,2.90 ,2.65 ,2.50 ,2.40 ,2.30 ,
#                    2.25,2.175,2.10 ,2.02,1.95,1.90,1.80,1.75 ,     1.7,1.65,1.60]
# interval_thrs_2 = [100,50,11.2,6.50 ,4.80 , 4.3  ,  4.1 ,3.80 ,3.50 ,3.30 ,3.10 ,
#                    2.9,2.80,2.60,2.50,2.35,2.31,2.30,2.29,      2.28 ,2.27,2.26]
# print(len(interval_thrs_1))
# print(len(interval_thrs_2))
#
# phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_0.png')
# phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_1.png')
# phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\h_2.png')
#
# plt.figure(figsize=(8, 6))
#
# plt.ylabel('coupling strength', fontsize=26, labelpad=10)
# plt.xlabel('filtration radius', fontsize=26)
#
# # 设置绘图的坐标轴范围
# plt.axis([0.10, 1.0, 1, 7.5])
# ax = plt.gca()
# # 设置纵坐标刻度间隔为0.1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# # 设置纵坐标刻度间隔为0.1
# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
#
# plt.imshow(phase_1, extent=( 0.33, 0.53,5.25, 7.15))
# plt.imshow(phase_2, extent=(0.55, 0.75, 5.25,  7.15))
# plt.imshow(phase_3, extent=(0.77, 0.97,5.25,  7.15))
#
# ax = plt.gca()
# ax.set_aspect(0.1)  # 纵轴的单位长度是横轴单位长度的0.65倍
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
#
# # 横排
# plt.text(0.425, 6.1, 'I', fontsize=20)
# plt.text(0.64, 6.1, 'II', fontsize=20)
# plt.text(0.86, 6.05,  'III', fontsize=20)
# # 纵排
# plt.text(0.928, 1.2, 'I', fontsize=20)
# plt.text(0.922, 1.85, 'II', fontsize=20)
# plt.text(0.916, 2.6, 'III', fontsize=20)
#
# ax.fill_between(cutoff, 0, interval_thrs_1, facecolor='green', alpha=0.2, linewidth=0)  # 区域的下边界为0，上边界为interval_thrs_1
# ax.fill_between(cutoff, interval_thrs_1, interval_thrs_2, facecolor='red', alpha=0.2, edgecolor="k", linewidth=0)
# ax.fill_between(cutoff, interval_thrs_2, 7.5, edgecolor="k", alpha=0.2, linewidth=0.0)
#
# plt.tight_layout()
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.png',  dpi=1200, bbox_inches='tight')
# # plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_egg_s.png',  dpi=1200, bbox_inches='tight')
# # plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.pdf',  dpi=1200, bbox_inches='tight')
#
# plt.show()
