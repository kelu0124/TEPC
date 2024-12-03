#=======================抽象====================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示

# cutoff = np.arange(0.05, 0.755, 0.018)
cutoff = [0.1, 0.121, 0.161,
          0.201, 0.241,0.282,
          0.322,0.362,0.403,
          0.443,0.483 ,  0.523,0.563,
          0.604,0.644,0.684,0.725,0.765,0.805,          0.9,1.0]
print('cutoff:', cutoff)
print(len(cutoff))
# interval_thrs_1 = [6.00 ,4.50 ,3.50 ,2.60 ,2.50 ,2.40 ,2.30 ,2.20 ,2.10 ,2.00 ,1.90 ,
#                   1.80 ,1.70 ,1.69 ,1.69 ,1.68 ,1.68,1.67  ]
# interval_thrs_2 = [7.50 ,6.70 ,4.50 ,3.90 ,3.50 ,3.10 ,2.80 ,2.70 ,2.70 ,2.70 ,2.50 ,
#                   2.40 ,2.30 ,2.29 ,2.28 ,2.27 ,2.26,2.26]
interval_thrs_1 = [4.8,  4.75 ,4.6,
                   3.95 ,3.55 ,3.2 ,
                   2.9, 2.65,2.4,
                   2.2 ,2 ,1.85 ,1.75 ,
                   1.7 ,1.65 ,1.6 ,1.6 ,1.59 ,1.59 ,                  1.58, 1.58]
interval_thrs_2 = [4.8,  4.75 ,4.6,
                   4.4,4.15 ,3.9,
                   3.65,  3.4 ,3.2 ,
                   3.0 ,2.85 ,  2.70 ,2.60 ,
                   2.50 ,2.40 ,2.30 ,2.29 ,2.29 ,2.29 ,                  2.28 ,2.28 ]
print(len(interval_thrs_1))
print(len(interval_thrs_2))

# phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_0.png')
# phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_1.png')
# phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_2.png')
# phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_3.png')
# phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_4.png')
# phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_5.png')
phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_6.png')
phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_7.png')
phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_8.png')


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
ax.axhline(1.6, color=custom_light_gray, linestyle='dashed')
ax.axhline(2.3, color=custom_light_gray, linestyle='dashed')
ax.axhline(4.6, color=custom_light_gray, linestyle='dashed')
ax.axhline(4.8, color=custom_light_gray, linestyle='dashed')
yticks = [1.6,2.3,4.6,4.8]
ytick_labels = ['1.6','2.3','4.6','4.8']
for i, tick_label in enumerate(ytick_labels):
    if tick_label in ['1.6','2.3','4.6','4.8']:
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
plt.text(0.676,  3.9, 'II', fontsize=20)
plt.text(0.862,  3.902,  'III', fontsize=20)
# 纵排
plt.text(0.928, 1.3, 'I', fontsize=20)
plt.text(0.922, 1.85, 'II', fontsize=20)
plt.text(0.916, 2.5, 'III', fontsize=20)
#
plt.text(0.65, 3, 'schizophrenia', fontsize=20)

ax.fill_between(cutoff, 0, interval_thrs_1, facecolor='green', alpha=0.2, linewidth=0)  # 区域的下边界为0，上边界为interval_thrs_1
ax.fill_between(cutoff, interval_thrs_1, interval_thrs_2, facecolor='red', alpha=0.2, edgecolor="k", linewidth=0)
ax.fill_between(cutoff, interval_thrs_2, 5, edgecolor="k", alpha=0.2, linewidth=0.0)

plt.tight_layout()
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.png',  dpi=1200, bbox_inches='tight')
plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_egg_s_2.png',  dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProjet\plot_eeg\feature\phase_transition_egg_s_1.pdf',  dpi=1200, bbox_inches='tight')

plt.show()

# #=======================抽象====================
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.ticker as ticker
#
# plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示
#
# # cutoff = np.arange(0.05, 0.755, 0.018)
# cutoff = [0.1, 0.125  , 0.161 ,
#           0.201 , 0.241 ,0.282,
#           0.3,0.322 ,0.34,0.362 ,0.403 ,0.443 ,0.483 ,
#           0.523 ,0.563 ,0.604,0.644 ,0.684 ,0.725 ,0.765 ,0.805,0.9 ,1.0]
#
# # couple = np.concatenate((couple,[20]))
# print('cutoff:', cutoff)
# print(len(cutoff))
#
# # interval_thrs_1 = [6.00 ,4.50 ,3.50 ,2.60 ,2.50 ,2.40 ,2.30 ,2.20 ,2.10 ,2.00 ,1.90 ,
# #                   1.80 ,1.70 ,1.69 ,1.69 ,1.68 ,1.68,1.67  ]
# # interval_thrs_2 = [7.50 ,6.70 ,4.50 ,3.90 ,3.50 ,3.10 ,2.80 ,2.70 ,2.70 ,2.70 ,2.50 ,
# #                   2.40 ,2.30 ,2.29 ,2.28 ,2.27 ,2.26,2.26]
# interval_thrs_1 = [7.25,  7.1 ,6.8,
#                    6.00 ,4.50 ,3.50 ,
#                    3.2, 2.95,2.70 ,2.55 ,2.40 ,2.30 ,2.20 ,2.10 ,2.00 ,1.90 ,
#                   1.80 ,1.71 ,1.69 ,1.69 ,1.68 ,1.68,1.67  ]
# interval_thrs_2 = [7.25,  7.1 ,6.8,
#                    6.5,6.0 ,5 ,
#                    4.5, 4.1 ,3.80 , 3.50 ,3.10 ,2.90 ,2.75 ,2.70 ,2.60 ,2.50 ,
#                   2.40 ,2.30 ,2.29 ,2.28 ,2.27 ,2.26,2.26]
# print(len(interval_thrs_1))
# print(len(interval_thrs_2))
#
# # phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_0.png')
# # phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_1.png')
# # phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_2.png')
# # phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_3.png')
# # phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_4.png')
# # phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_5.png')
# phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_6.png')
# phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_7.png')
# phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_8.png')
# plt.figure(figsize=(8, 6))
#
# plt.ylabel('coupling strength', fontsize=26, labelpad=10)
# plt.xlabel('filtration radius', fontsize=26)
#
# # 设置绘图的坐标轴范围
# plt.axis([0.10, 1.0, 1,7.5])
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
# ax.axhline(2.3, color=custom_light_gray, linestyle='dashed')
# ax.axhline(6.8, color=custom_light_gray, linestyle='dashed')
# ax.axhline(7.25, color=custom_light_gray, linestyle='dashed')
# yticks = [1.7,2.3]
# ytick_labels = ['1.7','2.3']
# for i, tick_label in enumerate(ytick_labels):
#     if tick_label in ['1.7','2.3']:
#         plt.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=18)
#     else:
#         plt.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=16)
#
# plt.tight_layout()
# # plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.png',  dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_egg_s_1.png',  dpi=1200, bbox_inches='tight')
# # plt.savefig(r'D:\pycharm\pythonProjet\plot_eeg\feature\phase_transition_egg_s_1.pdf',  dpi=1200, bbox_inches='tight')
#
# plt.show()


# #=======================真实情况=======================
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.ticker as ticker
#
# plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示
#
# # cutoff = np.arange(0.05, 0.755, 0.018)
# cutoff = [0.1,0.161 ,0.201 , 0.241 ,0.282,0.3,0.322 ,0.34,       0.362 ,0.403 ,0.443 ,0.483 ,0.523 ,0.563 ,
#           0.604,0.644 ,0.684 ,0.725 ,0.765 ,0.805,0.9 ,1.0]
#
# # couple = np.concatenate((couple,[20]))
# print('cutoff:', cutoff)
# print(len(cutoff))
#
# # interval_thrs_1 = [6.00 ,4.50 ,3.50 ,2.60 ,2.50 ,2.40 ,2.30 ,2.20 ,2.10 ,2.00 ,1.90 ,
# #                   1.80 ,1.70 ,1.69 ,1.69 ,1.68 ,1.68,1.67  ]
# # interval_thrs_2 = [7.50 ,6.70 ,4.50 ,3.90 ,3.50 ,3.10 ,2.80 ,2.70 ,2.70 ,2.70 ,2.50 ,
# #                   2.40 ,2.30 ,2.29 ,2.28 ,2.27 ,2.26,2.26]
# interval_thrs_1 = [19,9,6.00 ,4.50 ,3.50 , 3.2, 2.95,2.70 ,2.55 ,2.40 ,2.30 ,2.20 ,2.10 ,2.00 ,1.90 ,
#                   1.80 ,1.72 ,1.69 ,1.69 ,1.68 ,1.68,1.67  ]
# interval_thrs_2 = [150,80,8.50 ,6.90 ,5 ,4.5, 4.1 ,3.80 ,   3.50 ,3.10 ,2.90 ,2.75 ,2.70 ,2.60 ,2.50 ,
#                   2.40 ,2.30 ,2.29 ,2.28 ,2.27 ,2.26,2.26]
# print(len(interval_thrs_1))
# print(len(interval_thrs_2))
#
# phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_0.png')
# phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_1.png')
# phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_2.png')
# # phase_1 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_3.png')
# # phase_2 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_4.png')
# # phase_3 = Image.open(r'D:\pycharm\pythonProject\plot_eeg\feature\figure\s_5.png')
# plt.figure(figsize=(8, 6))
#
# plt.ylabel('coupling strength', fontsize=26, labelpad=10)
# plt.xlabel('filtration radius', fontsize=26)
#
# # 设置绘图的坐标轴范围
# plt.axis([0.10, 1.0, 1,8])
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
# ax.fill_between(cutoff, interval_thrs_2, 8, edgecolor="k", alpha=0.2, linewidth=0.0)
#
# plt.tight_layout()
# # plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_eeg_h.png',  dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\feature\phase_transition_egg_s_1.png',  dpi=1200, bbox_inches='tight')
# # plt.savefig(r'D:\pycharm\pythonProjet\plot_eeg\feature\phase_transition_egg_s_1.pdf',  dpi=1200, bbox_inches='tight')
#
# plt.show()