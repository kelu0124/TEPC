# =================  柱状图 _ 竖版============================
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

# 数据
accession_ids = ['GSE45719','GSE67835',  'GSE75748 time','GSE84133 m1', 'GSE84133 h4','GSE75748 cell','GSE84133 m2',
                 'GSE82187', 'GSE89232','GSE59114',  'GSE94820' ,  'GSE84133 h2',  'GSE84133 h1'     ]
TEMLPCN_results = [0.950,  0.836,0.983,0.896,0.954, 0.949, 0.872, 0.806,0.916,0.947,  0.970,  0.874  ,0.847 ]
PCA_results = [0.630, 0.675,0.830, 0.773, 0.883,0.898, 0.823, 0.771, 0.885, 0.918, 0.943 , 0.867, 0.845 ]

x = np.arange(len(accession_ids))  # 柱的索引
width = 0.42 # 柱的宽度

fig, ax = plt.subplots(figsize=(8, 8))

# # 绘制TEMLPCN柱状图
# bars1 = ax.bar(x - width/2, TEMLPCN_results, width, label='TEMLPCN',color='#E1B6B5')
# # 绘制PCA柱状图
# bars2 = ax.bar(x + width/2, PCA_results, width, label='PCA',  color='#F4DEBB')

# 绘制TEMLPCN柱状图
bars1 = ax.bar(x - width/2, TEMLPCN_results, width, label='TEPC',color='#9281DD')
# 绘制PCA柱状图
bars2 = ax.bar(x + width/2, PCA_results, width, label='PCA',  color='lightblue')

# 添加标签、标题和图例
# ax.set_xlabel('Accession ID')
ax.set_ylabel('Balanced Accuracy', fontsize=26)
ax.set_xticks(x)
ax.set_xticklabels(accession_ids, rotation=45, ha='right')
ax.legend()

# 添加数值标签
# def add_value_labels(ax, bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom',
#                     fontsize=18, rotation=90)  # 设置字体大小为8并旋转90度
#
# add_value_labels(ax, bars1)
# add_value_labels(ax, bars2)

# 设置 y 轴范围从0.6开始
ax.set_ylim(0.6, 1.0)

# 设置 x 轴范围
plt.xlim(-0.5, len(accession_ids) - 0.5)

# 隐藏坐标轴上方和右方的边界线
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 设置刻度字号
plt.yticks(fontsize=20)

ax.tick_params(axis='x', which='both',  width=1, length=5,labelsize=18)   # 设置x轴刻度在内部显示，并增加粗细
ax.tick_params(axis='y', which='both', width=1,length=5)  # 设置y轴粗细direction='in',
# 设置y轴刻度在内部显示，并增加粗细
# plt.gca().tick_params(axis='y',direction='in', width=1, length=5)

# 调整图例位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=2, fontsize=18 ,frameon=False)

# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig2_2.png' , dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig2_2.pdf' , dpi=800, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig2_2.svg' , dpi=800, bbox_inches='tight')
plt.tight_layout()
plt.show()




# # =================  折线图============================
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams["font.family"] = "Times New Roman"
#
# # 数据
# accession_ids = ['GSE45719', 'GSE59114', 'GSE67835', 'GSE75748 cell', 'GSE75748 time',
#                  'GSE82187', 'GSE84133 h1', 'GSE84133 h2', 'GSE84133 h4', 'GSE84133 m1', 'GSE84133 m2',
#                  'GSE89232', 'GSE94820']
# TEMLPCN_results = [0.950, 0.947, 0.836, 0.949, 0.983, 0.806, 0.847, 0.874, 0.954, 0.896, 0.872, 0.916, 0.970]
# PCA_results = [0.630, 0.918, 0.675, 0.898, 0.830, 0.771, 0.845, 0.867, 0.883, 0.773, 0.823, 0.885, 0.943]
#
# # 计算差值并排序
# difference = [p - t for p, t in zip(PCA_results, TEMLPCN_results)]
# sorted_indices = np.argsort(difference)
#
# # 根据差值排序后的索引对accession_ids、TEMLPCN_results和PCA_results进行重排
# sorted_accession_ids = [accession_ids[i] for i in sorted_indices]
# sorted_TEMLPCN_results = [TEMLPCN_results[i] for i in sorted_indices]
# sorted_PCA_results = [PCA_results[i] for i in sorted_indices]
#
# # 绘制折线图
# plt.figure(figsize=(8, 4))
# plt.plot(sorted_accession_ids, sorted_TEMLPCN_results, marker='o', color='#9281DD', label='TEMLPCN')
# plt.plot(sorted_accession_ids, sorted_PCA_results, marker='*', color= '#c56f53', label='PCA')
#
# # 在折线上添加数字标签
# for x, y in zip(sorted_accession_ids, sorted_TEMLPCN_results):
#     plt.text(x, y, '{:.3f}'.format(y), ha='center', va='bottom', fontsize=10)
#
# for x, y in zip(sorted_accession_ids, sorted_PCA_results):
#     plt.text(x, y, '{:.3f}'.format(y), ha='center', va='top', fontsize=10)
#
# plt.ylabel('Balanced Accuracy', fontsize=20)
# plt.xticks(rotation=45, ha='right', fontsize=13)
# plt.yticks(fontsize=14)
# plt.legend(loc='upper center', bbox_to_anchor=(0.78, 0.25), shadow=True, ncol=2, fontsize=14, frameon=False)  # 去掉图例的边框
#
# plt.grid(False)
#
# # 隐藏坐标轴上方和右方的边界线
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
#
# # 设置y轴刻度在内部显示，并增加粗细
# plt.gca().tick_params(axis='x', width=1, length=5)
# plt.gca().tick_params(axis='y', direction='in', width=1, length=5)
#
# plt.tight_layout()
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/fig2_1.png' , dpi=800, bbox_inches='tight')
# plt.show()

