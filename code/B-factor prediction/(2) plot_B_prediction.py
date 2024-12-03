# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
#
# plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示
# #
# pdbID = ['1FF4']
# couple = [20]
# cutoff = [0.87]
#
# # pdbID = ['2Y7L']
# # couple = [3]
# # cutoff = [1]
# for i in range(len(pdbID)):
# # for i in range(2):
#     atom_num = len(open(r'D:/pycharm/pythonProject/protein/364_xyz_Bfactor/364_xyzb_5.1/%s_ca.xyzb' % pdbID[i], 'r').readlines())
#     print('atom_num: ',atom_num)
#     B_factor = []
#     pdbfile = open('D:/pycharm/pythonProject/protein/364_xyz_Bfactor/364_xyzb_5.1/%s_ca.xyzb' % pdbID[i], 'r')
#     for line in pdbfile.readlines():
#         B_factor.append(float(line.split()[3]))
#
#     # 加载多元线性回归预测的B-factor
#     yFit = np.load('D:/pycharm/pythonProject/egg/results/B-factor/yFit_%s_%d_%.2f_3_1.npy'%(pdbID[i], couple[i], cutoff[i]))
#     print(yFit.shape)  #如 (1044,)含1044个元素的一维数组
#
#     ################### plot ###########
#     fig, ax = plt.subplots(figsize=(7, 3))
#     n = atom_num
#     yTrue = np.array(B_factor).reshape(-1, 1)
#     print(yTrue.shape)
#
#     x = [i for i in range(n)]
#     print(np.shape(x), np.shape(yFit), np.shape(yTrue))
#     ax.plot(x, yFit,'o-',label='TEPC',color='orange', linewidth=1.5, markersize=3.5)
#     ax.plot(x,yTrue,'d-',label='Experiment',color='#9281DD', linewidth=1.5, markersize=3.5)#'blue'
#
#     legend_font = FontProperties( size=18)  # 设置图例字体大小
#     ax.legend(loc=(0.3, 0.75), ncol=2,prop=legend_font, frameon=False)  # 设置frameon参数为False，去掉图例边框
#     # ax.legend(loc=(0.15, 0.75), ncol=2,prop=legend_font, frameon=False)  # 设置frameon参数为False，去掉图例边框
#
#     plt.xlabel('Residue Number', fontsize=24)
#     plt.ylabel('B-factor', fontsize=24)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     # plt.gca().spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的线
#     # plt.gca().spines['right'].set_visible(False)
#     plt.tight_layout()
#     plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/B-factor_pre_%s_%d_%.2f.png'%(pdbID[i],couple[i],cutoff[i]), dpi=1800, bbox_inches='tight')
#     plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/B-factor_pre_%s_%d_%.2f.svg'%(pdbID[i],couple[i],cutoff[i]), dpi=1800, bbox_inches='tight')
#     # plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/B-factor_pre_%s_%d_%.2f.pdf'%(pdbID[i],couple[i],cutoff[i]), dpi=800, bbox_inches='tight')
#     plt.show()

# #======================================step1数据处理
# #   http://bahar.labs.stonybrook.edu/ignm/iGNM_bfactor.php?gnm_id=2Y7L&viewer=jsmol 下载GNM预测B因子
# import numpy as np
# # 读取PDB文件并进行数据处理
# pdb_file_path = r'D:\pycharm\pythonProject\plot\data\2Y7L_bfactor_GNM.pdb'
#
# # 存储原子坐标的列表
# atom = []
# # 存储原子坐标和温度因子的列表
# atom_b = []
#
# # 遍历PDB文件的每一行
# with open(pdb_file_path, 'r') as pdb_file:
#     for line in pdb_file:
#         # 仅保留以"ATOM"开头的行
#         if line.startswith("ATOM"):
#             # 检查原子类型是否是CA
#             if line[13:15] == "CA":
#                 # 提取xyz坐标信息
#                 x = float(line[30:38])
#                 y = float(line[38:46])
#                 z = float(line[46:54])
#                 atom.append([x, y, z])
#
#                 # 提取xyz坐标和tempFactor信息
#                 temp_factor = float(line[61:66])
#                 atom_b.append([x, y, z, temp_factor])
#
# # # 将数据保存为numpy数组和文本文件,只含xyz坐标
# # atom_coordinates_np = np.array(atom)
# # np.save("C:/Users/administered/Desktop/MND笔记/protein/data/2Y7L_ca_GNM.npy", atom_coordinates_np)
# # # np.savetxt("C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_ca.txt", atom_coordinates_np)
# # 将数据保存为xyzb文件
# with open("C:/Users/administered/Desktop/MND笔记/protein/data/2Y7L_ca_GNM.xyzb", "w") as xyzb_file:
#     for data in atom_b:
#         x, y, z, temp_factor = data
#         xyzb_file.write(f"{x} {y} {z} {temp_factor}\n")
#
#
# #========================================#step2:检查数据
# 1FF4_ca_GNM.xyzb的前65行是否
# # 和官网提取的C:\Users\administered\Desktop\MND笔记\protein\data\处理1Q9B后的364个PDB数据1FF4_ca.xyzb前三列都是相同的
# #逐行比较两个文件的前三列，并输出所有不同的行号及其内容
# # 检查两个文件是否存在。
# # 读取两个文件的内容。
# # 比较两个文件的行数。如果行数不同，输出提示信息并以较少行数的文件为基础进行比较。
# # 逐行比较两个文件的前三列内容。
# # 如果找到不同的行，记录行号及其内容。
# # 最后输出所有不同的行号及其内容。如果所有行都相同，输出确认信息。
# import os
#
# # 定义文件路径
# file1_path = r'D:\pycharm\pythonProject\plot\data\2Y7L_ca_GNM_1.xyzb'
# file2_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\处理1Q9B后的364个PDB数据\2Y7L_ca.xyzb'
#
# # 检查文件是否存在
# if not os.path.exists(file1_path):
#     print(f"文件不存在: {file1_path}")
# elif not os.path.exists(file2_path):
#     print(f"文件不存在: {file2_path}")
# else:
#     # 读取文件内容
#     with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
#         lines1 = file1.readlines()
#         lines2 = file2.readlines()
#
#     # 获取较少行数的文件
#     min_lines = min(len(lines1), len(lines2))
#
#     if len(lines1) != len(lines2):
#         print(f"两个文件的行数不同：文件1有 {len(lines1)} 行，文件2有 {len(lines2)} 行。将以较少行数的文件进行比较。")
#
#     # 比较文件内容
#     different_lines = []
#     for i in range(min_lines):
#         # 获取每一行前三列
#         cols1 = lines1[i].split()[:3]
#         cols2 = lines2[i].split()[:3]
#
#         # 将前三列数值标准化为同样的小数位数进行比较
#         cols1_normalized = [f"{float(col):.3f}" for col in cols1]
#         cols2_normalized = [f"{float(col):.3f}" for col in cols2]
#
#         # 检查前三列是否相同
#         if cols1_normalized != cols2_normalized:
#             different_lines.append((i + 1, cols1, cols2))
#
#     # 输出不同的行号及其内容
#     if different_lines:
#         print(f"有 {len(different_lines)} 行不同:")
#         for line_num, cols1, cols2 in different_lines:
#             print(f"第 {line_num} 行不同:")
#             print(f"文件1: {cols1}")
#             print(f"文件2: {cols2}")
#     else:
#         print("两个文件的前三列完全相同")


# ===================   step3: 蛋白质 1cll gnm + Tepc + experiment 三折线可视化    =====================
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示
# 读取数据
#1cll
# gnm_data = pd.read_csv(r'D:\pycharm\pythonProjec t\plot\data\gnm7_1cll.csv', usecols=[1])
# experiment_data = pd.read_csv(r'D:\pycharm\pythonProject\plot\data\gnm7_1cll.csv', usecols=[2])
# theory_data = np.load(r'D:\pycharm\pythonProject\plot\data\1CLL_yFit_1711879166.npy')
# 定义文件路径
# gnm_file_path = r'D:\pycharm\pythonProject\plot\data\1FF4_ca_GNM_1.xyzb'
# experiment_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\处理1Q9B后的364个PDB数据\1FF4_ca.xyzb'
# gnm_file_path = r'D:\pycharm\pythonProject\plot\data\2Y7L_ca_GNM_1.xyzb'
# experiment_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\处理1Q9B后的364个PDB数据\2Y7L_ca.xyzb'
gnm_file_path = r'D:\pycharm\pythonProject\plot\data\2WUJ_ca_GNM_1.xyzb'
experiment_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\处理1Q9B后的364个PDB数据\2WUJ_ca.xyzb'

# 提取gnm文件的第四列数据
gnm_data = []
with open(gnm_file_path, 'r') as gnm_file:
    for line in gnm_file:
        cols = line.split()
        if len(cols) > 3:
            temp_factor = float(cols[3])
            gnm_data.append(temp_factor)

# 提取experiment文件的第四列数据
experiment_data = []
with open(experiment_file_path, 'r') as experiment_file:
    for line in experiment_file:
        cols = line.split()
        if len(cols) > 3:
            temp_factor = float(cols[3])
            experiment_data.append(temp_factor)

# theory_data = np.load(r'D:\pycharm\pythonProject\plot\data\yFit_1FF4_20_0.87_3_1.npy')
theory_data = np.load(r'D:\pycharm\pythonProject\plot\data\yFit_2WUJ_17_0.92_3_1.npy')
# theory_data = np.load(r'D:\pycharm\pythonProject\plot\data\yFit_2Y7L_3_1.00_3_1.npy')

# 绘制折线图
plt.figure(figsize=(7, 3))

# GNM 方法预测的 B 因子折线图
plt.plot(gnm_data, 'o-',label='GNM', color='#9281DD', markersize=2.5)

# 实验测得的 B 因子折线图
plt.plot(experiment_data,'s-',label='Experiment', color='green', markersize=2.5)

# Theory 预测的 B 因子折线图
plt.plot(theory_data, '*-',label='TEPC', color='orange', markersize=3)

# 设置图例和标签
# plt.legend(ncol=2, fontsize=16, loc=(0.22, 0.7),frameon=False)
plt.legend(ncol=2, fontsize=16, loc=(0.12, 0.7),frameon=False)
# plt.legend(ncol=2, fontsize=16, loc=(0.2, 0.7),frameon=False)
plt.xlabel('Residue Index',fontsize=24)
plt.ylabel('B-factor',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 设置y轴刻度在内部显示，并增加粗细
plt.gca().tick_params(axis='x', width=1, length=5)#direction='in',
plt.gca().tick_params(axis='y', width=1, length=5)#direction='in',
plt.grid(False)
# plt.gca().spines['top'].set_visible(False)  # 隐藏坐标轴上方和右方的线
# plt.gca().spines['right'].set_visible(False)

# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/B-factor_pre_1CLL.png', dpi=1200, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/B-factor_pre_1CLL.svg', dpi=1200, bbox_inches='tight')
# plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/B-factor_pre_1CLL.pdf', dpi=800, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot\data\picture\B-factor_pre_1FF4.svg', dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot\data\picture\B-factor_pre_1FF4.png', dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot\data\picture\B-factor_pre_2Y7L.svg', dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot\data\picture\B-factor_pre_2Y7L.png', dpi=1200, bbox_inches='tight')
plt.savefig(r'D:\pycharm\pythonProject\plot\data\picture\B-factor_pre_2WUJ.svg', dpi=1200, bbox_inches='tight')
plt.savefig(r'D:\pycharm\pythonProject\plot\data\picture\B-factor_pre_2WUJ.png', dpi=1200, bbox_inches='tight')

# 显示图形
plt.show()

