'''
# 批量读取'D:\pycharm\pythonProject\plot_eeg\data\txt/'路径下以'h'开头txt文件,对每个文件计算关联矩阵，并将所有小于0的相关系数置为0，保存为_corr_matrix.txt文件
# 遍历以'h'开头且以'_corr_matrix.txt'结尾的文件列表，对每个文件读取关联矩阵，并将所有关联矩阵相加。然后计算平均关联矩阵并保存到文件_average_corr_matrix.txt中

# import os
# import numpy as np
# import pandas as pd
#
# # 定义函数计算关联矩阵并处理负相关系数
# def compute_and_process_corr_matrix(txt_file):
#     # 读取txt文件中的数据
#     data = np.loadtxt(txt_file)
#     # 计算相关系数矩阵
#     CorrMat = np.corrcoef(data, rowvar=1)
#     # 将所有小于0的相关系数置为0
#     CorrMat[CorrMat < 0] = 0
#     # 对角元素置为0
#     np.fill_diagonal(CorrMat, 0)
#     return CorrMat
#
# # 指定原始数据目录路径
# input_directory_path = r'D:\pycharm\pythonProject\plot_eeg\data\txt'
# # 指定保存关联矩阵的目录路径
# output_directory_path = r'D:\pycharm\pythonProject\plot_eeg\data\h_txt'
#
# # 列出原始数据目录中所有以'h'开头的txt文件
# file_list = [file for file in os.listdir(input_directory_path) if file.startswith('h') and file.endswith('.txt')]
#
# # 遍历每个文件并处理关联矩阵
# for txt_file in file_list:
#     input_txt_path = os.path.join(input_directory_path, txt_file)
#     # 计算关联矩阵并处理负相关系数
#     corr_matrix = compute_and_process_corr_matrix(input_txt_path)
#     # 构建保存关联矩阵的文件路径
#     output_txt_path = os.path.join(output_directory_path, txt_file.replace('.txt', '_corr_matrix.txt'))
#     # 保存关联矩阵为文本文件
#     np.savetxt(output_txt_path, corr_matrix, fmt='%f', delimiter=',')
#     print("保存关联矩阵完成:", output_txt_path)
#
# #====================计算所有以'h'开头且以'_corr_matrix.txt'的平均==============
# # 获取以'h'开头且以'_corr_matrix.txt'结尾的文件列表
# corr_matrix_files = [file for file in os.listdir(output_directory_path) if file.startswith('h') and file.endswith('_corr_matrix.txt')]
#
# # 存储所有关联矩阵的总和
# total_corr_matrix = None
#
# # 遍历每个文件并处理关联矩阵
# for corr_matrix_file in corr_matrix_files:
#     input_corr_matrix_path = os.path.join(output_directory_path, corr_matrix_file)
#     # 读取关联矩阵文件
#     corr_matrix = np.loadtxt(input_corr_matrix_path, delimiter=',')
#     # 处理负相关系数
#     corr_matrix[corr_matrix < 0] = 0
#     # 将关联矩阵添加到总和中
#     if total_corr_matrix is None:
#         total_corr_matrix = corr_matrix
#     else:
#         total_corr_matrix += corr_matrix
#
# # 计算平均关联矩阵
# average_corr_matrix = total_corr_matrix / len(corr_matrix_files)
#
# # 保存平均关联矩阵到文件
# average_corr_matrix_path = os.path.join(output_directory_path, 'h_average_corr_matrix.txt')
# np.savetxt(average_corr_matrix_path, average_corr_matrix, fmt='%f', delimiter=',')
# print("保存平均关联矩阵完成:", average_corr_matrix_path)
#
# # 指定保存平均关联矩阵的文件路径
# average_corr_matrix_path = os.path.join(output_directory_path, 'h_average_corr_matrix.npy')
# # 保存平均关联矩阵到文件
# np.save(average_corr_matrix_path, average_corr_matrix)
# print("保存平均关联矩阵完成:", average_corr_matrix_path)


import os
import numpy as np

# 定义函数计算关联矩阵并处理负相关系数
def compute_and_process_corr_matrix(txt_file):
    # 读取txt文件中的数据
    data = np.loadtxt(txt_file)
    # 计算相关系数矩阵
    CorrMat = np.corrcoef(data, rowvar=1)
    # 将所有小于0的相关系数置为0
    CorrMat[CorrMat < 0] = 0
    # 对角元素置为0
    np.fill_diagonal(CorrMat, 0)
    return CorrMat

# 指定原始数据目录路径
input_directory_path = r'D:\pycharm\pythonProject\plot_eeg\data\txt'
# 指定保存关联矩阵的目录路径
output_directory_path = r'D:\pycharm\pythonProject\plot_eeg\data\s_txt'

# 列出原始数据目录中所有以's'开头的txt文件
file_list = [file for file in os.listdir(input_directory_path) if file.startswith('s') and file.endswith('.txt')]

# 遍历每个文件并处理关联矩阵
for txt_file in file_list:
    input_txt_path = os.path.join(input_directory_path, txt_file)
    # 计算关联矩阵并处理负相关系数
    corr_matrix = compute_and_process_corr_matrix(input_txt_path)
    # 构建保存关联矩阵的文件路径
    output_txt_path = os.path.join(output_directory_path, txt_file.replace('.txt', '_corr_matrix.txt'))
    # 保存关联矩阵为文本文件
    np.savetxt(output_txt_path, corr_matrix, fmt='%f', delimiter=',')
    print("保存关联矩阵完成:", output_txt_path)

# 获取以's'开头且以'_corr_matrix.txt'结尾的文件列表
corr_matrix_files = [file for file in os.listdir(output_directory_path) if file.startswith('s') and file.endswith('_corr_matrix.txt')]

# 存储所有关联矩阵的总和
total_corr_matrix = None

# 遍历每个文件并处理关联矩阵
for corr_matrix_file in corr_matrix_files:
    input_corr_matrix_path = os.path.join(output_directory_path, corr_matrix_file)
    # 读取关联矩阵文件
    corr_matrix = np.loadtxt(input_corr_matrix_path, delimiter=',')
    # 处理负相关系数
    corr_matrix[corr_matrix < 0] = 0
    # 将关联矩阵添加到总和中
    if total_corr_matrix is None:
        total_corr_matrix = corr_matrix
    else:
        total_corr_matrix += corr_matrix

# 计算平均关联矩阵
average_corr_matrix = total_corr_matrix / len(corr_matrix_files)

# 指定保存平均关联矩阵的文件路径
average_corr_matrix_path = os.path.join(output_directory_path, 's_average_corr_matrix.txt')
# 保存平均关联矩阵到文件
np.savetxt(average_corr_matrix_path, average_corr_matrix, fmt='%f', delimiter=',')
print("保存平均关联矩阵完成:", average_corr_matrix_path)

# 指定保存平均关联矩阵的文件路径
average_corr_matrix_path = os.path.join(output_directory_path, 's_average_corr_matrix.npy')
# 保存平均关联矩阵到文件
np.save(average_corr_matrix_path, average_corr_matrix)
print("保存平均关联矩阵完成:", average_corr_matrix_path)

'''

#======================plot corr_matrix  ===================
import os
import numpy as np
import matplotlib.pyplot as plt

# Load average correlation matrices
# s_average_corr_matrix_path = r'D:\pycharm\pythonProject\plot_eeg\data\s_average_corr_matrix.npy'
# h_average_corr_matrix_path = r'D:\pycharm\pythonProject\plot_eeg\data\h_average_corr_matrix.npy'
s_average_corr_matrix_path = r'D:\pycharm\pythonProject\plot_eeg\data\s_txt\s13_corr_matrix.txt'
h_average_corr_matrix_path = r'D:\pycharm\pythonProject\plot_eeg\data\h_txt\h13_corr_matrix.txt'

# s_average_corr_matrix = np.load(s_average_corr_matrix_path)
# h_average_corr_matrix = np.load(h_average_corr_matrix_path)
s_average_corr_matrix = np.loadtxt(s_average_corr_matrix_path, delimiter=',')
h_average_corr_matrix = np.loadtxt(h_average_corr_matrix_path, delimiter=',')

# Visualize s_average_corr_matrix
plt.figure(figsize=(10, 8))
ax_s = plt.matshow(s_average_corr_matrix, cmap='viridis', aspect='auto')#cmap='viridis'
y_s = plt.colorbar(ax_s.colorbar)
y_s.ax.tick_params(labelsize=20)
plt.xlabel('Electrode Number', fontsize=26, labelpad=8)
plt.ylabel('Electrode Number', fontsize=26, labelpad=1)
plt.tick_params(labelsize=20)
plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\figure\s13_corr_matrix.png', dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\figure\s_corr_matrix.svg', dpi=1200, bbox_inches='tight')
plt.close()

# Visualize h_average_corr_matrix
plt.figure(figsize=(10, 8))
ax_h = plt.matshow(h_average_corr_matrix, cmap='viridis', aspect='auto')
y_h = plt.colorbar(ax_h.colorbar)
y_h.ax.tick_params(labelsize=20)
plt.xlabel('Electrode Number', fontsize=26, labelpad=8)
plt.ylabel('Electrode Number', fontsize=26, labelpad=1)
plt.tick_params(labelsize=20)
plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\figure\h13_corr_matrix.png', dpi=1200, bbox_inches='tight')
# plt.savefig(r'D:\pycharm\pythonProject\plot_eeg\figure\h_corr_matrix.svg', dpi=1200, bbox_inches='tight')
plt.close()