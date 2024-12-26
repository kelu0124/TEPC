# import numpy as np
# # 读取PDB文件并进行数据处理
# pdb_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\pbd官网数据_364\1CLL.pdb'
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
# # 将数据保存为numpy数组和文本文件,只含xyz坐标
# atom_coordinates_np = np.array(atom)
# np.save("C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_ca.npy", atom_coordinates_np)
# # np.savetxt("C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_ca.txt", atom_coordinates_np)
#
# # 将数据保存为xyzb文件
# with open("C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_ca.xyzb", "w") as xyzb_file:
#     for data in atom_b:
#         x, y, z, temp_factor = data
#         xyzb_file.write(f"{x} {y} {z} {temp_factor}\n")

'''
#检查提取数据是否数据处理是否正确

import csv

# 读取csv文件中的第三列数据
csv_file_path = r'D:\pycharm\pythonProject\plot\gnm7_1cll.csv'
csv_values = []
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # 跳过文件头部
    for row in csv_reader:
        csv_values.append(float(row[2]))

# 读取xyzb文件中的第四列数据
xyzb_file_path = r'C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_ca.xyzb'
xyzb_values = []
with open(xyzb_file_path, 'r') as xyzb_file:
    for line in xyzb_file:
        temp_factor = float(line.split()[3])
        xyzb_values.append(temp_factor)

# 比较两个列表的长度是否相等
if len(csv_values) == len(xyzb_values):
    # 逐个比较两个列表的对应元素是否相等
    all_equal = all(a == b for a, b in zip(csv_values, xyzb_values))
    if all_equal:
        print("每个数值相等")
    else:
        print("存在不相等的数值")
else:
    print("列表长度不相等")
'''



# '''
# #利用预测的B因子替换以"ATOM"开头且原子类型是"CA"的行的温度因子信息,保存需要替换行的信息为pdb格式：
# # 首先读取原始PDB文件中的每一行数据，并将以"ATOM"开头的行存储在atom_lines列表中。
# # 然后，我们加载预测的B因子数据，并确保其与PDB文件中的行数匹配。
# # 接下来，我们使用预测的B因子值替换原始PDB文件中相应行的温度因子信息，并将修改后的行写入新的PDB文件中。
#
# import numpy as np
#
# # 读取原始的PDB文件路径和预测的B因子数据路径
# pdb_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\pbd官网数据_364\1CLL.pdb'
# predicted_b_factor_file_path = 'C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_yFit_1711879166.npy'
#
# # 加载原始PDB文件的数据
# atom_lines = []
# with open(pdb_file_path, 'r') as pdb_file:
#     for line in pdb_file:
#         if line.startswith("ATOM") and line[13:15] == "CA":  # 仅保留原子类型为"CA"的行
#             atom_lines.append(line)
#
# # 加载预测的B因子数据
# predicted_b_factors = np.load(predicted_b_factor_file_path)
#
# # 替换PDB文件中的温度因子信息
# if len(atom_lines) == len(predicted_b_factors):
#     new_atom_lines = []
#     for i, line in enumerate(atom_lines):
#         new_temp_factor = f"{predicted_b_factors[i]:6.2f}"  # 格式化预测的B因子值
#         new_line = line[:60] + new_temp_factor + line[66:]  # 替换温度因子信息
#         new_atom_lines.append(new_line)
#
#     # 将替换后的数据保存为新的PDB文件
#     with open(r'C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_new.pdb', 'w') as new_pdb_file:
#         new_pdb_file.write(''.join(new_atom_lines))
# else:
#     print("Error: The number of lines in the PDB file does not match the number of predicted B factors.")
#
'''
#用预测B因子替换官方PDB文件

import numpy as np
import pandas as pd

# 读取原始的PDB文件路径和预测的B因子数据路径
pdb_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\pbd官网数据_364\1CLL.pdb'
predicted_b_factor_file_path = 'C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_yFit_1711879166.npy'
gnm_predicted_b_factor_file_path = 'C:/Users/administered/Desktop/MND笔记/protein/data/gnm7_1cll.csv'

# 加载原始PDB文件的数据
atom_lines = []
with open(pdb_file_path, 'r') as pdb_file:
    for line in pdb_file:
        if line.startswith("ATOM") and line[13:15] == "CA":  # 仅保留原子类型为"CA"的行
            atom_lines.append(line)

# 加载预测的B因子数据
predicted_b_factors = np.load(predicted_b_factor_file_path)

# 加载gnm预测的B因子数据
gnm_predicted_b_factors = pd.read_csv(gnm_predicted_b_factor_file_path, header=None)[1].values[1:]
gnm_predicted_b_factors = gnm_predicted_b_factors.astype(float)  # 将数据转换为浮点数

# 替换PDB文件中的温度因子信息
if len(atom_lines) == len(predicted_b_factors) and len(atom_lines) == len(gnm_predicted_b_factors):
    new_atom_lines = []
    for i, line in enumerate(atom_lines):
        new_temp_factor = f"{gnm_predicted_b_factors[i]:6.2f}"  # 格式化gnm预测的B因子值
        new_line = line[:60] + new_temp_factor + line[66:]  # 替换温度因子信息
        new_atom_lines.append(new_line)

    # 将替换后的数据保存为新的PDB文件
    with open('C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_GNM.pdb', 'w') as new_pdb_file:
        new_pdb_file.write(''.join(new_atom_lines))
else:
    print("Error: The number of lines in the PDB file does not match the number of predicted B factors.")

'''

#只替换1CLL.pdb中以"ATOM"开头且原子类型为"CA"的行的 B 因子，同时保留其他非替换行的信息，并将结果保存为1CLL_GNM1.pdb
# 逐行读取1CLL.pdb文件中的数据，对于符合条件的行（以"ATOM"开头且原子类型为"CA"），将预测的 B 因子值替换进去，然后将所有行重新组合并写入到1CLL_GNM1.pdb文件中
import numpy as np
import pandas as pd

# 读取原始的PDB文件路径和预测的B因子数据路径
pdb_file_path = r'C:\Users\administered\Desktop\MND笔记\protein\data\pbd官网数据_364\1CLL.pdb'
predicted_b_factor_file_path = 'C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_yFit_1711879166.npy'
gnm_predicted_b_factor_file_path = 'C:/Users/administered/Desktop/MND笔记/protein/data/gnm7_1cll.csv'

# 加载原始PDB文件的数据
with open(pdb_file_path, 'r') as pdb_file:
    pdb_lines = pdb_file.readlines()

# 加载预测的B因子数据
predicted_b_factors = np.load(predicted_b_factor_file_path)

# 加载gnm预测的B因子数据
gnm_predicted_b_factors = pd.read_csv(gnm_predicted_b_factor_file_path, header=None)[1].values[1:]
gnm_predicted_b_factors = gnm_predicted_b_factors.astype(float)  # 将数据转换为浮点数

# 替换PDB文件中的温度因子信息
new_pdb_lines = []
predicted_b_factors_index = 0
for line in pdb_lines:
    if line.startswith("ATOM") and line[13:15] == "CA":  # 仅替换原子类型为"CA"的行
        new_temp_factor = f"{gnm_predicted_b_factors[predicted_b_factors_index]:6.2f}"  # 格式化gnm预测的B因子值
        new_line = line[:60] + new_temp_factor + line[66:]  # 替换温度因子信息
        predicted_b_factors_index += 1
        new_pdb_lines.append(new_line)
    else:
        new_pdb_lines.append(line)

# 将替换后的数据保存为新的PDB文件
with open('C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_GNM1.pdb', 'w') as new_pdb_file:
    new_pdb_file.write(''.join(new_pdb_lines))

# 替换PDB文件中的温度因子信息
# new_pdb_lines = []
# predicted_b_factors_index = 0
# for line in pdb_lines:
#     if line.startswith("ATOM") and line[13:15] == "CA":  # 仅替换原子类型为"CA"的行
#         new_temp_factor = f"{predicted_b_factors[predicted_b_factors_index]:6.2f}"  # 格式化预测的B因子值
#         new_line = line[:60] + new_temp_factor + line[66:]  # 替换温度因子信息
#         predicted_b_factors_index += 1
#         new_pdb_lines.append(new_line)
#     else:
#         new_pdb_lines.append(line)
#
# # 将替换后的数据保存为新的PDB文件
# with open('C:/Users/administered/Desktop/MND笔记/protein/data/1CLL_new1.pdb', 'w') as new_pdb_file:
#     new_pdb_file.write(''.join(new_pdb_lines))