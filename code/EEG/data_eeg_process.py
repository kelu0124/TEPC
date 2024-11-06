import os
import pyedflib
from scipy.signal import butter, filtfilt

#======================利用提取alpha波段数据并保存到_alpha.txt==================
# Butterworth滤波器参数
'''
参数：
lowcut：带通滤波器的下限频率。
highcut：带通滤波器的上限频率。
fs：信号的采样频率。
order：Butterworth滤波器的阶数，默认为2。
返回值：b, a：Butterworth滤波器的分子和分母系数。
'''
# def butter_bandpass(lowcut, highcut, fs, order=2):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
#
# # 应用滤波器
# def apply_bandpass_filter(data, lowcut, highcut, fs, order=2):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     filtered_data = filtfilt(b, a, data)
#     return filtered_data
#
# # 定义函数将EDF文件转换为文本文件
# def convert_edf_to_txt_alpha(edf_file, txt_file):
#     # 打开EDF文件
#     f = pyedflib.EdfReader(edf_file)
#     # 获取信号数量
#     num_signals = f.signals_in_file
#     print("文件:", edf_file)
#     print("信号数量:", num_signals)
#     # 初始化一个字典来存储信号数据
#     signals = {f.getLabel(i): [] for i in range(num_signals)}
#     # 读取每个信号的数据
#     for i in range(num_signals):
#         signals[f.getLabel(i)] = f.readSignal(i)
#     # 关闭EDF文件
#     f.close()
#
#     # 将提取的数据写入新的文本文件
#     with open(txt_file, 'w') as txt_f:
#         # 只保留alpha波段数据
#         alpha_lowcut = 8  # alpha波段下限
#         alpha_highcut = 12.5  # alpha波段上限
#         fs = 250  # 采样频率
#         # 对每个信号进行处理
#         for signal, data in signals.items():
#             # 对信号进行分段处理
#             for i in range(0, len(data), fs * 30):
#                 segment = data[i:i + fs * 30]  # 每个30秒的段
#                 # 移除伪迹（这里假设没有伪迹数据）
#                 # 应用滤波器提取alpha波段数据
#                 filtered_data = apply_bandpass_filter(segment, alpha_lowcut, alpha_highcut, fs)
#                 # 将alpha波段数据写入文本文件
#                 for value in filtered_data:
#                     txt_f.write(str(value) + ' ')
#                 txt_f.write('\n')
#
#     # 计算行数和列数
#     num_rows = num_signals
#     num_cols = len(filtered_data)
#     print("行数:", num_rows)
#     print("列数:", num_cols)
#
# # 包含EDF文件的目录路径
# directory_path = r'D:\pycharm\pythonProject\plot_eeg\data/'
# # 列出目录中的所有文件
# file_list = os.listdir(directory_path)
# # 过滤出只包含EDF文件
# edf_files = [file for file in file_list if file.endswith('.edf')]
#
# # 处理每个EDF文件
# for edf_file in edf_files:
#     input_edf_file = os.path.join(directory_path, edf_file)
#     # 生成输出文本文件的文件名，将'.edf'替换为'_alpha.txt'
#     output_txt_file = os.path.join(directory_path, edf_file.replace('.edf', '_alpha.txt'))
#     # 将EDF文件转换为文本文件
#     convert_edf_to_txt_alpha(input_edf_file, output_txt_file)
# print("转换完成。")



# #=====================将原始的edf文件处理成txt文件，并print行列数=============
#定义函数将EDF文件转换为文本文件
# def convert_edf_to_txt(edf_file, txt_file):
#     # 打开EDF文件
#     f = pyedflib.EdfReader(edf_file)
#     # 获取信号数量
#     num_signals = f.signals_in_file
#     print("文件:", edf_file)
#     print("信号数量:", num_signals)
#     # 初始化一个字典来存储信号数据
#     signals = {f.getLabel(i): [] for i in range(num_signals)}
#     # 读取每个信号的数据
#     for i in range(num_signals):
#         signals[f.getLabel(i)] = f.readSignal(i)
#     # 关闭EDF文件
#     f.close()
#     # 将提取的数据写入新的文本文件
#     with open(txt_file, 'w') as txt_f:
#         for signal, data in signals.items():
#             for value in data:
#                 txt_f.write(str(value) + ' ')
#             txt_f.write('\n')
#     # 计算行数和列数
#     num_rows = num_signals
#     num_cols = len(data)
#     print("行数:", num_rows)
#     print("列数:", num_cols)
#
# # 包含EDF文件的目录路径
# directory_path = r'D:\pycharm\pythonProject\plot_eeg\data/'
# # 列出目录中的所有文件
# file_list = os.listdir(directory_path)
# # 过滤出只包含EDF文件
# edf_files = [file for file in file_list if file.endswith('.edf')]
#
# # 处理每个EDF文件
# for edf_file in edf_files:
#     input_edf_file = os.path.join(directory_path, edf_file)
#     # 生成输出文本文件的文件名，将'.edf'替换为'.txt'
#     output_txt_file = os.path.join(directory_path, edf_file.replace('.edf', '.txt'))
#     # 将EDF文件转换为文本文件
#     convert_edf_to_txt(input_edf_file, output_txt_file)
# print("转换完成。")


'''
#========================== 定义函数来读取EDF文件并打印其内容======================
def read_edf_file(file_path):
    # 打开EDF文件
    f = pyedflib.EdfReader(file_path)

    # 打印文件头信息
    print("EDF Header:")
    print("------------")
    print(f.getHeader())
    # print("EDF Version:", f.getHeader()['version'])
    print("Patient ID:", f.getHeader()['patient_additional'])
    print("Recording Date:", f.getStartdatetime().isoformat())
    print("Number of Signals:", f.signals_in_file)
    print("Signal Labels:", f.getSignalLabels())
    # print("Signal Physical Dimensions:", f.getPhysicalDimension())
    print("Signal Sample Rates:", f.getSampleFrequencies())

    # 打印每个信号的前几个样本
    print("\nSignal Data:")
    print("------------")
    for i in range(f.signals_in_file):
        print("Signal Label:", f.getLabel(i))
        print("Signal Sample Data:", f.readSignal(i)[:10])  # 打印前10个样本
        print("--------------")

    # 关闭EDF文件
    f.close()

# 替换 'your_edf_file.edf' 为你的EDF文件路径
edf_file_path = r'D:\pycharm\pythonProject\plot_eeg\data\edf\s01.edf'

# 读取并打印EDF文件内容
read_edf_file(edf_file_path)

#=====================检验转化txt文件后的行和列数情况，以及每行对应原始edf信号通道============
def read_txt_file(file_path):
    # 初始化行数和列数的变量
    num_rows = 0
    num_cols = 0

    # 尝试打开文件
    try:
        with open(file_path, 'r') as txt_file:
            # 读取文件的所有行
            lines = txt_file.readlines()

            # 假设第一行有数据，则设置列数为该行值的数量
            if lines:
                values = lines[0].split()  # 假设值之间由空格分隔
                num_cols = len(values)

                # 遍历每一行
            for line in lines:
                # 去除行尾的换行符并分割值
                values = line.strip().split()

                # 打印每行前10个值（如果可用）
                print("Row {}: {}".format(num_rows + 1, values[:10]))

                # 增加行数
                num_rows += 1

                # 打印行数和列数
            print("Number of Rows:", num_rows)
            print("Number of Columns:", num_cols)

    except FileNotFoundError:
        print(f"File {file_path} not found.")

    # 替换 'your_txt_file.txt' 为你的文本文件路径

txt_file_path = r'D:\pycharm\pythonProject\plot_eeg\data\s_txt\s01.txt'

# 读取并打印文本文件内容
read_txt_file(txt_file_path)

'''
