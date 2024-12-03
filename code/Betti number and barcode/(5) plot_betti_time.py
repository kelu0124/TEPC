import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import gudhi
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import scipy.io as scio

plt.rcParams["font.family"] = "Times New Roman" #绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示

subjectID = 'healthy'
# subjectID = 'schizophrenia'

# eegfile = np.loadtxt(r'C:\Users\administered\Desktop\图3图4\data_eeg\%s_EC.txt'%subjectID)
# CorrMat = np.corrcoef(eegfile,rowvar=1)  # rowvar=1 对行进行分析
# CorrMat_new = pd.DataFrame(CorrMat.round(3))
# CorrMat_new[CorrMat_new < 0] = 0               #  将负相关的地方令值为0
# CorrMat_new[np.eye(128,dtype=bool)] = 0      # 令对角元素全为0，128为matrix的维数,也是时间序列的个数

CorrMat_new = np.load(r'D:\python\result\EEG\npy\couple_npy/%s_average_corr_matrix.npy'%subjectID[0])
# CorrMat_new_max = max(CorrMat_new.max())
# CorrMat_new_min = min(CorrMat_new.min())
bin_num = 10

# cutoff = [CorrMat_new_min + (x+1) * (CorrMat_new_max - CorrMat_new_min)/bin_num for x in range(bin_num)]
# print('cutoff:',cutoff)
# print(len(cutoff))
# couple = np.arange(0,11,1)
# couple = np.arange(0.1,1,0.1)
# couple = np.arange(0.05,1,0.1)

# couple = [0.1, 0.42, 0.9]
couples = [2.8]
if subjectID == 'healthy':
    cutoffs = [0.070,0.141,0.211,0.281,0.351,0.422,0.492,0.562,0.633,0.703] # healthy
    # cutoffs = [0.035, 0.070, 0.105, 0.141, 0.176, 0.211,0.246, 0.281, 0.316, 0.351, 0.386, 0.422,0.457, 0.492, 0.527, 0.562, 0.597, 0.632,0.667, 0.703]
else:
    cutoffs = [0.081,0.161,0.241,0.322,0.403,0.483,0.564,0.644,0.725,0.805] # schizophrenia
    # cutoffs = [0.040, 0.081, 0.121, 0.161, 0.201, 0.242, 0.282, 0.322, 0.362, 0.403, 0.443, 0.483, 0.523, 0.564, 0.604, 0.644, 0.684, 0.725, 0.765, 0.805]
cutoffs = [0.351,]
for j in range(len(couples)):
    for i in range(len(cutoffs)):
        print(cutoffs[i],couples[j])
        #================  Time-betti number  =============
        betti_num_time = np.load(r'D:\python\result\EEG\npy\betti_npy/evolution_betti_number_%s_cutoff_%.3f_couple_%.3f.npy' % (subjectID, cutoffs[i], couples[j]))

        # plt.plot(betti_num_time[:, 0], betti_num_time[:, 1], 'g-.o')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,2], 'r')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,3], 'b')

        plt.plot(betti_num_time[:, 1], betti_num_time[:, 0], color = (0.9, 0.2, 0.2),label = 'Betti 0')  # 交换 x 轴和 y 轴的绘制顺序
        plt.plot(betti_num_time[:,2], betti_num_time[:, 0], color = (0.1, 0.5, 0.8),label = 'Betti 1')
        plt.plot(betti_num_time[:,3], betti_num_time[:, 0], 'g',label = 'Betti 2')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,1], 'g-.o',label='betti 0')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,2], 'r',label='betti 1')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,3], 'b',label='betti 2')
        # lable 加粗和放大
        # font = FontProperties(weight='bold',  size=18)
        # plt.legend(prop=font)
        plt.xlim(-3, 25)
        plt.ylim(0, 1000)
        # 设置刻度标签的字体大小
        plt.tick_params(labelsize=18)
        #plt.title('%s_cutoff_%.2f_couple_%.2f'%(subjectID,cutoff[i],couple[j]))
        plt.xlabel('betti number', fontsize=26)
        plt.ylabel('Time', fontsize=26, labelpad=-5)      #      , labelpad=0.01)
        plt.legend(fontsize = 14, loc = 'upper center')

        plt.savefig(r'D:\python\result\EEG\fig\evolution_betti/evolution_betti_number_%s_cutoff_%.3f_couple_%.3f_去除空白.png' % (subjectID, cutoffs[i], couples[j]), dpi=1200, bbox_inches='tight')
        # plt.savefig('C:/Users/administered/Desktop/MND笔记/protein/柯璐-24.3.11/evolution_betti_number_%s_cutoff_%.2f_couple_%.2f_去除空白.svg' % (subjectID, cutoff[i], couple[j]), dpi=1200, bbox_inches='tight')
        plt.tight_layout()  # 自动调整子图布局，使之更紧凑
        #plt.show()
        plt.close()

        # # 确保betti0\1\2出现的每个值只输出一次
        # seen_values = set()
        # # 遍历 betti0_list
        # print("betti 0:")
        # for value in betti_num_time[:, 1]:
        #     if value not in seen_values:
        #         print(value)
        #         seen_values.add(value)
        # # 遍历 betti1_list
        # print("betti 1:")
        # for value in betti_num_time[:, 2]:
        #     if value not in seen_values:
        #         print(value)
        #         seen_values.add(value)
        # # 遍历 betti2_list
        # print("betti 2:")
        # for value in betti_num_time[:, 3]:
        #     if value not in seen_values:
        #         print(value)
        #         seen_values.add(value)
