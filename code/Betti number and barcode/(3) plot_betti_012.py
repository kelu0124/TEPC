import numpy as np
import random
import matplotlib.pyplot as plt
import gudhi
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import scipy.io as scio

plt.rcParams["font.family"] = "Times New Roman"  # 绘图中的所有文本元素，包括坐标轴标签和刻度标签，都将使用Times New Roman字体显示

nmax = 10000
nstart = 100000
n = 19   # for 128 time series
h = 1.0e-3
delta = 10
gamma = 60
beta = 8/3
sigma = 4.0
# couple = 1
rk1 = 7.0

# subjectID = 'healthy'
subjectID = 'schizophrenia'

x = [random.random() for _ in range(n)]
y = [random.random() for _ in range(n)]
z = [random.random() for _ in range(n)]
dx, dy, dz = np.zeros(n), np.zeros(n), np.zeros(n)
ATOM_X, ATOM_Y, ATOM_Z = np.zeros(n), np.zeros(n), np.zeros(n)
mat = np.zeros((n,n))
singlec03s10 = []
amplitudezc03s10 = []

def dery(x,y,z,n,h,delta,gamma,beta,rk,couple,mat):
    for i in range(n):
        coupley = 0
        for j in range(n):
            if i != j:
                coupley = coupley + couple*mat[i][j]*(x[i]-x[j])
        if i == 0:
            i1 = n-1
        else:
            i1 = i - 1
        if i == n -1:
            i2 = 0
        else:
            i2 = i + 1
        cc1 = rk*(x[i2] - x[i1])
        dx[i] = x[i] + h*(delta*(y[i] - x[i]))
        dy[i] = y[i] + h*(gamma*x[i] - y[i]-x[i]*z[i]+ coupley +cc1)
        dz[i] = z[i] + h*(x[i]*y[i] - beta*z[i])
    for i in range(n):
        x[i] = dx[i]
        y[i] = dy[i]
        z[i] = dz[i]
    return x, y, z

# eegfile = np.loadtxt('/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/EEG/dataset/%s_average_corr_matrix.txt'%subjectID[0])
# CorrMat = np.corrcoef(eegfile,rowvar=1)  # rowvar=1 对行进行分析
# CorrMat_new = pd.DataFrame(CorrMat.round(3))
# CorrMat_new[CorrMat_new < 0] = 0               #  将负相关的地方令值为0
# CorrMat_new[np.eye(19,dtype=np.bool)] = 0      # 令对角元素全为0，128为matrix的维数,也是时间序列的个数

# CorrMat_new = pd.DataFrame(np.load(
#     '/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/EEG/dataset/%s_average_corr_matrix.npy'%subjectID[0]))
CorrMat_new = pd.DataFrame(np.load(
    r'D:\python\result\EEG\npy\couple_npy/%s_average_corr_matrix.npy'%subjectID[0]))

# CorrMat_new_max = max(CorrMat_new.max())
# CorrMat_new_min = min(CorrMat_new.min())

# bin_num = 10
# cutoff = [CorrMat_new_min + (x+1) * (CorrMat_new_max - CorrMat_new_min)/bin_num for x in range(bin_num)]
# print('cutoff:',cutoff)
# print(len(cutoff))
#
# couple = np.arange(0,11,1)
# # couple = np.arange(0.1,1,0.1)
# # couple = np.arange(0.05,1,0.1)
# print('couple:',couple)
# print(len(couple))
# cutoff = 0.07
# couple = 0

couples = np.arange(2.8,3.1,2)
if subjectID == 'healthy':
    # cutoffs = [0.070,0.141,0.211,0.281,0.351,0.422,0.492,0.562,0.633,0.703] # healthy
    cutoffs = [0.035, 0.070, 0.105, 0.141, 0.176, 0.211,0.246, 0.281, 0.316, 0.351, 0.386, 0.422,0.457, 0.492, 0.527, 0.562, 0.597, 0.632,0.667, 0.703]
else:
    # cutoffs = [0.081,0.161,0.241,0.322,0.403,0.483,0.564,0.644,0.725,0.805] # schizophrenia
    cutoffs = [0.040, 0.081, 0.121, 0.161, 0.201, 0.242, 0.282, 0.322, 0.362, 0.403, 0.443, 0.483, 0.523, 0.564, 0.604, 0.644, 0.684, 0.725, 0.765, 0.805]

#cutoffs = [0.141]

for couple in couples:
    for cutoff in cutoffs:
        print('cutoff:', cutoff)
        print('couple:', couple)
        for i in range(n):
            for j in range(n):
                if j != i:
                    # distance = (ATOM_X[i]-ATOM_X[j])**2+(ATOM_Y[i]-ATOM_Y[j])**2+(ATOM_Z[i]-ATOM_Z[j])**2
                    # if distance_matrix_original[i][j] < cutoff[1]:
                    if CorrMat_new[i][j] < cutoff:
                        mat[i][j] = -1      # this value does not agree with 1 in Eq(1) of Chaos paper
                    else:
                        mat[i][j] = 0
                    # mat[i][i] = mat[i][i] - mat[i][j]

        #================  plot filtration parameter - barcode  =============
        amplitudemat = np.load(r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/%s_singlec03s10_cutoff_%.3f_couple_%.3f.npy' % (
            subjectID[0], cutoff, couple), singlec03s10)
        # amplitudemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/%s_npy/%s_amplitudezc03s10_cutoff_%.3f_couple_%.3f.npy' %(subjectID[0],subjectID[0],cutoff,couple))
        # print(amplitudemat[0:2,:])

        #========    绘制持久性条形码（persistence barcode）
        rips_complex = gudhi.RipsComplex(points=mat, max_edge_length=6)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        diag = simplex_tree.persistence(min_persistence=0)
        # plt.rcParams['font.size'] = 22
        gudhi.plot_persistence_barcode(diag)
        # plt.show()
        # exit()
        print(diag)
        # exit()
        diag_dim_0 = [  i  for i in diag if i[0]==0]
        diag_dim_1 = [  i  for i in diag if i[0]==1]
        diag_dim_2 = [  i  for i in diag if i[0]==2]
        print(len(diag_dim_0))
        print(len(diag_dim_1))
        print(len(diag_dim_2))

        diag_dim_0[0] = (0,(0,6))  # set inf in the diag into 12
        # print(diag_dim_2)
        # print(diag_dim_2[0][1][0],diag_dim_2[0][1][1])
        # print(sorted(diag_dim_2, key=lambda x: np.abs(x[1][0]-x[1][1])))
        # exit()
        # diag_dim_0_sorted = sorted(diag_dim_0, key=lambda x: np.abs(x[1][0]-x[1][1]),reverse=True)
        # diag_dim_1_sorted = sorted(diag_dim_1, key=lambda x: np.abs(x[1][0]-x[1][1]),reverse=True)
        # diag_dim_2_sorted = sorted(diag_dim_2, key=lambda x: np.abs(x[1][0]-x[1][1]),reverse=True)
        diag_dim_0_sorted = sorted(diag_dim_0, key=lambda x: np.abs(x[1][0]))
        diag_dim_1_sorted = sorted(diag_dim_1, key=lambda x: np.abs(x[1][0]))
        diag_dim_2_sorted = sorted(diag_dim_2, key=lambda x: np.abs(x[1][0]))
        print(len(diag_dim_0_sorted))
        print(len(diag_dim_1_sorted))
        print(len(diag_dim_2_sorted))

        # max_value_diag_dim_0 = max(x[1][1] for x in diag_dim_0_sorted)
        # if diag_dim_1_sorted != []:
        #     max_value_diag_dim_1 = max(x[1][1] for x in diag_dim_1_sorted)
        # else:
        #     max_value_diag_dim_1 = 0
        #
        # if diag_dim_2_sorted != []:
        #     max_value_diag_dim_2 = max(x[1][1] for x in diag_dim_2_sorted)
        # else:
        #     max_value_diag_dim_2 = 0
        # max_value = max(max_value_diag_dim_0,max_value_diag_dim_1,max_value_diag_dim_2)
        #
        # diag_dim_0_sorted = [(x[0], (x[1][0], x[1][1] / max_value)) for x in diag_dim_0_sorted]
        # if diag_dim_1_sorted != []:
        #     diag_dim_1_sorted = [(x[0], (x[1][0], x[1][1] / max_value)) for x in diag_dim_1_sorted]
        # if diag_dim_2_sorted != []:
        #     diag_dim_2_sorted = [(x[0], (x[1][0], x[1][1] / max_value)) for x in diag_dim_2_sorted]

        # num = 0
        # for point in diag_dim_2_sorted:
        #     num +=1
        #     plt.plot(point[1],[num]*2,'r-',linewidth=8)
        # plt.xlabel('filtration parameter',fontsize=16)
        # plt.ylabel('Betti 2',fontsize=16)
        # plt.axis([0,12,0,5])
        # plt.show()
        # exit()

        #================  plot filtration parameter - barcode  =============
        fig = plt.figure(dpi=100, figsize=(8,8))  #约束布局（constrained_layout=True）
        # fig = plt.figure(dpi=100, constrained_layout=True)
        gs = GridSpec(6,1,figure=fig,hspace=1)

        ax1 = fig.add_subplot(gs[0:3,0])
        num = 0
        for point in diag_dim_0_sorted:
            num += 1
            plt.plot(point[1],[num]*2,color=(0.9, 0.2, 0.2),linestyle = '-',linewidth=4)
        # plt.xlabel('filtration parameter',fontsize=16)
        plt.text(5.83,0.3,'>',color=(0.9, 0.2, 0.2),fontsize=20, weight='bold')  #调整箭头向右，向下，加粗
        plt.ylabel('$\\beta_0$',fontsize=18, labelpad=1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=14)
        plt.axis([0,6,-2,21])
        plt.xticks(np.linspace(0, 6, 11))
        ax1.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        ax2 = fig.add_subplot(gs[3:5,0])
        num = 0
        for point in diag_dim_1_sorted:
            num += 1
            plt.plot(point[1],[num]*2,color=(0.1, 0.5, 0.8),linestyle = '-',linewidth=4)
        # plt.xlabel('filtration parameter',fontsize=16)
        plt.ylabel('$\\beta_1$',fontsize=18, labelpad=8)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=14)
        plt.axis([0,6,0,6])
        plt.xticks(np.linspace(0, 6, 11))
        ax2.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        ax3 = fig.add_subplot(gs[5:6,0])
        num = 0
        for point in diag_dim_2_sorted:
            num += 1
            plt.plot(point[1],[num]*2,'g-',linewidth=4)
        plt.xlabel('filtration parameter',fontsize=26)
        plt.ylabel('$\\beta_2$',fontsize=18, labelpad=3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=14)
        plt.axis([0,6,0,2])
        plt.xticks(np.linspace(0, 6, 11))
        ax3.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0])
        # plt.tight_layout()
        # plt.savefig('/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/EEG/fig/betti/persistence_barcode_rips_%s_cutoff_%.3f_couple_%.3f.png'%(subjectID,cutoff,couple),bbox_inches='tight')
        plt.savefig(r'D:\python\result\EEG\fig/persistence_barcode_rips_%s_cutoff_%.2f_couple_%.2f.png'%(subjectID,cutoff,couple),bbox_inches='tight')
        # plt.show()
        plt.close()
        #break
