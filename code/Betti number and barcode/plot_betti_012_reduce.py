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

'''
三个betti数的图，是分别关于betti0,1,2的三个图，以betti 0为例，画一个二维图，x轴是filtration 参数，
比如10个值，从0.1到 0.2 ，。。。1.0,纵坐标是不同的样本数从1到14，（14个值） 这样一共就有14*10个点，
每个点用不同的颜色表示，比如（0.1，1）这个点，反映的是第一个样本的健康人和生病人在filtration 参数为0.1时，
betti 0的差值的颜色，你要先把14*10个差值算出来，比如介于-10 和10之间，那么就搞一个colorbar，
10代表蓝色，-10代表绿色，按照这个方法把三个betti数的图画出来
'''

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

subjectID = 'healthy'
# subjectID = 'schizophrenia'

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

# CorrMat_new = pd.DataFrame(np.load(
#     r'D:\python\result\EEG\npy\couple_npy/%s_average_corr_matrix.npy'%subjectID[0]))

couples = np.arange(2.8,3.1,2)
# couples = np.arange(0,10,0.2)

healthy_cutoffs = [0.070,0.141,0.211,0.281,0.351,0.422,0.492,0.562,0.633,0.703]
schizophrenia_cutoffs = [0.081,0.161,0.241,0.322,0.403,0.483,0.564,0.644,0.725,0.805]
filtration_parameter_cutoffs = {float(i+1)/10:(healthy_cutoffs[i],schizophrenia_cutoffs[i]) for i in range(10)}

for couple in couples:
    for filtration_parameter, cutoff in filtration_parameter_cutoffs.items(): # 这里的cutoff是二维元组
        print('cutoff:', cutoff)
        print('couple:', couple)
        reduce_betti_numbers = []
        # 下面计算当前过滤参数下的病人和健康人贝蒂数
        # for people in range(1,15):

        for people in range(1,15):

            # 计算健康人贝蒂数
            eegfile = np.loadtxt(r'D:\python\result\EEG\dataset\dataverse_files\healthy/output_h%d.txt' % people)
            CorrMat = np.corrcoef(eegfile, rowvar=1)  # rowvar=1 对行进行分析
            CorrMat_new = pd.DataFrame(CorrMat.round(3))
            CorrMat_new[CorrMat_new < 0] = 0  # 将负相关的地方令值为0
            CorrMat_new[np.eye(19, dtype=np.bool)] = 0  # 令对角元素全为0，128为matrix的维数,也是时间序列的个数
            for i in range(n):
                for j in range(n):
                    if j != i:
                        if CorrMat_new[i][j] < cutoff[0]:
                            mat[i][j] = -1      # this value does not agree with 1 in Eq(1) of Chaos paper
                        else:
                            mat[i][j] = 0
            rips_complex = gudhi.RipsComplex(points=mat, max_edge_length=6)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
            diag = simplex_tree.persistence(min_persistence=0)
            healthy_betti_numbers = [len([x for x in diag if x[0] == i]) for i in range(3)]

            # 计算病人贝蒂数
            eegfile = np.loadtxt(r'D:\python\result\EEG\dataset\dataverse_files\healthy/output_h%d.txt' % people)
            CorrMat = np.corrcoef(eegfile, rowvar=1)  # rowvar=1 对行进行分析
            CorrMat_new = pd.DataFrame(CorrMat.round(3))
            CorrMat_new[CorrMat_new < 0] = 0  # 将负相关的地方令值为0
            CorrMat_new[np.eye(19, dtype=np.bool)] = 0  # 令对角元素全为0，128为matrix的维数,也是时间序列的个数
            for i in range(n):
                for j in range(n):
                    if j != i:
                        if CorrMat_new[i][j] < cutoff[1]:
                            mat[i][j] = -1  # this value does not agree with 1 in Eq(1) of Chaos paper
                        else:
                            mat[i][j] = 0
            rips_complex = gudhi.RipsComplex(points=mat, max_edge_length=6)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
            diag = simplex_tree.persistence(min_persistence=0)
            schizophrenia_betti_numbers = [len([x for x in diag if x[0] == i]) for i in range(3)]

            # 计算贝蒂数差值
            reduce_betti_number = [healthy_betti_numbers[i] - schizophrenia_betti_numbers[i] for i in range(3)]
            reduce_betti_numbers.append(reduce_betti_number) # 记录所有样本所有贝蒂数的差
            np.save(r'D:\python\result\EEG\npy/reduce_betti_%.1f.npy'%filtration_parameter,np.array(reduce_betti_numbers))
