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
import sys

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


# eegfile = np.loadtxt('PDB_data/fMRI_EEG_data/data/eeg/%s_EC.txt'%subjectID)
# eegfile = np.loadtxt('%s_EC.txt'%subjectID)
# eegfile = np.loadtxt('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/egg/data_eeg/%s_EC.txt'%subjectID)
#
# CorrMat = np.corrcoef(eegfile,rowvar=1)  # rowvar=1 对行进行分析
# CorrMat_new = pd.DataFrame(CorrMat.round(3))
# CorrMat_new[CorrMat_new < 0] = 0               #  将负相关的地方令值为0
# CorrMat_new[np.eye(19,dtype=np.bool)] = 0      # 令对角元素全为0，128为matrix的维数,也是时间序列的个数
# CorrMat_new_max = max(CorrMat_new.max())
# CorrMat_new_min = min(CorrMat_new.min())
# bin_num = 10
# cutoff = [CorrMat_new_min + (x+1) * (CorrMat_new_max - CorrMat_new_min)/bin_num for x in range(bin_num)]
# print('cutoff:',cutoff)
# print(len(cutoff))

# couple = np.arange(0.1,1,0.1)
# print('couple:',couple)
# print(len(couple))

# cutoff = cutoff[7]
# couple = 0.5
CorrMat_new = np.load(r'D:\python\result\EEG\npy\couple_npy/%s_average_corr_matrix.npy'%subjectID[0])

couples = [2.8]
if subjectID == 'healthy':
    # cutoffs = [0.070,0.141,0.211,0.281,0.351,0.422,0.492,0.562,0.633,0.703] # healthy
    cutoffs = [0.035, 0.070, 0.105, 0.141, 0.176, 0.211,0.246, 0.281, 0.316, 0.351, 0.386, 0.422,0.457, 0.492, 0.527, 0.562, 0.597, 0.632,0.667, 0.703]
else:
    # cutoffs = [0.081,0.161,0.241,0.322,0.403,0.483,0.564,0.644,0.725,0.805] # schizophrenia
    cutoffs = [0.040, 0.081, 0.121, 0.161, 0.201, 0.242, 0.282, 0.322, 0.362, 0.403, 0.443, 0.483, 0.523, 0.564, 0.604, 0.644, 0.684, 0.725, 0.765, 0.805]

for couple in couples:
    for cutoff in cutoffs:

        for i in range(n):
            for j in range(n):
                if j != i:
                    # distance = (ATOM_X[i]-ATOM_X[j])**2+(ATOM_Y[i]-ATOM_Y[j])**2+(ATOM_Z[i]-ATOM_Z[j])**2
                    # if distance_matrix_original[i][j] < cutoff[1]:
                    if CorrMat_new[i][j] < cutoff:
                        mat[i][j] = -1  # this value does not agree with 1 in Eq(1) of Chaos paper
                    else:
                        mat[i][j] = 0
                    # mat[i][i] = mat[i][i] - mat[i][j]

        temp_points = [ [] for _ in range(n)]
        betti_num_time = np.zeros((1000,4))
        for i in range(nmax+nstart+1):
            time = h * i
            x, y, z = dery(x,y,z,n,h,delta,gamma,beta,rk1,couple,mat)

            if np.mod(i,1000) == 0:
                # print(x,y,z)
                # exit()
                print(i/1000, x[0],y[0],z[0])
            if i > nstart:
                syn = 0
                expect = 0
                cov = 0
                for j in range(n):
                    expect = expect + y[j]/n
                for j in range(n):
                    cov = cov + (y[j] - expect)**2
                cov = np.sqrt(cov/n)/expect
                if np.mod(i,1000) == 0:
                    print('synchronization index:',cov)
                if np.mod(i,10) == 0:
                    tempmat = x.copy()
                    amplitudezc03s10.append(tempmat)
                    singlec03s10.append([x[0],y[0],z[0],x[9],y[9],z[9]])

                    for k in range(n):
                        temp_points[k] = [x[k],y[k],z[k]]
                    print(np.shape(temp_points))
                    rips_complex = gudhi.RipsComplex(points=temp_points, max_edge_length=6)
                    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
                    diag = simplex_tree.persistence(min_persistence=0)
                    diag_dim_0 = [  ii  for ii in diag if ii[0]==0]
                    diag_dim_1 = [  ii  for ii in diag if ii[0]==1]
                    diag_dim_2 = [  ii  for ii in diag if ii[0]==2]
                    betti_num_time[int((i-nstart)/10)-1][0] = int((i-nstart)/10)
                    betti_num_time[int((i-nstart)/10)-1][1] = len(diag_dim_0)
                    betti_num_time[int((i-nstart)/10)-1][2] = len(diag_dim_1)
                    betti_num_time[int((i-nstart)/10)-1][3] = len(diag_dim_2)

        np.save(r'D:\python\result\EEG\npy\betti_npy/evolution_betti_number_%s_cutoff_%.3f_couple_%.3f.npy'%(subjectID,cutoff,couple),betti_num_time)
        # exit()
        # betti_num_time = np.load('results/eeg/%s/evolution_betti_number_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,subjectID,cutoff,couple))
        # print(betti_num_time)
        # plt.plot(betti_num_time[:,0], betti_num_time[:,1], 'g-.o',label='betti 0')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,2], 'r',label='betti 1')
        # plt.plot(betti_num_time[:,0], betti_num_time[:,3], 'b',label='betti 2')
        # plt.legend()
        # plt.title('%s_cutoff_%.2f_couple_%.2f'%(subjectID,cutoff,couple))
        # plt.xlabel('Time')
        # plt.ylabel('betti number')
        # plt.show()


