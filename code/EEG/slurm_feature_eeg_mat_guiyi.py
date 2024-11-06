import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import sys
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import scipy.io as scio
from time import process_time
plt.rcParams['font.family'] = 'Times New Roman'
start = process_time()

nmax = 10000
nstart = 100000
n = 19   # for 128 time series
h = 1.0e-3
delta = 10
gamma = 60
beta = 8/3
sigma = 4.0
rk1 = 7.0

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

# 加载相关矩阵文件
eegfile = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/data/h_average_corr_matrix.npy')
# eegfile = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/data/s_average_corr_matrix.npy')
CorrMat = eegfile
# Calculate max and min values for normalization
CorrMat_new_max = CorrMat.max().max()
CorrMat_new_min = CorrMat.min().min()
print('CorrMat_new_max:', CorrMat_new_max)
print('CorrMat_new_min:', CorrMat_new_min)
CorrMat_normalized = (CorrMat - CorrMat_new_min) / (CorrMat_new_max - CorrMat_new_min)
print("Normalized CorrMat:\n", CorrMat_normalized)
print(np.shape(CorrMat_normalized))

couple = float(sys.argv[1])
cutoff = float(sys.argv[2])

for i in range(n):
    for j in range(n):
        if j != i:
            # distance = (ATOM_X[i]-ATOM_X[j])**2+(ATOM_Y[i]-ATOM_Y[j])**2+(ATOM_Z[i]-ATOM_Z[j])**2
            # if distance_matrix_original[i][j] < cutoff[1]:
            if CorrMat_normalized[i][j] < cutoff:
                mat[i][j] = -1      # this value does not agree with 1 in Eq(1) of Chaos paper
            else:
                mat[i][j] = 0
            # mat[i][i] = mat[i][i] - mat[i][j]
print('mat:',mat)

for i in range(nmax+nstart+1):
    time = h * i
    x, y, z = dery(x,y,z,n,h,delta,gamma,beta,rk1,couple,mat)
    # print('i=',i)
    if np.mod(i,1000) == 0:
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
            # singlec03s10.append([x[0],y[0],z[0],x[49],y[49],z[49]])
            singlec03s10.append([x[0], y[0], z[0], x[18], y[18], z[18]])
# # print('size of amplitudezc03s10:',np.shape(amplitudezc03s10))

np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_singlec03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple),singlec03s10)
np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_amplitudezc03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple),amplitudezc03s10)
np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_couplematrix_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple),mat)
# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_singlec03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple),singlec03s10)
# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_amplitudezc03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple),amplitudezc03s10)
# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_couplematrix_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple),mat)

# ========================== step 6 feature =====================
f1 = np.mean(amplitudezc03s10,axis=0)  #对amplitudezc03s10每一列进行平均
f2 = np.max(amplitudezc03s10,axis=0)
f3 = np.min(amplitudezc03s10,axis=0)
f4 = np.median(amplitudezc03s10,axis=0)
f5 = np.var(amplitudezc03s10,axis=0)
f6 = np.std(amplitudezc03s10,axis=0)
feature = np.zeros((n, 6))
feature[:, 0] = f1  # 将 f1 的值赋给数组 feature 的第一列
feature[:, 1] = f2
feature[:, 2] = f3
feature[:, 3] = f4
feature[:, 4] = f5
feature[:, 5] = f6
np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_feature_cutoff_%.3f_couple_%.3f.npy' % (cutoff,couple), feature)
# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_feature_cutoff_%.3f_couple_%.3f.npy' % (cutoff,couple), feature)
# ============================================= plot =========================================
singlemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_singlec03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple))
amplitudemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_amplitudezc03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple))
couplemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/h_npy/h_couplematrix_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple))
# singlemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_singlec03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple))
# amplitudemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_amplitudezc03s10_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple))
# couplemat = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/s_npy/s_couplematrix_cutoff_%.3f_couple_%.3f.npy'%(cutoff,couple))

singlemat = np.array(singlemat)
amplitudemat = np.array(amplitudemat)
couplemat = np.array(couplemat)

# ==================== plot1 ====================
ax = plt.matshow(amplitudemat,cmap='rainbow', fignum=0, aspect="auto")
y = plt.colorbar(ax.colorbar)
y.ax.tick_params(labelsize=16) #设置颜色条的刻度标签的字体大小为16
plt.xlabel('electrode number', fontsize=28)
plt.ylabel('Time', fontsize=28)
plt.tick_params(labelsize=16) #设置坐标轴刻度标签的字体大小
ax = plt.gca()
ax.xaxis.set_ticks_position('bottom') # x 轴刻度线的位置设置在图形的底部
ax.invert_yaxis()
# plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/h_hpcc_electrode_number_time_filtration_%.3f_couple_%.3f.png' % (cutoff, couple), dpi=1200, bbox_inches ='tight')
# plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/s_hpcc_electrode_number_time_filtration_%.3f_couple_%.3f.png' % (cutoff, couple), dpi=1200, bbox_inches ='tight')
# plt.savefig(r'/public/home/chenlong666/desktop/my_desk1/h01task/figure/h_hpcc_electrode_number_time_filtration_%.3f_couple_%.3f.svg' % (cutoff, couple), dpi=1200, bbox_inches ='tight')
# plt.show()
plt.close()

# ax = plt.matshow(amplitudemat, cmap='rainbow', fignum=0, aspect="auto")
# # 移除颜色条
# plt.colorbar(ax.colorbar).remove()
# # 移除坐标轴标签和刻度
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
#                 labelbottom=False, labeltop=False, labelleft=False, labelright=False)
# plt.savefig('1.png', dpi=1200, bbox_inches='tight')
# plt.savefig('1.svg', dpi=1200, bbox_inches='tight')
# plt.close()

# ==================== plot2 ====================
plt.plot(singlemat[:,0],singlemat[:,2],color='#6ADC88', marker='o', linestyle='-')
plt.xlabel('X', fontsize=24)
plt.ylabel('Z', fontsize=24)
plt.tick_params(labelsize=16)
plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/h_hpcc_X_Z_filtration_%.3f_couple_%.3f.png' % (cutoff, couple), dpi=1200,bbox_inches ='tight')
# plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/s_hpcc_X_Z_filtration_%.3f_couple_%.3f.png' % (cutoff, couple), dpi=1200,bbox_inches ='tight')
# plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/h_hpcc_X_Z_filtration_%.3f_couple_%.3f.svg' % (cutoff, couple), dpi=1200,bbox_inches ='tight')
#plt.show()
plt.close()

# ==================== plot3 ====================
ax = plt.matshow(couplemat,cmap='viridis', aspect="auto")
y = plt.colorbar(ax.colorbar)
y.ax.tick_params(labelsize=20) #设置颜色条的刻度标签的字体大小为16
plt.xlabel('electrode number', fontsize=26, labelpad=8)  # 调整X轴标签的字体大小
plt.ylabel('electorde number', fontsize=26, labelpad=1)    # 调整Y轴标签的字体大小并设置间隔
plt.tick_params(labelsize=20)
# plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/s_hpcc_corr_matrix_filtration_%.3f_couple_%.3f.png' % (cutoff, couple),dpi=1200, bbox_inches ='tight')
plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/h_hpcc_corr_matrix_filtration_%.3f_couple_%.3f.png' % (cutoff, couple),dpi=1200, bbox_inches ='tight')
# plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture/h_hpcc_corr_matrix_filtration_%.3f_couple_%.3f.svg' % (cutoff, couple),dpi=1200, bbox_inches ='tight')
#plt.show()
plt.close()

#=============================将三个图绘制一起
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Plot 1 - Amplitudemat
print("正在创建图1...")
ax = axes[0]
im = ax.matshow(amplitudemat, cmap='rainbow', aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_xlabel('electrode number', fontsize=22)
ax.set_ylabel('Time', fontsize=22)
ax.tick_params(labelsize=16)
ax.xaxis.set_ticks_position('bottom')
ax.invert_yaxis()
# Plot 2 - Singlemat
print("正在创建图2...")
ax = axes[1]
ax.plot(singlemat[:,0],singlemat[:,2],color='#6ADC88', marker='o', linestyle='-')
ax.set_xlabel('X', fontsize=22)
ax.set_ylabel('Z', fontsize=22)
ax.tick_params(labelsize=14)
# Plot 3 - Couplemat
print("正在创建图3...")
ax = axes[2]
im = ax.matshow(couplemat, cmap='viridis', aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_xlabel('electrode number', fontsize=20)  # 调整X轴标签的字体大小
ax.set_ylabel('electrode number', fontsize=20)   # 调整Y轴标签的字体大小
#ax.tick_params(labelsize=14)  # 调整刻度标签的字体大小
# 调整子图之间的间距和位置，避免横纵坐标与图重叠
plt.subplots_adjust(wspace=0.4)
plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture_all_h/couple_%.3f_cutoff_%.3f_all_h.png' % (couple,cutoff), bbox_inches='tight')
# plt.savefig('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/picture_all_s/couple_%.3f_cutoff_%.3f_all_s.png'% (couple,cutoff), bbox_inches='tight')
plt.close()

end = process_time()
time = (end - start)/3600
run_time = np.around(time,3)
print("run time：%.3f  h" % run_time)