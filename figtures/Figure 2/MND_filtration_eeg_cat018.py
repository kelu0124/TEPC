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
plt.rcParams['font.sans-serif']=['Times New Roman']

start = process_time()

nmax = 5000
nstart = 50000
n = 120   # for 120 time series
h = 1.0e-3
delta = 10
gamma = 60
beta = 8/3
sigma = 4.0
# couple = 1
rk1 = 7.0
subjectID = 'atom'

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
        dx[i] = x[i] + h*(delta*(y[i] - x[i])+ coupley +cc1)
        dy[i] = y[i] + h*(gamma*x[i] - y[i]-x[i]*z[i]+ coupley +cc1)
        dz[i] = z[i] + h*(x[i]*y[i] - beta*z[i]+ coupley +cc1)
    for i in range(n):
        x[i] = dx[i]
        y[i] = dy[i]
        z[i] = dz[i]
    return x, y, z

# fMRIfile = open('PDB_data/fMRI_EEG_data/data/fMRI/ROISignalAAL90HCP_%s.txt'%subjectID,'r')
# fMRIfile = np.loadtxt('PDB_data/fMRI_EEG_data/data/fMRI/ROISignalAAL90HCP_%s.txt'%subjectID)
# fMRIfile = np.loadtxt('ROISignalAAL90HCP_%s.txt'%subjectID)
# print(np.shape(fMRIfile))
# print(type(fMRIfile))
# eegfile = np.loadtxt('PDB_data/fMRI_EEG_data/data/eeg/%s_EC.txt'%subjectID)
# eegfile = np.loadtxt('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/egg/data_eeg/%s_EC.txt'%subjectID)
eegfile = np.loadtxt(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/atom_coordinates.txt')

print(np.shape(eegfile))
CorrMat = np.corrcoef(eegfile,rowvar=1)  # rowvar=1 对行进行分析
# print('Pearson coefficient matrix:',CorrMat)
print(np.shape(CorrMat))

CorrMat_new = pd.DataFrame(CorrMat.round(3))
print(CorrMat_new)
# print(np.amin(CorrMat), np.amax(CorrMat))

CorrMat_new[CorrMat_new < 0] = 0               #  将负相关的地方令值为0
# np.fill_diagonal(CorrMat_new,0)
CorrMat_new[np.eye(120,dtype=np.bool)] = 0      # 令对角元素全为0，120为matrix的维数
print(CorrMat_new)
CorrMat_new_max = max(CorrMat_new.max())
CorrMat_new_min = min(CorrMat_new.min())
print(CorrMat_new_max, CorrMat_new_min)
bin_num = 5
cutoff = [CorrMat_new_min + (x+1) * (CorrMat_new_max - CorrMat_new_min)/bin_num for x in range(bin_num)]
print('cutoff:',cutoff)
#exit()
cutoff = float(sys.argv[1])
couple = float(sys.argv[2])

# cutoff = 0.3
# couple = 0.9

for i in range(n):
    for j in range(n):
        if j != i:
            # distance = (ATOM_X[i]-ATOM_X[j])**2+(ATOM_Y[i]-ATOM_Y[j])**2+(ATOM_Z[i]-ATOM_Z[j])**2
            # if distance_matrix_original[i][j] < cutoff[1]:
            if CorrMat_new[i][j] < cutoff:
                mat[i][j] = -1      # this value does not agree with 1 in Eq(1) of Chaos paper
            else:
                mat[i][j] = 0
            # mat[i][i] = mat[i][i] - mat[i][j]  #注释不同/不注释一条线
# print('mat:',mat)
# print('diagonal element of mat:',np.diagonal(mat))
# print('max of mat:',np.amax(mat),'min of mat:',np.amin(mat))
# exit()

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
            singlec03s10.append([x[0],y[0],z[0],x[49],y[49],z[49]])
# # print('size of amplitudezc03s10:',np.shape(amplitudezc03s10))

# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/egg/result_egg/singlec03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple),singlec03s10)
# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/egg/result_egg/amplitudezc03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple),amplitudezc03s10)
# np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/egg/result_egg/couplematrix_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple),mat)
np.save(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/特征/singlec03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple),singlec03s10)
np.save(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/特征/amplitudezc03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple),amplitudezc03s10)
np.save(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/特征/couplematrix_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple),mat)
singlemat = np.load(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/特征/singlec03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple))
amplitudemat = np.load(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/特征/amplitudezc03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple))
couplemat = np.load(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/特征/couplematrix_%s_cutoff_%.2f_couple_%.2f.npy'%(subjectID,cutoff,couple))

singlemat = np.array(singlemat)
amplitudemat = np.array(amplitudemat)
couplemat = np.array(couplemat)

# # ==================== plot ====================
# ax = plt.matshow(amplitudemat,cmap='rainbow', fignum=0, aspect="auto")
# y = plt.colorbar(ax.colorbar)
# y.ax.tick_params(labelsize=24) #设置颜色条的刻度标签的字体大小为16
# # plt.xlabel('electrode number', fontsize=24)
# # plt.ylabel('Time', fontsize=24)
# plt.xlabel('Residue number', fontsize=32)
# plt.ylabel('Time', fontsize=32)
# plt.tick_params(labelsize=24) #设置坐标轴刻度标签的字体大小
# ax = plt.gca()
# ax.xaxis.set_ticks_position('bottom') # x 轴刻度线的位置设置在图形的底部
# ax.invert_yaxis()
# #plt.title('{}, filtration = {:.3f}'.format(subjectID, cutoff[ii]), fontsize=20)
# # plt.rcParams['figure.figsize'] = (8.0, 4.0)
# plt.savefig(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/图片/hpcc_electrode_number_time_filtration_%.2f_couple_%.2f.png' % (cutoff, couple), dpi=1200, bbox_inches ='tight')
# # plt.show()
# plt.close()

ax = plt.matshow(amplitudemat, cmap='rainbow', fignum=0, aspect="auto")
plt.axis('off')  # 关闭坐标轴和标签显示
colorbar = plt.colorbar(ax.colorbar)
colorbar.remove()  # 去掉颜色条
plt.savefig(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/图片/hpcc_electrode_number_time_filtration_%.2f_couple_%.2f.png' % (cutoff, couple), dpi=1200, bbox_inches='tight')
plt.savefig(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/图片/hpcc_electrode_number_time_filtration_%.2f_couple_%.2f.svg' % (cutoff, couple), dpi=1200, bbox_inches='tight')
plt.close()


# plt.plot(singlemat[:,0],singlemat[:,2],'ro-')
# plt.xlabel('X', fontsize=24)
# plt.ylabel('Z', fontsize=24)
# plt.tick_params(labelsize=16)
# #plt.title(' {}, filtration = {:.3f}'.format(subjectID, cutoff[ii]), fontsize=20)
# plt.savefig('/public/home/chenlong666/desktop/my_desk1/process/hpcc_X_Z_filtration_%.2f_couple_%.2f.png' % (cutoff, couple), bbox_inches ='tight')
# #plt.show()
# plt.close()
#
# #======================== 3 ==================
# ax = plt.matshow(couplemat,cmap='rainbow', aspect="auto")
# y = plt.colorbar(ax.colorbar)
# y.ax.tick_params(labelsize=16) #设置颜色条的刻度标签的字体大小为16
# plt.xlabel('electrode number', fontsize=24, labelpad=8)  # 调整X轴标签的字体大小
# plt.ylabel('electorde number', fontsize=24, labelpad=1)    # 调整Y轴标签的字体大小并设置间隔
# plt.tick_params(labelsize=16)
# #plt.title('{}, filtration = {:.3f}'.format(subjectID, cutoff[ii]), fontsize=20)
# plt.savefig('/public/home/chenlong666/desktop/my_desk1/process/hpcc_corr_matrix_filtration_%.2f_couple_%.2f.png' % (cutoff, couple), bbox_inches ='tight')
# #plt.show()
# plt.close()


end = process_time()
time = (end - start)/3600
run_time = np.around(time,3)
print("run time：%.3f  h" % run_time)