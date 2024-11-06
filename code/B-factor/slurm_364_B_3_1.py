import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
import sys
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from time import process_time

start = process_time()

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
        cc1 = rk*(x[i2] - x[i1])        # 计算出相邻两点在x方向上的距离
        dx[i] = x[i] + h * (delta * (y[i] - x[i]))
        dy[i] = y[i] + h * (gamma * x[i] - y[i] - x[i] * z[i] + coupley + cc1)
        dz[i] = z[i] + h * (x[i] * y[i] - beta * z[i])
    for i in range(n):
        x[i] = dx[i]
        y[i] = dy[i]
        z[i] = dz[i]
    return x, y, z

pdbID = str(sys.argv[1])

atom_num = []
max_distance = []
B_factor = []
atomnum = len(open(r'/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID, 'r').readlines())
atom_num.append(atomnum)
pdbfile = open('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID, 'r')
point = []  # 记录xyz
tem_b_factor = []  # 记录单个蛋白的B-factor
for line in pdbfile.readlines():
    point.append([float(line.split()[0]), float(line.split()[1]), float((line.split()[2]))])
    tem_b_factor.append(float(line.split()[3]))
B_factor.append(tem_b_factor)
matrix = squareform(pdist(point))  # 距离矩阵
distance = dist.squareform(matrix)  # 再将这个距离矩阵转换为一个一维数组
maxdistance = np.max(distance)  # 求出数组中的最大值，即该蛋白质最大距离
max_distance.append(maxdistance)

nmax = 10000
nstart = 100000
n = atom_num[0]
h = 1.0e-3
delta = 10
gamma = 60  # 28 or 60
beta = 8/3
sigma = 4.0
couple = float(sys.argv[2])
rk1 = 7.0
#pdbID = str(sys.argv[1])
print(pdbID,n)
eta = 3  # 3 or 6
kappa = 1  # 1 or 2

x = [random.random() for _ in range(n)]
y = [random.random() for _ in range(n)]
z = [random.random() for _ in range(n)]
dx, dy, dz = np.zeros(n), np.zeros(n), np.zeros(n)
ATOM_X, ATOM_Y, ATOM_Z = np.zeros(n), np.zeros(n), np.zeros(n)
mat = np.zeros((n, n))
singlec03s10 = []
amplitudezc03s10 = []

#========================
pdbfile = open('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID, 'r')
point = []
for line in pdbfile.readlines():
    point.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
# print(np.shape(point))
distance_matrix_original = squareform(pdist(point))
distance_matrix_original = np.around(distance_matrix_original)

#################### connectivity matrix ################
connmat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            connmat[i][j] = 1 - np.exp(-(distance_matrix_original[i][j] / eta) ** kappa)
print(np.unique(dist.squareform(np.around(connmat, 2))))  # [0.  0.1 0.2 0.3 0.4 0.5 0.6]
# cutoff = np.unique(dist.squareform(np.around(connmat,2)))

################## this is Persistent Laplacian filtration process ##############
# the max of connmat_new_origin is 0.63, the min is 0, set the interval 0.63/5=0.126 for example
connmat_max = np.unique(dist.squareform(np.around(connmat, 2)))[-1]
connmat_min = np.unique(dist.squareform(np.around(connmat, 2)))[0]
print('connmat_max:', connmat_max, 'connmat_min:', connmat_min)
connmat_new_origin = np.around(connmat, 2)
print('connmat_new:', connmat_new_origin)

cutoff = float(sys.argv[3])

for i in range(n):
    for j in range(n):
        if j != i:
            # distance = (ATOM_X[i]-ATOM_X[j])**2+(ATOM_Y[i]-ATOM_Y[j])**2+(ATOM_Z[i]-ATOM_Z[j])**2
            # if distance_matrix_original[i][j] < cutoff:
            if connmat_new_origin[i][j] < cutoff:
                mat[i][j] = -1      # this value does not agree with 1 in Eq(1) of Chaos paper
            else:
                mat[i][j] = 0
            mat[i][i] = mat[i][i] - mat[i][j]  #删除,结果是否有很大改变

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
            cov = cov + (y[j] - expect) ** 2
        cov = np.sqrt(cov/n)/expect
        if np.mod(i,1000) == 0:
            print('synchronization index:',cov)
        if np.mod(i,10) == 0:
            tempmat = x.copy()
            amplitudezc03s10.append(tempmat)
            # singlec03s10.append([x[0],y[0],z[0],x[49],y[49],z[49]])
print('size of amplitudezc03s10:',np.shape(amplitudezc03s10))

##########################  多元线性回归 ################

f1 = np.mean(amplitudezc03s10,axis=0)
f2 = np.max(amplitudezc03s10,axis=0)
f3 = np.min(amplitudezc03s10,axis=0)
f4 = np.median(amplitudezc03s10,axis=0)
f5 = np.var(amplitudezc03s10,axis=0)
f6 = np.std(amplitudezc03s10,axis=0)
X = np.zeros((n,6))
X[:,0] = f1
X[:,1] = f2
X[:,2] = f3
X[:,3] = f4
X[:,4] = f5
X[:,5] = f6
print(X.shape)
print('X:',X)
np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/result/1_364_3_1/1_%s_3_1/npy_%s_3_1/X_%s_%d_%.2f_3_1.npy'%(pdbID,pdbID,pdbID,couple,cutoff),X)
# Y = B_factor[i]
X = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/result/1_364_3_1/1_%s_3_1/npy_%s_3_1/X_%s_%d_%.2f_3_1.npy'%(pdbID,pdbID,pdbID,couple,cutoff))
# exit()
Y = np.array(B_factor[0]).reshape(-1,1)
print(Y.shape)
print('Y:',Y)
model = sm.OLS(Y, sm.add_constant(X)) #生成模型, need to add constant by hand
result = model.fit() #模型拟合
result.summary() #模型描述
print('results:',result.summary())
print('Parameters:', result.params)  # 输出模型的参数估计值
yFit = result.fittedvalues  # 拟合模型计算出的 y值

R_result = []
R_squared = np.around(result.rsquared,3)
R_squared_adj = np.around(result.rsquared_adj,3)
R_result.append(R_squared)
R_result.append(R_squared_adj)
print(R_squared)
print(R_squared_adj)

end = process_time()
time = (end - start)/3600
run_time = np.around(time,3)
print("run time：%.3f  h" % run_time)

df = pd.DataFrame({'R_squared': [R_result[0]],'R_squared_adj': [R_result[1]],'couple': [couple],'cutoff': [cutoff],'run_time': [run_time]})
df.to_csv('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/result/1_364_3_1/1_%s_3_1/csv_%s_3_1/R_%s_%d_%.2f_3_1.csv'%(pdbID,pdbID, pdbID, couple, cutoff), index=False, header=False)
