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

def calculate_couple(i, x, Laplacian,couple):
    coupley = 0
    for j in range(n):
        if i != j:
            coupley = coupley + couple * Laplacian[i][j] * (x[i] - x[j])
    return coupley

def calculate_cc1(i, x):
    if i == 0:
        i1 = n - 1
    else:
        i1 = i - 1
    if i == n - 1:
        i2 = 0
    else:
        i2 = i + 1
    cc1 = (x[i2] - x[i1])
    return cc1

def dery(x, y, z, n, h, a, b, c, rk, couple, Laplacian):
    x = np.array(x)  # 将x转换为NumPy数组
    y = np.array(y)  # 将y转换为NumPy数组
    z = np.array(z)  # 将z转换为NumPy数组

    for i in range(n):
        k1x = h * (a * (y[i] - x[i]))
        k1y = h * ((c - a) * x[i] - x[i] * z[i] + c * y[i] + calculate_couple(i, x, Laplacian, couple) + rk * calculate_cc1(i, x))
        k1z = h * (x[i] * y[i] - b * z[i])

        k2x = h * (a * ((y[i] + 0.5 * k1y) - (x[i] + 0.5 * k1x)))
        k2y = h * ((c - a) * (x[i] + 0.5 * k1x) - (x[i] + 0.5 * k1x) * (z[i] + 0.5 * k1z) + c * (y[i] + 0.5 * k1y) + calculate_couple(i, x, Laplacian, couple) + rk * calculate_cc1(i,x + 0.5 * k1x))
        k2z = h * ((x[i] + 0.5 * k1x) * (y[i] + 0.5 * k1y) - b * (z[i] + 0.5 * k1z))

        k3x = h * (a * ((y[i] + 0.5 * k2y) - (x[i] + 0.5 * k2x)))
        k3y = h * ((c - a) * (x[i] + 0.5 * k2x) - (x[i] + 0.5 * k2x) * (z[i] + 0.5 * k2z) + c * (y[i] + 0.5 * k2y) + calculate_couple(i, x, Laplacian, couple) + rk * calculate_cc1(i, x + 0.5 * k2x))
        k3z = h * ((x[i] + 0.5 * k2x) * (y[i] + 0.5 * k2y) - b * (z[i] + 0.5 * k2z))

        k4x = h * (a * ((y[i] + k3y) - (x[i] + k3x)))
        k4y = h * ((c - a) * (x[i] + k3x) - (x[i] + k3x) * (z[i] + k3z) + c * (y[i] + k3y) + calculate_couple(i, x, Laplacian, couple) + rk * calculate_cc1(i, x + k3x))
        k4z = h * ((x[i] + k3x) * (y[i] + k3y) - b * (z[i] + k3z))

        x[i] += (1 / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y[i] += (1 / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z[i] += (1 / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
    return x, y, z

def main(couple,cutoff):
    a = 35 # 调整为 Rössler 系统的参数
    b = 3  # 调整为 Rössler 系统的参数
    c = 28 # 调整为 Rössler 系统的参数

    x = [random.random() for _ in range(n)]
    y = [random.random() for _ in range(n)]
    z = [random.random() for _ in range(n)]
    # 打印
    print("x:", x)
    print("y:", y)
    print("z:", z)

    dx, dy, dz = np.zeros(n), np.zeros(n), np.zeros(n)
    ATOM_X, ATOM_Y, ATOM_Z = np.zeros(n), np.zeros(n), np.zeros(n)
    mat = np.zeros((n,n))
    singlec03s10 = []
    amplitudezc03s10 = []

    # ========================== step1 关联矩阵（即相关系数矩阵值0~1）=====================
    CorrMat = np.corrcoef(file,rowvar=1)  # rowvar=1 对行进行分析，即每一行代表一个变量，计算不同变量之间的相关性。
    # print('Pearson coefficient matrix:',CorrMat)
    print(np.shape(CorrMat)) #(300, 300)
    CorrMat_new = pd.DataFrame(CorrMat.round(3))  #关联矩阵 CorrMat 转换为一个 Pandas DataFrame，并将矩阵中的值保留三位小数后打印出来
    # print(CorrMat_new)
    # print(np.amin(CorrMat), np.amax(CorrMat))
    CorrMat_new[CorrMat_new < 0] = 0               #  将关联矩阵中小于零的值设为零，即将负相关的地方的值置为0
    # np.fill_diagonal(CorrMat_new,0)
    CorrMat_new[np.eye(n,dtype=np.bool)] = 0      # 令对角元素全为0，300为matrix的维数
    # print(CorrMat_new)
    # ========================== step2 计算连接矩阵 =====================
    #################### connectivity matrix ################
    connmat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                connmat[i][j] = 1 - np.exp(-(CorrMat_new[i][j]/eta)**kappa)
    # print('connmat:',connmat)
    print(np.unique(dist.squareform(np.around(connmat,2))))
    # exit()
    # ========================== step3 计算cutoff =====================
    ################## this is Persistent Laplacian filtration process ##############
    connmat_max = np.unique(dist.squareform(np.around(connmat,2)))[-1]
    connmat_min = np.unique(dist.squareform(np.around(connmat,2)))[0]
    print('connmat_max:',connmat_max,'connmat_min:',connmat_min)
    connmat_new_origin = np.around(connmat,2)
    print('connmat_new:',connmat_new_origin)

    # bin_num = 5 # this number is changable. #在持久拉普拉斯滤波过程中，分箱的作用是将连接矩阵中的数值范围划分为多个区间，以便对连接强度进行离散化处理
    # cutoff = [connmat_min + (x+1) * (connmat_max-connmat_min)/bin_num for x in range(bin_num)]
    # print('cutoff:',cutoff)
    # exit()

    print('cutoff:',cutoff)
    # ========================== step4 变成连接矩阵（值0或1） =====================
    for i in range(n):
        for j in range(n):
            if j != i:
                if connmat_new_origin[i][j] < cutoff:  # 如果小于给定的阈值 cutoff，则将 mat[i][j] 设置为0
                    mat[i][j] = 0
                else:  # 否则，将 mat[i][j] 设置为1，表示两个变量之间存在连接
                    mat[i][j] = 1
                # mat[i][i] = mat[i][i] - mat[i][j]  #注释不同/不注释一条线
    print('mat:',mat)
    # print('diagonal element of mat:',np.diagonal(mat))
    # print('max of mat:',np.amax(mat),'min of mat:',np.amin(mat))
    # exit()

    # ========================== step5 构建拉普拉斯矩阵 =====================
    # 计算归一化度矩阵 D（对角元素为节点度数的倒数的平方根）
    # normalized_degrees = 1 / np.sqrt(np.sum(mat, axis=1))   #axis=1 表示沿着行的方向进行求和，即计算每个节点的度数（与节点相连的边的数量）
    # D = np.diag(normalized_degrees)
    #计算归一化度矩阵 D 时添加一些条件来处理分母为零的情况
    sum_of_degrees = np.sum(mat, axis=1)
    normalized_degrees = np.zeros_like(sum_of_degrees, dtype=np.float64)  # 创建一个与度数相同的数组，用于存储归一化度

    for i, degree_sum in enumerate(sum_of_degrees):
        if degree_sum != 0:
            normalized_degrees[i] = 1 / np.sqrt(degree_sum)

    # 创建归一化度矩阵 D
    D = np.diag(normalized_degrees)

    #计算未归一化度矩阵 D（对角元素为节点度数）
    # degrees = np.sum(mat, axis=1)   # 计算每个节点的度数（与节点相连的边的数量）
    # D = np.diag(degrees)
    # 构建拉普拉斯矩阵 L = D - mat
    Laplacian = D - mat
    # 将拉普拉斯矩阵的对角线元素设置为0
    # np.fill_diagonal(Laplacian, 0)
    # 打印拉普拉斯矩阵的形状和值
    print("Shape of Laplacian matrix:", np.shape(Laplacian))
    print("Laplacian matrix:")
    print(Laplacian)

    for i in range(nmax+nstart+1):
        time = h * i
        x, y, z = dery(x,y,z,n,h,a,b,c,rk1,couple,Laplacian)
        # print('i=',i)
        if np.mod(i,1000) == 0:
            print(i/1000, x[0],y[0],z[0])
        if (nmax/2+nstart) > i > nstart-nmax/2:
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
    print('size of amplitudezc03s10:',np.shape(amplitudezc03s10))
    # np.save(r'D:\python\result\allaml\feature\rossler/%s_singlec03s10_cutoff_%.3f_couple_%.3f.npy'%(dataset,cutoff,couple),singlec03s10)
    # np.save(r'D:\python\result\allaml\feature\rossler/%s_amplitudezc03s10_cutoff_%.3f_couple_%.3f.npy'%(dataset,cutoff,couple),amplitudezc03s10)
    # np.save(r'D:\python\result\allaml\feature\rossler/%s_Laplacian_cutoff_%.3f_couple_%.3f.npy'%(dataset,cutoff,couple),Laplacian)

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
    np.save(r'/public/home/chenlong666/desktop/my_desk1/coil_20/feature/rossler/%s_4RK_feature_cutoff_%.3f_couple_%.3f.npy' % (dataset,cutoff,couple), feature)
    # 打印feature数组的维数
    print("Feature数组维数：", feature.shape)

    end = process_time()
    time = (end - start)/3600
    run_time = np.around(time,3)
    print("run time：%.2f  h" % run_time)

if __name__ == '__main__':
    dataset = 'allaml'
    start = process_time()
    # filename = '/mnt/ufs18/home-192/jiangj33/BozhengDou/desktop/GSE/data/%s_full_X.txt'%dataset
    # file = np.loadtxt(filename)
    file = np.load(r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/coil_20_reduced.npy')
    nmax = 1000
    nstart = 10000
    n = np.shape(file)[0]  # for 300 samples
    print(np.shape(file), n)  # (300, 22431)
    h = 1.0e-4
    sigma = 4.0
    rk1 = 7.0
    eta = 3
    kappa = 1
    # cutoff = [0.184, 0.208, 0.232, 0.256, 0.28]
    # couple = np.arange(16, 31, 5)
    couple = np.array([float(sys.argv[1])])
    print("couple: ",couple)
    cutoff = [float(sys.argv[2])]
    print("cutoff: ", cutoff)

    for i in couple:
        for j in cutoff:
            main(i,j)
