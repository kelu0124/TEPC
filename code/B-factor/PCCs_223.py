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
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error, r2_score

pdbID = str(sys.argv[1])

atom_num = []
B_factor = []
atomnum = len(open(r'/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID, 'r').readlines())
atom_num.append(atomnum)
pdbfile = open('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID, 'r')
tem_b_factor = []  # 记录单个蛋白的B-factor
for line in pdbfile.readlines():
    tem_b_factor.append(float(line.split()[3]))
B_factor.append(tem_b_factor)

couple = float(sys.argv[2])
cutoff = float(sys.argv[3])

##########################  多元线性回归 ################

#223
X = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/result/1_364_3_1/1_%s_3_1/npy_%s_3_1/X_%s_%d_%.2f_3_1.npy'%(pdbID,pdbID,pdbID,couple,cutoff))

# #134+7
# X = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/2_134_3_1_out/2_134_3_1_npy/X_%s_%d_%.2f_3_1.npy'%(pdbID,couple,cutoff))

# Y = B_factor[i]
yTrue = np.array(B_factor[0]).reshape(-1, 1)
print(yTrue.shape)
# print('Y:',Y)
model = sm.OLS(yTrue, sm.add_constant(X)) #生成模型, need to add constant by hand
result = model.fit() #模型拟合
result.summary() #模型描述
print('results:',result.summary())
print('Parameters:', result.params)  # 输出模型的参数估计值
yFit = result.fittedvalues  # 拟合模型计算出的 y值
print(np.shape(yFit), np.shape(yTrue))
# 保存 yFit 到文件
np.save('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/PCC_223/Y_npy/yFit_%s_%d_%.2f_3_1.npy'%(pdbID, couple, cutoff), yFit)

# 计算yTrue与yFit的皮尔逊系数
#pearson_r, _ = scipy.stats.pearsonr(yTrue.flatten(), yFit.flatten())
#pearson_r = np.around(pearson_r, 3)

# # 计算yTrue与yFit的皮尔逊系数
pearson_r = scipy.stats.pearsonr(yTrue, yFit)
pearson_r = np.around(pearson_r, 3)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(yTrue, yFit))
rmse = np.around(rmse, 3)

# 计算决定系数（R^2）
r_squared = r2_score(yTrue, yFit)
r_squared = np.around(r_squared, 3)
# 将评价指标保存到CSV文件中
df = pd.DataFrame({'pdbID': [pdbID], 'n_num': [atom_num[0]], 'couple': [couple], 'cutoff': [cutoff],
                'Pearson_r': [pearson_r], 'RMSE': [rmse], 'R_squared': [r_squared]})
# df.to_csv('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/result/1_364_3_1/1_%s_3_1/csv_%s_3_1/R_%s_%d_%.2f_3_1.csv'%(pdbID,pdbID, pdbID, couple, cutoff), index=False, header=False)
df.to_csv('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/PCC_223/result/PCC_%s_%d_%.2f_3_1.csv'%(pdbID, couple, cutoff), index=False, header=False)
