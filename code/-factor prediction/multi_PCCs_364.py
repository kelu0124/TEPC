import numpy as np
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
import os

# 364
#pdbID=['1ABA', '1AHO', '1AIE', '1AKG', '1ATG', '1BGF', '1BX7', '1BYI', '1CCR', '1CYO', '1DF4', '1E5K', '1ES5', '1ETL', '1ETM', '1ETN', '1EW4', '1F8R', '1FF4', '1FK5', '1GCO', '1GK7', '1GVD', '1GXU', '1H6V', '1HJE', '1I71', '1IDP', '1IFR', '1K8U', '1KMM', '1KNG', '1KR4', '1KYC', '1LR7', '1MF7', '1N7E', '1NKD', '1NKO', '1NLS',
#	'1NNX', '1NOA', '1NOT', '1O06', '1O08', '1OB4', '1OB7', '1OPD', '1P9I', '1PEF', '1PEN', '1PMY', '1PZ4', '1Q9B', '1QAU', '1QKI', '1QTO', '1R29', '1R7J', '1RJU', '1RRO', '1SAU', '1TGR', '1TZV', '1U06', '1U7I', '1U9C', '1UHA', '1UKU', '1ULR', '1UOY', '1USE', '1USM', '1UTG', '1V05', '1V70', '1VRZ', '1W2L', '1WBE', '1WHI',
#	'1WLY', '1WPA', '1X3O', '1XY1', '1XY2', '1Y6X', '1YJO', '1YZM', '1Z21', '1ZCE', '1ZVA', '2A50', '2AGK', '2AH1', '2B0A', '2BCM', '2BF9', '2BRF', '2C71', '2CE0', '2CG7', '2COV', '2CWS', '2D5W', '2DKO', '2DPL', '2DSX', '2E10', '2E3H', '2EAQ', '2EHP', '2EHS', '2ERW', '2ETX', '2FB6', '2FG1', '2FN9', '2FQ3', '2G69', '2G7O',
#	'2G7S', '2GKG', '2GOM', '2GXG', '2GZQ', '2HQK', '2HYK', '2I24', '2I49', '2IBL', '2IGD', '2IMF', '2IP6', '2IVY', '2J32', '2J9W', '2JKU', '2JLI', '2JLJ', '2MCM', '2NLS', '2NR7', '2NUH', '2O6X', '2OA2', '2OCT', '2OHW', '2OKT', '2OL9', '2OLX', '2PKT', '2PLT', '2PMR', '2POF', '2PPN', '2PSF', '2PTH', '2Q4N', '2Q52', '2QJL',
#	'2R16', '2R6Q', '2RB8', '2RE2', '2RFR', '2V9V', '2VE8', '2VH7', '2VIM', '2VPA', '2VQ4', '2VY8', '2VYO', '2W1V', '2W2A', '2W6A', '2WJ5', '2WUJ', '2WW7', '2WWE', '2X1Q', '2X25', '2X3M', '2X5Y', '2X9Z', '2XHF', '2Y0T', '2Y72', '2Y7L', '2Y9F', '2YLB', '2YNY', '2ZCM', '2ZU1',  '3A0M', '3A7L', '3AMC', '3AUB', '3B5O', '3BA1',
#	'3BED', '3BQX', '3BZQ', '3BZZ', '3DRF', '3DWV', '3E5T', '3E7R', '3EUR', '3F2Z', '3F7E', '3FCN', '3FE7', '3FKE', '3FMY', '3FOD', '3FSO', '3FTD', '3FVA', '3G1S', '3GBW', '3GHJ', '3HFO', '3HHP', '3HNY', '3HP4', '3HWU', '3HYD', '3HZ8', '3I2V', '3I2Z', '3I4O', '3I7M', '3IHS', '3IVV', '3K6Y', '3KBE', '3KGK', '3KZD', '3L41',
#	'3LAA', '3LAX', '3LG3', '3LJI', '3M3P', '3M8J', '3M9J', '3M9Q', '3MAB', '3MD4', '3MD5', '3MEA', '3MGN', '3MRE', '3N11', '3NE0', '3NGG', '3NPV', '3NVG', '3NZL', '3O0P', '3O5P', '3OBQ', '3OQY', '3P6J', '3PD7', '3PES', '3PID', '3PIW', '3PKV', '3PSM', '3PTL', '3PVE', '3PZ9', '3PZZ', '3Q2X', '3Q6L', '3QDS', '3QPA', '3R6D',
#	'3R87', '3RQ9', '3RY0', '3RZY', '3S0A', '3SD2', '3SEB', '3SED', '3SO6', '3SR3', '3SUK', '3SZH', '3T0H', '3T3K', '3T47', '3TDN', '3TOW', '3TUA', '3TYS', '3U6G', '3U97', '3UCI', '3UR8', '3US6', '3V1A', '3V75', '3VN0', '3VOR', '3VUB', '3VVV', '3VZ9', '3W4Q', '3ZBD', '3ZIT', '3ZRX', '3ZSL', '3ZZP', '3ZZY', '4A02', '4ACJ', '4AE7', '4AM1', '4ANN', '4AVR', '4AXY', '4B6G', '4B9G', '4DD5', '4DKN', '4DND', '4DPZ', '4DQ7', '4DT4', '4EK3', '4ERY', '4ES1', '4EUG', '4F01', '4F3J', '4FR9', '4G14', '4G2E', '4G5X', '4G6C', '4G7X', '4GA2', '4GMQ', '4GS3', '4H4J', '4H89', '4HDE', '4HJP', '4HWM', '4IL7', '4J11', '4J5O', '4J5Q', '4J78', '4JG2', '4JVU', '4JYP', '4KEF', '5CYT', '6RXN']

# 134+7  couple = np.arange(1,31,1) 1~30
# pdbID=[	'1YJO', '2OL9', '2OLX', '3FVA', '3HYD', '3NVG', '3Q2X',
# 	'1F8R', '1GCO', '1H6V', '1IDP', '1KMM', '1QKI', '1WLY', '2A50', '2AH1', '2BCM', '2COV', '2D5W', '2DPL', '2E10', '2ETX', '2FN9', '2I49', '2IMF', '2J32', '2J9W',
#   '2O6X', '2POF', '2PSF', '2Q52', '2RE2', '2VE8', '2VPA', '2VYO', '2W1V', '2W2A', '2X1Q', '2X9Z', '2XHF', '2Y7L', '2YLB', '2YNY', '2ZCM', '2ZU1', '3AMC', '3B5O',
#   '3BA1', '3DRF', '3DWV', '3FTD', '3G1S', '3HHP', '3K6Y', '3L41', '3LG3', '3LJI', '3M3P', '3MGN', '3MRE', '3N11', '3NE0', '3NPV', '3OQY', '3PID', '3PKV', '3PTL',
# 	'3PVE', '3PZ9', '3QDS', '3R6D', '3SEB', '3SR3', '3SUK', '3SZH', '3T0H', '3TDN', '3TUA', '3U6G', '3UR8', '3US6', '3V1A', '3V75', '3VN0', '3VOR', '3VUB', '3VVV',
# 	'3VZ9', '3W4Q', '3ZBD', '3ZIT', '3ZRX', '3ZSL', '3ZZP', '3ZZY', '4A02', '4ACJ', '4AE7', '4AM1', '4ANN', '4AVR', '4AXY', '4B6G', '4B9G', '4DD5', '4DKN', '4DND',
# 	'4DPZ', '4DQ7', '4DT4', '4EK3', '4ERY', '4ES1', '4EUG', '4F01', '4F3J', '4FR9', '4G14', '4G2E', '4G5X', '4G6C', '4G7X', '4GA2', '4GMQ', '4GS3', '4H4J', '4H89',
# 	'4HDE', '4HJP', '4HWM', '4IL7', '4J11', '4J5O', '4J5Q', '4J78', '4JG2', '4JVU', '4JYP', '4KEF', '5CYT', '6RXN']

# pdbID = ['1F8R','1GCO', '1H6V','1KMM', '1QKI','2AH1','2D5W','3HHP']
'''
# 223 couple = 0~30/0~20；
pdbID=['1ABA', '1AHO', '1AIE', '1AKG', '1ATG', '1BGF', '1BX7', '1BYI', '1CCR', '1CYO', '1DF4', '1E5K', '1ES5', '1ETL', '1ETM', '1ETN', '1EW4', '1FF4', '1FK5', '1GK7',
       '1GVD', '1GXU', '1HJE', '1I71', '1IFR', '1K8U', '1KNG', '1KR4', '1KYC', '1LR7', '1MF7', '1N7E', '1NKD', '1NKO', '1NLS', '1NNX', 
# couple = np.arange(0,21,1) 
       '1NOA', '1NOT', '1O06', '1O08', '1OB4', '1OB7', '1OPD', '1P9I', '1PEF', '1PEN', '1PMY', '1PZ4', '1Q9B', '1QAU', '1QTO', '1R29', '1R7J', '1RJU', '1RRO', '1SAU', '1TGR', '1TZV', '1U06', '1U7I', 
       '1U9C', '1UHA', '1UKU', '1ULR', '1UOY', '1USE', '1USM', '1UTG', '1V05', '1V70', '1VRZ', '1W2L', '1WBE', '1WHI', '1WPA', '1X3O', '1XY1', '1XY2', '1Y6X', '1YZM', 
       '1Z21', '1ZCE', '1ZVA', '2AGK', '2B0A', '2BF9', '2BRF', '2C71', '2CE0', '2CG7', '2CWS', '2DKO', '2DSX', '2E3H', '2EAQ', '2EHP', '2EHS', '2ERW', '2FB6', '2FG1', 
       '2FQ3', '2G69', '2G7O', '2G7S', '2GKG', '2GOM', '2GXG', '2GZQ', '2HQK', '2HYK', '2I24', '2IBL', '2IGD', '2IP6', '2IVY', '2JKU', '2JLI', '2JLJ', '2MCM', '2NLS',
       '2NR7', '2NUH', '2OA2', '2OCT', '2OHW', '2OKT', '2PKT', '2PLT', '2PMR', '2PPN', '2PTH', '2Q4N', '2QJL', '2R16', '2R6Q', '2RB8', '2RFR', '2V9V', '2VH7', '2VIM',
       '2VQ4', '2VY8', '2W6A', '2WJ5', '2WUJ', '2WW7', '2WWE', '2X25', '2X3M', '2X5Y', '2Y0T', '2Y72', '2Y9F', '3A0M', '3A7L', '3AUB', '3BED', '3BQX', '3BZQ', '3BZZ', 
       '3E5T', '3E7R', '3EUR', '3F2Z', '3F7E', '3FCN', '3FE7', '3FKE', '3FMY', '3FOD', '3FSO', '3GBW', '3GHJ', '3HFO', '3HNY', '3HP4', '3HWU', '3HZ8', '3I2V', '3I2Z', 
       '3I4O', '3I7M', '3IHS', '3IVV', '3KBE', '3KGK', '3KZD', '3LAA', '3LAX', '3M8J', '3M9J', '3M9Q', '3MAB', '3MD4', '3MD5', '3MEA', '3NGG', '3NZL', '3O0P', '3O5P',
       '3OBQ', '3P6J', '3PD7', '3PES', '3PIW', '3PSM', '3PZZ', '3Q6L', '3QPA', '3R87', '3RQ9', '3RY0', '3RZY', '3S0A', '3SD2', '3SED', '3SO6', '3T3K', '3T47', '3TOW', 
       '3TYS', '3U97', '3UCI']
'''
pdbID = ['1NOA', '1NOT', '1O06', '1O08', '1OB4', '1OB7', '1OPD', '1P9I', '1PEF', '1PEN', '1PMY', '1PZ4', '1Q9B', '1QAU', '1QTO', '1R29', '1R7J', '1RJU', '1RRO', '1SAU', '1TGR', '1TZV', '1U06', '1U7I', 
       '1U9C', '1UHA', '1UKU', '1ULR', '1UOY', '1USE', '1USM', '1UTG', '1V05', '1V70', '1VRZ', '1W2L', '1WBE', '1WHI', '1WPA', '1X3O', '1XY1', '1XY2', '1Y6X', '1YZM', 
       '1Z21', '1ZCE', '1ZVA', '2AGK', '2B0A', '2BF9', '2BRF', '2C71', '2CE0', '2CG7', '2CWS', '2DKO', '2DSX', '2E3H', '2EAQ', '2EHP', '2EHS', '2ERW', '2FB6', '2FG1', 
       '2FQ3', '2G69', '2G7O', '2G7S', '2GKG', '2GOM', '2GXG', '2GZQ', '2HQK', '2HYK', '2I24', '2IBL', '2IGD', '2IP6', '2IVY', '2JKU', '2JLI', '2JLJ', '2MCM', '2NLS',
       '2NR7', '2NUH', '2OA2', '2OCT', '2OHW', '2OKT', '2PKT', '2PLT', '2PMR', '2PPN', '2PTH', '2Q4N', '2QJL', '2R16', '2R6Q', '2RB8', '2RFR', '2V9V', '2VH7', '2VIM',
       '2VQ4', '2VY8', '2W6A', '2WJ5', '2WUJ', '2WW7', '2WWE', '2X25', '2X3M', '2X5Y', '2Y0T', '2Y72', '2Y9F', '3A0M', '3A7L', '3AUB', '3BED', '3BQX', '3BZQ', '3BZZ', 
       '3E5T', '3E7R', '3EUR', '3F2Z', '3F7E', '3FCN', '3FE7', '3FKE', '3FMY', '3FOD', '3FSO', '3GBW', '3GHJ', '3HFO', '3HNY', '3HP4', '3HWU', '3HZ8', '3I2V', '3I2Z', 
       '3I4O', '3I7M', '3IHS', '3IVV', '3KBE', '3KGK', '3KZD', '3LAA', '3LAX', '3M8J', '3M9J', '3M9Q', '3MAB', '3MD4', '3MD5', '3MEA', '3NGG', '3NZL', '3O0P', '3O5P',
       '3OBQ', '3P6J', '3PD7', '3PES', '3PIW', '3PSM', '3PZZ', '3Q6L', '3QPA', '3R87', '3RQ9', '3RY0', '3RZY', '3S0A', '3SD2', '3SED', '3SO6', '3T3K', '3T47', '3TOW', 
       '3TYS', '3U97', '3UCI']

couple = np.arange(0,21,1)
# couple = np.arange(1,31,1)  #[ 0  1  2  3  4  5  6  7  8  9 10]
print('couple:',couple)
# print(len(couple))
# exit()
atom_num = []
max_distance = []
for i in range (len(pdbID)):
    atomnum = len(open(r'/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID[i], 'r').readlines())
    #atomnum = len(open(r'D:/pycharm/pythonProject/protein/364_xyz_Bfactor/364_xyzb_5.1/%s_ca.xyzb' % pdbID[i], 'U').readlines())
    atom_num.append(atomnum)
    #pdbfile = open('D:/pycharm/pythonProject/protein/364_xyz_Bfactor/364_xyzb_5.1/%s_ca.xyzb' % pdbID[i], 'r')
    pdbfile = open('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID[i], 'r')
    point = []  # 记录xyz
    for line in pdbfile.readlines():
        point.append([float(line.split()[0]), float(line.split()[1]), float((line.split()[2]))])
    matrix = squareform(pdist(point))  # 距离矩阵
    #print(matrix)
    distance = dist.squareform(matrix)  # 再将这个距离矩阵转换为一个一维数组
    maxdistance = np.max(distance)  # 求出数组中的最大值，即该蛋白质最大距离
    max_distance.append(maxdistance)
# print(atom_num, len(atom_num))

#for i in range(1):
for i in range (len(pdbID)):
    pdbID_value = pdbID[i]
    eta = 3  # 3 or 6
    kappa = 1  # 1 or 2
    n = atom_num[i]
    print(pdbID[i])
    #pdbfile = open('D:/pycharm/pythonProject/protein/364_xyz_Bfactor/364_xyzb_5.1/%s_ca.xyzb' % pdbID[i], 'r')
    pdbfile = open('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/364_xyzb/%s_ca.xyzb' % pdbID[i], 'r')
    point = []
    for line in pdbfile.readlines():
        point.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
    distance_matrix_original = squareform(pdist(point))
    distance_matrix_original = np.around(distance_matrix_original)

    #################### connectivity matrix ################
    connmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                connmat[i][j] = 1 - np.exp(-(distance_matrix_original[i][j]/eta)**kappa)
    # print(np.unique(dist.squareform(np.around(connmat, 2))))  # [0.  0.1 0.2 0.3 0.4 0.5 0.6]
    # cutoff = np.unique(dist.squareform(np.around(connmat,2)))

    ################## this is Persistent Laplacian filtration process ##############
    # the max of connmat_new_origin is 0.63, the min is 0, set the interval 0.63/5=0.126 for example
    connmat_max = np.unique(dist.squareform(np.around(connmat, 2)))[-1]
    connmat_min = np.unique(dist.squareform(np.around(connmat, 2)))[0]
    #print('connmat_max:', connmat_max, 'connmat_min:', connmat_min)
    connmat_new_origin = np.around(connmat, 2)
    # cutoff = list(map(lambda x: 1-np.exp(-(x/eta)**2), [0, 12.5, 25.5, 36.5]))
    # print(list(map(lambda x: 1-np.exp(-(x/eta)**2), [0, 12.5, 25.5, 36.5])))
    # [0.0, 0.02668897560993555, 0.10647226908648977, 0.20598482103315574]
    bin_num = 10  # this number is changable.
    cutoff = [0] + [connmat_min + (x + 1) * (connmat_max - connmat_min) / bin_num for x in range(bin_num)]
    print('cutoff:', cutoff)
    # exit()
    #==============================================
    for j in range(len(couple)):
    # for j in range(1):
        couple_value = couple[j]
        for k in range(len(cutoff)):
        # for k in range(1):
            cutoff_value = cutoff[k]

            f = open('%s_%d_%.2f_3_1.pbs' % (pdbID_value, couple_value, cutoff_value), 'w')
            f.write('#!/bin/bash\n')
            f.write('########## Define Resources Needed with SBATCH Lines ##########\n')
            f.write('#SBATCH --nodes=1  \n')
            f.write('#SBATCH --time=00:05:00             # limit of wall clock time - how long the job will run (same as -t)\n')
            f.write('#SBATCH --ntasks=5                  # number of tasks - how many tasks (nodes) that you require (same as -n)\n')
            f.write('#SBATCH --cpus-per-task=2           # number of CPUs (or cores) per task (same as -c)\n')
            f.write('#SBATCH --mem=8G                    # memory required per node - amount of memory (in bytes)\n')
            #file_path = '/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/2_134_3_1_out' #指定err\out保存位置，避免一个文件夹里存储太多
            f.write('#SBATCH --output=/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/pcc_out_err/%s_%d_%.2f_3_1.out\n' % (pdbID_value, couple_value, cutoff_value))
            f.write('#SBATCH --error=/mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/pcc_out_err/%s_%d_%.2f_3_1.err\n' % (pdbID_value, couple_value, cutoff_value))

            f.write('cd /mnt/ufs18/home-192/jiangj33/KeLu/desktop/B-factor/PCC_223')  # 切换到作业所在目录
            f.write('\n')

            f.write('python PCCs_223.py %s %d %.2f' % (pdbID_value, couple_value, cutoff_value))
            f.close()
            cmd = 'sbatch %s_%d_%.2f_3_1.pbs' % (pdbID_value, couple_value, cutoff_value)
            os.system(cmd)




