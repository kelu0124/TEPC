#!/usr/bin/python
# !/bin/bash
import numpy as np
from numpy import arange
import os
import math
import numpy
import pandas as pd

eegfile = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/data/h_average_corr_matrix.npy')
# eegfile = np.load('/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/data/s_average_corr_matrix.npy')
# eegfile = np.load('h_average_corr_matrix.npy')
CorrMat = eegfile  # eegfile已将所有小于0的相关系数和对角元素置为0
print("Original CorrMat:\n", CorrMat)
# CorrMat_new = pd.DataFrame(CorrMat.round(3))
# CorrMat_new[CorrMat_new < 0] = 0
# np.fill_diagonal(CorrMat_new.values, 0)  # Set diagonal elements to 0

# Calculate max and min values for normalization
CorrMat_new_max = CorrMat.max().max()
CorrMat_new_min = CorrMat.min().min()
print('CorrMat_new_max:', CorrMat_new_max)
print('CorrMat_new_min:', CorrMat_new_min)
CorrMat_normalized = (CorrMat - CorrMat_new_min) / (CorrMat_new_max - CorrMat_new_min)
# print("Normalized CorrMat:\n", CorrMat_normalized)

cutoff = np.arange(0.1, 1.1, 0.1)
print('cutoff:', cutoff)
print(len(cutoff))

# couple = np.arange(12,30.1,2)
# couple = np.arange(1.1,10.1,0.1)
couple = np.arange(0, 1.1, 0.2)
print('couple:', couple)
print(len(couple))
# exit()

for i in range(len(cutoff)):
    # for i in range(3,4):
    # i=3
    for j in range(len(couple)):
        # for j in range(1):

        f = open('%.3f_%.3f_MND_filtration.job' % (couple[j], cutoff[i]), 'w')
        f.write('#!/bin/bash\n')
        f.write('########## Define Resources Needed with SBATCH Lines ##########\n')
        f.write('#SBATCH --nodes=1  \n')
        f.write('#SBATCH --time=00:05:00             # limit of wall clock time - how long the job will run (same as -t)\n')
        f.write('#SBATCH --ntasks=5                  # number of tasks - how many tasks (nodes) that you require (same as -n)\n')
        f.write('#SBATCH --cpus-per-task=2           # number of CPUs (or cores) per task (same as -c)\n')
        f.write('#SBATCH --mem=8G                    # memory required per node - amount of memory (in bytes)\n')
        f.write('#SBATCH --output=/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/out/h_%.3f_%.3f.out\n' % (couple[j], cutoff[i]))
        # f.write('#SBATCH --output=/mnt/ufs18/home-192/jiangj33/KeLu/desktop/eeg/out/s_%.3f_%.3f.out\n' % (couple[j], cutoff[i]))
        f.write('#SBATCH --error=h_%.3f_%.3f.err\n' % (couple[j], cutoff[i]))

        f.write('python slurm_feature_eeg_mat_guiyi.py %f %f' % (couple[j], cutoff[i]))
        f.close()
        cmd = 'sbatch %.3f_%.3f_MND_filtration.job' % (couple[j], cutoff[i])
        os.system(cmd)





