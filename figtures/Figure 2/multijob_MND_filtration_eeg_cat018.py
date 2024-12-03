#!/usr/bin/python
#!/bin/bash
import numpy as np
from numpy import arange
import os
import math
import numpy
import pandas as pd

subjectID = 'atom'
#eegfile = np.loadtxt(r'C:\Users\administered\Desktop\原子坐标\atom_coordinates.txt')
eegfile = np.loadtxt(r'/public/home/chenlong666/desktop/my_desk1/原子坐标/atom_coordinates.txt')

CorrMat = np.corrcoef(eegfile,rowvar=1)  # rowvar=1 对行进行分析
CorrMat_new = pd.DataFrame(CorrMat.round(3))
CorrMat_new[CorrMat_new < 0] = 0               #  将负相关的地方令值为0
CorrMat_new[np.eye(120,dtype=np.bool)] = 0      # 令对角元素全为0，128为matrix的维数,也是时间序列的个数
CorrMat_new_max = max(CorrMat_new.max())
CorrMat_new_min = min(CorrMat_new.min())
bin_num = 5
cutoff = [CorrMat_new_min + (x+1) * (CorrMat_new_max - CorrMat_new_min)/bin_num for x in range(bin_num)]
# cutoff = [0] + [CorrMat_new_min + (x + 1) * (CorrMat_new_max - CorrMat_new_min) / bin_num for x in range(bin_num)]
print('cutoff:',cutoff)
print(len(cutoff))

couple = np.arange(0.1,1.1,0.2)
print('couple:',couple)
print(len(couple))



for j in range(len(couple)):
# for j in range(1):
    for i in range(len(cutoff)):
    # # for i in range(1):
    #     f = open('%.2f_%.3f_%s_MND_filtration.job'%(couple[j],cutoff[i],subjectID), 'w')
    #     f.write('#!/bin/bash\n')
    #     f.write('########## Define Resources Needed with SBATCH Lines ##########\n')
    #     f.write('#SBATCH --nodes=1  \n')
    #     f.write('#SBATCH --time=2:00:00             # limit of wall clock time - how long the job will run (same as -t)\n')
    #     f.write('#SBATCH --ntasks=5                  # number of tasks - how many tasks (nodes) that you require (same as -n)\n')
    #     f.write('#SBATCH --cpus-per-task=2           # number of CPUs (or cores) per task (same as -c)\n')
    #     f.write('#SBATCH --mem=8G                    # memory required per node - amount of memory (in bytes)\n')
    #     f.write('#SBATCH --output=%x.out\n')
    #     f.write('#SBATCH --error=%x.error.out\n')
    #
    #     f.write('python MND_filtration_eeg_cat018.py %f %f'%(cutoff[i],couple[j]))
    #     f.close()
    #     cmd = 'sbatch %.2f_%.3f_%s_MND_filtration.job'%(couple[j],cutoff[i],subjectID)
    #     os.system(cmd)
        with open('%.2f_%.2f_%s_MND_filtration.pbs' % (couple[j], cutoff[i], subjectID), 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('#PBS -l nodes=node05:ppn=1\n')
            f.write('#PBS -l walltime=2:00:00\n')
            f.write('#PBS -o %s_MND_filtration.out\n' % (subjectID))
            f.write('#PBS -e %s_MND_filtration.err\n' % (subjectID))
            f.write('\n')
            f.write('cd /public/home/chenlong666/desktop/my_desk1/原子坐标\n')
            f.write('\n')
            f.write('python MND_filtration_eeg_cat018.py %.2f %.2f\n' % (cutoff[i],couple[j]))

        # Submit the PBS script
        os.system('qsub %.2f_%.2f_%s_MND_filtration.pbs' % (couple[j],cutoff[i], subjectID))