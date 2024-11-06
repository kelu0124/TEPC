import numpy as np
import matplotlib.pyplot as plt
import string

plt.rcParams["font.family"] = "Times New Roman"
interval = [0., 39., 166.]  # for polygon16
pdbid = 'polygon16'
couple = 10
point_polygon16 = [(0.00, 100.00), (-38.27, 92.39), (-70.71, 70.71), (-92.39, 38.27), (-100.00, 0.00), (-92.39, -38.27), (-70.71, -70.71), (-38.27, -92.39), (0.00, -100.00), (38.27, -92.39), (70.71, -70.71), (92.39, -38.27), (100.00, 0.00), (92.39, 38.27), (70.71, 70.71), (38.27, 92.39)]
x = [i[0] for i in point_polygon16]
y = [i[1] for i in point_polygon16]

plt.figure(figsize=(22, 22))
for ii in range(len(interval)):
    singlemat = np.load(r'C:\Users\administered\Desktop\图3图4\特征\singlec03s10_%s_cutoff_%.1f.npy' % (pdbid, interval[ii]))
    amplitudemat = np.load(r'C:\Users\administered\Desktop\图3图4\特征\amplitudezc03s10_%s_cutoff_%.1f.npy' % (pdbid, interval[ii]))
    couplemat = np.load(r'C:\Users\administered\Desktop\图3图4\特征\couplematrix_%s_cutoff_%.1f.npy' % (pdbid, interval[ii]))
    singlemat = np.array(singlemat)
    amplitudemat = np.array(amplitudemat)
    couplemat = np.array(couplemat)

    # 第一列的3个图
    ax1 = plt.subplot(3, 3, ii * 3 + 1)
    sc1 = ax1.matshow(couplemat, cmap='GnBu_r', aspect="auto")
    a = plt.colorbar(sc1)
    a.ax.tick_params(labelsize=28)
    plt.xlabel('Node number', fontsize=32, labelpad=20)
    plt.ylabel('Node number', fontsize=32, labelpad=5)
    plt.tick_params(labelsize=28)

    # 第二列的3个图
    ax2 = plt.subplot(3, 3, ii * 3 + 2)
    sc2 = ax2.matshow(amplitudemat, cmap='rainbow', aspect="auto")
    y = plt.colorbar(sc2)
    y.ax.tick_params(labelsize=28)
    plt.xlabel('Node number', fontsize=32)
    plt.ylabel('Time', fontsize=32)
    plt.tick_params(labelsize=28)
    ax2 = plt.gca()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.invert_yaxis()

    # 第三列的3个图
    ax3 = plt.subplot(3, 3, ii * 3 + 3)
    sc3 = plt.plot(singlemat[:, 0], singlemat[:, 2], 'o-', color='#6ADC88')
    plt.xlabel('X', fontsize=32)
    plt.ylabel('Z', fontsize=32)
    plt.tick_params(labelsize=28)

plt.tight_layout()
plt.savefig('%s_couplestrength_%d.png' % (pdbid, couple), dpi=1100, bbox_inches='tight')
#plt.savefig('%s_couplestrength_%d.pdf' % (pdbid, couple), dpi=650, bbox_inches='tight')
#plt.savefig('%s_couplestrength_%d.svg' % (pdbid, couple), dpi=1200, bbox_inches='tight')

#plt.show()
