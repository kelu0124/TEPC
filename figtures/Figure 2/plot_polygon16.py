import numpy as np 
import random
import matplotlib.pyplot as plt
import string

plt.rcParams["font.family"] = "Times New Roman"

# interval =  np.arange(40,60,0.5)
# interval_0 =  np.arange(4,40,0.5) # 5-30,step is 2
# interval = np.insert(interval_0, 0, [0]) # put 0 in the first position
# interval = [0., 42. , 81., 118., 149., 173., 190. ,199.] # for polygon15
# interval = [ 0.,39. , 77. ,111. ,141., 166., 185., 196., 200.] # for polygon16
interval = [ 0. , 39., 166.] # for polygon16
print(interval)
# pdbid = '1a5r'
# pdbid = '2rap'
# pdbid = 'polygon15'
# pdbid = '2rvq'
pdbid = 'polygon16'
couple = 10
point_polygon16 = [(0.00,100.00), (-38.27,92.39), (-70.71,70.71), (-92.39,38.27), (-100.00,0.00), (-92.39,-38.27), (-70.71,-70.71), (-38.27,-92.39), (0.00,-100.00), (38.27,-92.39), (70.71,-70.71), (92.39,-38.27), (100.00,0.00), (92.39,38.27), (70.71,70.71), (38.27,92.39)]
x = [ i[0] for i in point_polygon16]
y = [ i[1] for i in point_polygon16]
print(len(x))

# plt.scatter(x, y, color='k',marker='^',s=25)
# for i in range(len(x)):
#     for j in range(len(y)):
#         plt.plot([x[i], x[j]], [y[i], y[j]], color='black')
# # plt.show()
# # plt.savefig('picture/%s/%s.pdf'%(pdbid,pdbid),bbox_inches = 'tight')
# exit()



plt.figure(figsize=(22, 22))
for ii in range(len(interval)):
# for ii in [1,5]:
# for ii in range(0,1):
    print('cutoff:',interval[ii])

    singlemat = np.load(r'C:\Users\administered\Desktop\图3图4\特征\singlec03s10_%s_cutoff_%.1f.npy'%(pdbid,interval[ii]))
    amplitudemat = np.load(r'C:\Users\administered\Desktop\图3图4\特征\amplitudezc03s10_%s_cutoff_%.1f.npy'%(pdbid,interval[ii]))
    couplemat = np.load(r'C:\Users\administered\Desktop\图3图4\特征\couplematrix_%s_cutoff_%.1f.npy'%(pdbid,interval[ii]))

    # singlemat = np.load('results/%s/singlec03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(pdbid,pdbid,interval[ii],couple))
    # amplitudemat = np.load('results/%s/amplitudezc03s10_%s_cutoff_%.2f_couple_%.2f.npy'%(pdbid,pdbid,interval[ii],couple))
    # couplemat = np.load('results/%s/couplematrix_%s_cutoff_%.2f_couple_%.2f.npy'%(pdbid,pdbid,interval[ii],couple))

    # singlemat = np.load('results/%s/singlec03s10_%s_cutoff_%.1f_cs_%d.npy'%(pdbid,pdbid,interval[ii],couple))
    # amplitudemat = np.load('results/%s/amplitudezc03s10_%s_cutoff_%.1f_cs_%d.npy'%(pdbid,pdbid,interval[ii],couple))
    # couplemat = np.load('results/%s/couplematrix_%s_cutoff_%.1f_cs_%d.npy'%(pdbid,pdbid,interval[ii],couple))

    singlemat = np.array(singlemat)
    amplitudemat = np.array(amplitudemat)
    couplemat = np.array(couplemat)

    ax1 = plt.subplot(3,3,ii*3+1)
    sc1 = ax1.matshow(couplemat,cmap='rainbow', aspect="auto")
    a=plt.colorbar(sc1)
    a.ax.tick_params(labelsize=20) #设置颜色条的刻度标签的字体大小为16
    plt.xlabel('Node number', fontsize=26, labelpad=20)
    plt.ylabel('Node number', fontsize=26, labelpad=5)
    plt.tick_params(labelsize=20)
    ax1.text(-0.13, 0.98, string.ascii_lowercase[ii*3], transform=ax1.transAxes,fontproperties='Times New Roman',
            size=32, weight='bold')

    ax2 = plt.subplot(3,3,ii*3+2)
    sc2 = ax2.matshow(amplitudemat,cmap='rainbow', aspect="auto")
    y = plt.colorbar(sc2)
    y.ax.tick_params(labelsize=20)
    plt.xlabel('Node number', fontsize=26)
    plt.ylabel('Time', fontsize=26)
    plt.tick_params(labelsize=20)
    ax2 = plt.gca()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.invert_yaxis()
    ax2.text(-0.15, 0.98, string.ascii_lowercase[ii*3+1], transform=ax2.transAxes,fontproperties='Times New Roman',
            size=32, weight='bold')

    ax3 = plt.subplot(3,3,ii*3+3)
    sc3 = plt.plot(singlemat[:,0],singlemat[:,2],'ro-')
    plt.xlabel('X', fontsize=24)
    plt.ylabel('Z',fontsize=24)
    plt.tick_params(labelsize=20)
    ax3.text(-0.13, 0.98, string.ascii_lowercase[ii*3+2], transform=ax3.transAxes,fontproperties='Times New Roman',
            size=32, weight='bold')

plt.tight_layout()  # 自动调整子图布局，使之更紧凑
# plt.tight_layout(h_pad=0.01)  # 增加子图之间的垂直距离
# plt.subplots_adjust(top=0.9, bottom=0.1)  # 微调子图与顶部的距离，避免与坐标轴标签重叠
plt.savefig('%s_couplestrength_%d.png'%(pdbid,couple),dpi=1200,bbox_inches = 'tight')
plt.savefig('%s_couplestrength_%d.svg'%(pdbid,couple),dpi=1200,bbox_inches = 'tight')
plt.show()


