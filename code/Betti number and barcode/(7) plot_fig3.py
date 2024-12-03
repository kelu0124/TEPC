import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import gudhi
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from PIL import Image

def plot_connective_matrix(xpos,ypos,path):
    couplemat = np.load(path)
    couplemat = np.array(couplemat)
    ax = axs[xpos, ypos]  # 选择子图
    cax = ax.matshow(couplemat, cmap='viridis', aspect="auto")
    ax.set_xlabel('electrode number', fontsize=12, labelpad=6)
    ax.set_ylabel('electrode number', fontsize=12, labelpad=-2)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10,pad = 0)

    # 添加颜色条
    cbar = fig.colorbar(cax, ax=ax, location='right', shrink=1)  # shrink 参数可以调整颜色条的大小
    vmin, vmax = cax.get_clim()
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=8)

def plot_rainbow(xpos,ypos,path):
    amplitudemat = np.load(path)
    amplitudemat = np.array(amplitudemat)
    ax = axs[xpos, ypos]  # 选择子图
    cax = ax.matshow(amplitudemat, cmap='rainbow', aspect="auto")
    ax.set_xlabel('electrode number', fontsize=12,labelpad=-1)
    ax.set_ylabel('Time', fontsize=12,labelpad=-2)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_ticks_position('bottom')  # x 轴刻度线的位置设置在图形的底部
    ax.invert_yaxis()
    vmin, vmax = float(amplitudemat.min()), float(amplitudemat.max())
    ticks = [i for i in range(int(vmin), math.ceil(vmax) + 1) if i % 10 == 0]
    fig = ax.get_figure()  # 如果 ax 已经是一个子图对象，可以通过这种方式获取 Figure 对象
    colorbar = fig.colorbar(cax, ax=ax, location='right', shrink=1)  # 假设颜色条在图的右侧
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=8)

def plot_butterfly(xpos,ypos,path):
    singlemat = np.load(path)
    singlemat = np.array(singlemat)
    ax = axs[xpos, ypos]  # 选择子图
    ax.plot(singlemat[:, 0], singlemat[:, 2], color='#6ADC88', marker='o', linestyle='-',markersize = 1)
    ax.set_xlabel('X', fontsize=10,labelpad=-2)
    ax.set_ylabel('Z', fontsize=10,labelpad=-1)
    ax.tick_params(labelsize=10)

def plot_Betti_time(xpos,ypos,path):
    ax = axs[xpos, ypos]  # 选择子图
    betti_num_time = np.load(path)
    if xpos != 1:
        ax.plot(betti_num_time[:, 1], betti_num_time[:, 0], color=(0.9, 0.2, 0.2), label='Betti 0',linewidth = 0.8)  # 交换 x 轴和 y 轴的绘制顺序
        ax.plot(betti_num_time[:, 2], betti_num_time[:, 0], color=(0.1, 0.5, 0.8), label='Betti 1',linewidth = 0.8)
        ax.plot(betti_num_time[:, 3], betti_num_time[:, 0], 'g', label='Betti 2',linewidth = 0.8)
        ax.set_xlim(-2, 21)
    else:
        ax.plot(betti_num_time[:, 1], betti_num_time[:, 0], color=(0.9, 0.2, 0.2), label='Betti 0',linewidth=0.8)  # 交换 x 轴和 y 轴的绘制顺序
        ax.plot(betti_num_time[:, 2]*3, betti_num_time[:, 0], color=(0.1, 0.5, 0.8), label='Betti 1', linewidth=0.8)
        ax.plot(betti_num_time[:, 3]*3, betti_num_time[:, 0], 'g', label='Betti 2', linewidth=0.8)
        ax.set_xlim(-2, 21)
        ax.set_xticks(np.linspace(0, 20, 7))
        ax.set_xticklabels(['0','1','2','3','4','5','20'])

    ax.set_ylim(0, 1000)
    # 设置刻度标签的字体大小
    ax.tick_params(labelsize=8)
    ax.set_xlabel('betti number', fontsize=12,labelpad=-1)
    ax.set_ylabel('Time', fontsize=12,labelpad=-4)
    if xpos == 0:
        ax.legend(fontsize=8, loc='upper center')

def plot_Betti_filtration_radius(xpos, ypos, path,subjectID,cutoff):
    amplitudemat = np.load(path)
    amplitudemat = np.array(amplitudemat)
    n = 19
    CorrMat_new = pd.DataFrame(np.load(r'D:\python\result\EEG\npy\couple_npy/%s_average_corr_matrix.npy'%subjectID[0]))
    mat = np.zeros((n, n))
    ax = axs[xpos, ypos]  # 选择子图

    # 隐藏原始子图
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_frame_on(False)

    # 上部InsetAxes
    inset_ax1 = inset_axes(ax,
                           width="140%", height="240%",  # 宽度为50%，高度为20%
                           loc='upper center',  # 位于上部中心
                           bbox_to_anchor=(0.15, 0.85, 0.7, 0.15),  # 调整位置以适合你的需要
                           bbox_transform=ax.transAxes,
                           borderpad=0)

    # 中部InsetAxes
    inset_ax2 = inset_axes(ax,
                           width="140%", height="80%",  # 宽度为50%，高度为20%
                           loc='center',  # 位于中心
                           bbox_to_anchor=(0.15, 0.3, 0.7, 0.15),  # 调整位置以适合你的需要
                           bbox_transform=ax.transAxes,
                           borderpad=0)

    # 下部InsetAxes
    inset_ax3 = inset_axes(ax,
                           width="140%", height="60%",  # 宽度为50%，高度为20%
                           loc='lower center',  # 位于下部中心
                           bbox_to_anchor=(0.15, 0, 0.7, 0.15),  # 调整位置以适合你的需要
                           bbox_transform=ax.transAxes,
                           borderpad=0)
    for i in range(n):
        for j in range(n):
            if j != i:
                if CorrMat_new[i][j] < cutoff:
                    mat[i][j] = -1
                else:
                    mat[i][j] = 0
    rips_complex = gudhi.RipsComplex(points=mat, max_edge_length=6)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence(min_persistence=0)
    diag_dim_0 = [i for i in diag if i[0] == 0]
    diag_dim_1 = [i for i in diag if i[0] == 1]
    diag_dim_2 = [i for i in diag if i[0] == 2]

    diag_dim_0[0] = (0, (0, 6))
    diag_dim_0_sorted = sorted(diag_dim_0, key=lambda x: np.abs(x[1][0]))
    diag_dim_1_sorted = sorted(diag_dim_1, key=lambda x: np.abs(x[1][0]))
    diag_dim_2_sorted = sorted(diag_dim_2, key=lambda x: np.abs(x[1][0]))

    num = 0
    for point in diag_dim_0_sorted:
        num += 1
        inset_ax1.plot(point[1], [num] * 2, color=(0.9, 0.2, 0.2), linestyle='-', linewidth=1)
    inset_ax1.text(5.7, -1.1, '>', color=(0.9, 0.2, 0.2), fontsize=10, weight='bold')  # 调整箭头向右，向下，加粗
    inset_ax1.set_ylabel('$\\beta_0$', fontsize=10, labelpad=-3)
    # inset_ax1.yticks(fontsize=14)
    inset_ax1.axis([0, 6, -1, 20])
    inset_ax1.set_xticks(np.linspace(0, 6, 6),fontsize=0.8)
    inset_ax1.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])

    num = 0
    for point in diag_dim_1_sorted:
        num += 1
        inset_ax2.plot(point[1], [num] * 2, color=(0.1, 0.5, 0.8), linestyle='-', linewidth=1)
    inset_ax2.set_ylabel('$\\beta_1$', fontsize=10, labelpad=-1)
    # inset_ax2.yticks(fontsize=14)
    inset_ax2.axis([0, 6, 0, 3])
    inset_ax2.set_xticks(np.linspace(0, 6, 6),fontsize=0.8)
    inset_ax2.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    inset_ax2.set_yticks([0,3],fontsize = 0.8)
    inset_ax2.set_yticklabels([0,3])

    num = 0
    for point in diag_dim_2_sorted:
        num += 1
        inset_ax3.plot(point[1], [num] * 2, 'g-', linewidth=1)
    inset_ax3.set_xlabel('filtration radius', fontsize=12, labelpad=-1)
    inset_ax3.set_ylabel('$\\beta_2$', fontsize=10, labelpad=-1)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=14)
    inset_ax3.axis([0, 6, 0, 2])
    inset_ax3.set_xticks(np.linspace(0, 6, 6),fontsize=0.8)
    y_ticks = np.linspace(0, 2, 2)  # 根据num的值设置刻度数量
    inset_ax3.set_yticks(y_ticks)
    inset_ax3.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    # inset_ax3.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

def plot_phase_diagram_healthy(xpos,ypos):
    cutoff = [0.1, 0.141, 0.176,
              0.211, 0.246, 0.281, 0.316, 0.351, 0.387, 0.422,
              0.457, 0.492, 0.527, 0.562,
              0.598, 0.633, 0.668, 0.703,
              0.8, 0.9, 1.0]
    interval_thrs_1 = [4.4, 4.33, 4.2,
                       3.9, 3.4, 3, 2.78, 2.48, 2.38, 2.23,
                       2.13, 2.03, 1.93, 1.85,
                       1.79, 1.73, 1.7, 1.64,
                       1.58, 1.52, 1.47]
    interval_thrs_2 = [4.4, 4.33, 4.2,
                       4.13, 4.05, 3.95, 3.8, 3.5, 3.25, 3.05,
                       2.85, 2.65, 2.45, 2.40,
                       2.38, 2.34, 2.30, 2.25,
                       2.22, 2.18, 2.15]

    phase_1 = Image.open(r'D:\python\result\EEG\fig\fig3/healthy_chaos.png')
    phase_2 = Image.open(r'D:\python\result\EEG\fig\fig3/healthy_halfchaos.png')
    phase_3 = Image.open(r'D:\python\result\EEG\fig\fig3/healthy_stable.png')
    ax = axs[xpos,ypos]
    ax.set_ylabel('coupling strength', fontsize=12, labelpad=5)
    ax.set_xlabel('filtration radius', fontsize=12)
    # 设置绘图的坐标轴范围
    ax.axis([0.10, 1.0, 1, 5.05])
    # ax = plt.gca()
    # 设置纵坐标刻度间隔为0.1
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # 设置纵坐标刻度间隔为0.1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # 使用自定义的浅灰色来设置y轴位置为0.35和0.4的虚线并标记刻度
    custom_light_gray = (0.7, 0.7, 0.7)  # 自定义浅灰色的RGB值
    ax.axhline(1.47, color=custom_light_gray, linestyle='dashed')
    ax.axhline(2.15, color=custom_light_gray, linestyle='dashed')
    ax.axhline(4.2, color=custom_light_gray, linestyle='dashed')
    ax.axhline(4.4, color=custom_light_gray, linestyle='dashed')
    yticks = [1.47, 2.15, 4.2, 4.4]
    ytick_labels = ['1.47', '2.15', '4.2', '4.4']
    for i, tick_label in enumerate(ytick_labels):
        if tick_label in ['1.47', '2.15', '4.2', '4.4']:
            ax.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=8)
        else:
            ax.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=8)

    ax.imshow(phase_1, extent=(0.42, 0.59,3.15, 4.0))
    ax.imshow(phase_2, extent=(0.61, 0.78,3.15, 4.0))
    ax.imshow(phase_3, extent=(0.8, 0.97,3.15, 4.0))

    # ax = plt.gca()
    ax.set_aspect(0.2)  # 纵轴的单位长度是横轴单位长度的0.65倍
    ax.tick_params(axis='both', labelsize=8)  # 同时设置x轴和y轴刻度标签的字体大小

    # 横排
    ax.text(0.502, 3.5, 'I', fontsize=8)
    ax.text(0.676, 3.5, 'II', fontsize=8)
    ax.text(0.862, 3.5, 'III', fontsize=8)
    # 纵排
    ax.text(0.9, 1.12, 'I', fontsize=8)
    ax.text(0.89, 1.76, 'II', fontsize=8)
    ax.text(0.88, 2.4, 'III', fontsize=8)

    # ax.set_title('healthy', fontsize=12)

    ax.fill_between(cutoff, 0, interval_thrs_1, facecolor='green', alpha=0.2,
                    linewidth=0)  # 区域的下边界为0，上边界为interval_thrs_1
    ax.fill_between(cutoff, interval_thrs_1, interval_thrs_2, facecolor='red', alpha=0.2, edgecolor="k", linewidth=0)
    ax.fill_between(cutoff, interval_thrs_2, 5.05, edgecolor="k", alpha=0.2, linewidth=0.0)

def plot_phase_diagram_schizophrenia(xpos,ypos):
    cutoff = [0.1, 0.121, 0.161,
              0.201, 0.241, 0.282,
              0.322, 0.362, 0.403,
              0.443, 0.483, 0.523, 0.563,
              0.604, 0.644, 0.684, 0.725, 0.765, 0.805, 0.9, 1.0]
    interval_thrs_1 = [4.8, 4.75, 4.6,
                       3.95, 3.55, 3.2,
                       2.9, 2.65, 2.4,
                       2.2, 2, 1.85, 1.75,
                       1.7, 1.62, 1.57, 1.50, 1.48, 1.42, 1.33, 1.20]
    interval_thrs_2 = [4.8, 4.75, 4.6,
                       4.4, 4.15, 3.9,
                       3.65, 3.4, 3.2,
                       3.0, 2.85, 2.70, 2.60,
                       2.50, 2.45, 2.40, 2.38, 2.34, 2.30, 2.25, 2.21]
    phase_1 = Image.open(r'D:\python\result\EEG\fig\fig3/schizophrenia_chaos.png')
    phase_2 = Image.open(r'D:\python\result\EEG\fig\fig3/schizophrenia_halfchaos.png')
    phase_3 = Image.open(r'D:\python\result\EEG\fig\fig3/schizophrenia_stable.png')
    ax = axs[xpos,ypos]
    ax.set_ylabel('coupling strength', fontsize=12, labelpad=5)
    ax.set_xlabel('filtration radius', fontsize=12)

    # 设置绘图的坐标轴范围
    ax.axis([0.10, 1.0, 1, 5.2])
    # 设置纵坐标刻度间隔为0.1
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # 设置纵坐标刻度间隔为0.1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # 使用自定义的浅灰色来设置y轴位置为0.35和0.4的虚线并标记刻度
    custom_light_gray = (0.7, 0.7, 0.7)  # 自定义浅灰色的RGB值
    ax.axhline(1.19, color=custom_light_gray, linestyle='dashed')
    ax.axhline(2.2, color=custom_light_gray, linestyle='dashed')
    ax.axhline(4.6, color=custom_light_gray, linestyle='dashed')
    ax.axhline(4.8, color=custom_light_gray, linestyle='dashed')
    yticks = [1.19, 2.2, 4.6, 4.8]
    ytick_labels = ['1.19', '2.2', '4.6', '4.8']
    for i, tick_label in enumerate(ytick_labels):
        if tick_label in ['1.19', '2.2', '4.6', '4.8']:
            ax.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=8)
        else:
            ax.text(0.089, yticks[i], tick_label, ha='right', va='center', color='red', fontsize=8)

    ax.imshow(phase_1, extent=(0.42, 0.59, 3.55, 4.4))
    ax.imshow(phase_2, extent=(0.61, 0.78, 3.55, 4.4))
    ax.imshow(phase_3, extent=(0.8, 0.97, 3.55, 4.4))
    ax.set_aspect(0.2)  # 纵轴的单位长度是横轴单位长度的0.65倍
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    ax.tick_params(axis='both', labelsize=8)  # 同时设置x轴和y轴刻度标签的字体大小

    # 横排
    ax.text(0.502, 3.9, 'I', fontsize=8)
    ax.text(0.676, 3.9, 'II', fontsize=8)
    ax.text(0.862, 3.902, 'III', fontsize=8)
    # 纵排
    ax.text(0.9, 1.12, 'I', fontsize=8)
    ax.text(0.89, 1.7, 'II', fontsize=8)
    ax.text(0.88, 2.4, 'III', fontsize=8)

    # ax.set_title('schizophrenia', fontsize=12)

    ax.fill_between(cutoff, 0, interval_thrs_1, facecolor='green', alpha=0.2,
                    linewidth=0)  # 区域的下边界为0，上边界为interval_thrs_1
    ax.fill_between(cutoff, interval_thrs_1, interval_thrs_2, facecolor='red', alpha=0.2, edgecolor="k", linewidth=0)
    ax.fill_between(cutoff, interval_thrs_2, 5.2, edgecolor="k", alpha=0.2, linewidth=0.0)

def plot_betti_reduce(xpos,ypos):
    betti0_npy, betti1_npy, betti2_npy = [], [], []
    for i in range(10):
        betti_npy = np.load(fr'D:\python\result\EEG\npy/reduce_betti_{round((i + 1) / 10, 1)}.npy')
        column0 = betti_npy[:, 0].tolist()
        column1 = betti_npy[:, 1].tolist()
        column2 = betti_npy[:, 2].tolist()
        betti0_npy.append(column0)
        betti1_npy.append(column1)
        betti2_npy.append(column2)
    betti0_npy = np.array(betti0_npy).T
    betti1_npy = np.array(betti1_npy).T
    betti2_npy = np.array(betti2_npy).T

    colors = [(0, 0, 0.8), (0, 0.5, 1), (0.95, 1, 1), (1, 0.5, 0), (0.7, 0, 0)]  # RGBA tuples
    cmap_name = 'custom_blue_white_red'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # x轴和y轴的刻度标签
    xticklabels = ['','0.2','','0.4','','0.6','','0.8','','1.0']
    yticklabels = [str(i) for i in range(1, 15)]

    ax1 = axs[xpos, ypos]
    norm = TwoSlopeNorm(vmin=np.min(betti0_npy), vcenter=0, vmax=np.max(betti0_npy))
    im1 = ax1.imshow(betti0_npy, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    # 添加颜色条
    cbar = fig.colorbar(im1, ax=ax1, orientation='vertical')
    cbar.set_ticks(np.arange(int(np.min(betti0_npy))+1,int(np.max(betti0_npy))+1,2))
    cbar.set_ticklabels([str(v) for v in np.arange(int(np.min(betti0_npy))+1,int(np.max(betti0_npy))+1,2)])
    cbar.ax.tick_params(labelsize=8)

    # 设置x轴和y轴的刻度标签
    ax1.set_xticks(range(len(xticklabels)))
    ax1.set_xticklabels(xticklabels)
    ax1.set_yticks(range(len(yticklabels)))
    ax1.set_yticklabels(yticklabels)

    # 设置x轴和y轴的标签
    ax1.set_xlabel('filtration radius', size=12,labelpad=-1)
    ax1.set_ylabel('Sample', size=12,labelpad=-1)

    # 设置刻度标签的大小
    ax1.tick_params(axis='both', labelsize=8)

    ax2 = axs[xpos, ypos + 1]
    norm = TwoSlopeNorm(vmin=np.min(betti1_npy), vcenter=0, vmax=np.max(betti1_npy))
    im1 = ax2.imshow(betti1_npy, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    # 添加颜色条
    cbar = fig.colorbar(im1, ax=ax2, orientation='vertical')
    cbar.set_ticks(np.arange(int(np.min(betti1_npy)) + 1, int(np.max(betti1_npy)) + 1, 2))
    cbar.set_ticklabels([str(v) for v in np.arange(int(np.min(betti1_npy)) + 1, int(np.max(betti1_npy)) + 1, 2)])
    cbar.ax.tick_params(labelsize=8)
    # 设置x轴和y轴的刻度标签
    ax2.set_xticks(range(len(xticklabels)))
    ax2.set_xticklabels(xticklabels)
    ax2.set_yticks(range(len(yticklabels)))
    ax2.set_yticklabels(yticklabels)

    # 设置x轴和y轴的标签
    ax2.set_xlabel('filtration radius', size=12,labelpad=-1)
    ax2.set_ylabel('Sample', size=12,labelpad=-1)

    # 设置刻度标签的大小
    ax2.tick_params(axis='both', labelsize=8)

    ax3 = axs[xpos, ypos + 2]
    norm = TwoSlopeNorm(vmin=np.min(betti2_npy), vcenter=0, vmax=np.max(betti2_npy))
    im1 = ax3.imshow(betti2_npy, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    # 添加颜色条
    cbar = fig.colorbar(im1, ax=ax3, orientation='vertical')
    cbar.set_ticks(np.arange(int(np.min(betti2_npy)), int(np.max(betti2_npy))+1, 1))
    cbar.set_ticklabels([str(v) for v in np.arange(int(np.min(betti2_npy)), int(np.max(betti2_npy))+1, 1)])
    cbar.ax.tick_params(labelsize=8)
    # 设置x轴和y轴的刻度标签
    ax3.set_xticks(range(len(xticklabels)))
    ax3.set_xticklabels(xticklabels)
    ax3.set_yticks(range(len(yticklabels)))
    ax3.set_yticklabels(yticklabels)

    # 设置x轴和y轴的标签
    ax3.set_xlabel('filtration radius', size=12,labelpad=-1)
    ax3.set_ylabel('Sample', size=12,labelpad=-1)

    # 设置刻度标签的大小
    ax3.tick_params(axis='both', labelsize=8)

def start_drawing():
    plot_connective_matrix(0,0,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                               r'h_couplematrix_cutoff_0.281_couple_%.3f.npy'%couple)
    plot_connective_matrix(1,0,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                               r's_couplematrix_cutoff_0.322_couple_%.3f.npy'%couple)
    plot_connective_matrix(2,0,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                               r'h_couplematrix_cutoff_0.633_couple_%.3f.npy'%couple)
    plot_connective_matrix(3,0,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                               r's_couplematrix_cutoff_0.725_couple_%.3f.npy'%couple)
    plot_rainbow(0,1,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                r'h_amplitudezc03s10_cutoff_0.281_couple_%.3f.npy'%couple)
    plot_rainbow(1,1,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                r's_amplitudezc03s10_cutoff_0.322_couple_%.3f.npy'%couple)
    plot_rainbow(2,1,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                r'h_amplitudezc03s10_cutoff_0.633_couple_%.3f.npy'%couple)
    plot_rainbow(3,1,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                r's_amplitudezc03s10_cutoff_0.725_couple_%.3f.npy'%couple)
    plot_butterfly(0,2,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                  r'h_singlec03s10_cutoff_0.281_couple_%.3f.npy'%couple)
    plot_butterfly(1,2,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                  r's_singlec03s10_cutoff_0.322_couple_%.3f.npy'%couple)
    plot_butterfly(2,2,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                  r'h_singlec03s10_cutoff_0.633_couple_%.3f.npy'%couple)
    plot_butterfly(3,2,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                  r's_singlec03s10_cutoff_0.725_couple_%.3f.npy'%couple)
    plot_Betti_time(0,3,r'D:\python\result\EEG\npy\betti_npy/'
                                   r'evolution_betti_number_healthy_cutoff_0.351_couple_%.3f.npy'%couple)
    plot_Betti_time(1,3,r'D:\python\result\EEG\npy\betti_npy/'
                                   r'evolution_betti_number_schizophrenia_cutoff_0.403_couple_%.3f.npy'%couple)
    plot_Betti_time(2,3,r'D:\python\result\EEG\npy\betti_npy/'
                                   r'evolution_betti_number_healthy_cutoff_0.633_couple_%.3f.npy'%couple)
    plot_Betti_time(3,3,r'D:\python\result\EEG\npy\betti_npy/'
                                   r'evolution_betti_number_schizophrenia_cutoff_0.725_couple_%.3f.npy'%couple)
    plot_Betti_filtration_radius(0,4,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                   r'h_amplitudezc03s10_cutoff_0.351_couple_%.3f.npy'%couple,'healthy',0.351)
    plot_Betti_filtration_radius(1,4,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                   r's_amplitudezc03s10_cutoff_0.403_couple_%.3f.npy'%couple,'schizophrenia',0.403)
    plot_Betti_filtration_radius(2,4,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                   r'h_amplitudezc03s10_cutoff_0.633_couple_%.3f.npy'%couple,'healthy',0.633)
    plot_Betti_filtration_radius(3,4,r'D:\python\result\EEG\npy\couple_npy\amplitudezc03s10_npy/'
                                   r's_amplitudezc03s10_cutoff_0.725_couple_%.3f.npy'%couple,'schizophrenia',0.725)
    plot_phase_diagram_healthy(4,0)
    plot_phase_diagram_schizophrenia(4, 1)
    plot_betti_reduce(4,2)
    axs[0,0].text(-10,-1,'a',size = 20)
    axs[1,0].text(-10,-1,'b',size = 20)
    axs[2,0].text(-10,-1,'c',size = 20)
    axs[3,0].text(-10,-1,'d',size = 20)
    axs[4,0].text(-0.28,5.5,'e',size = 20)
    axs[4,1].text(-0.2,5.5,'f',size = 20)
    axs[4,2].text(-3,15.3,'g',size = 20)
    axs[4,3].text(-3,14.7,'h',size = 20)
    axs[4,4].text(-3,14.7,'i',size = 20)
    axs[0, 0].text(-10, 12, 'healthy', size=14,rotation=90)
    axs[2, 0].text(-10, 12, 'healthy', size=14,rotation=90)
    axs[1, 0].text(-10, 15.8, 'schizophrenia', size=14,rotation=90)
    axs[3, 0].text(-10, 15.8, 'schizophrenia', size=14,rotation=90)
    axs[4, 0].text(-0.25, 2.3, 'healthy', size=14,rotation=90)
    axs[4, 1].text(-0.25, 1.6, 'schizophrenia', size=14,rotation=90)

plt.rcParams['font.family'] = 'Times New Roman'
couple = 2.8

# 创建一个5x5的子图网格
fig, axs = plt.subplots(5, 5, figsize=(12, 10), constrained_layout=True) # 使用constrained_layout自动调整布局

# cutoffs = [0.070,0.141,0.211,0.281,0.351,0.422,0.492,0.562,0.633,0.703] # healthy
# cutoffs = [0.081,0.161,0.241,0.322,0.403,0.483,0.564,0.644,0.725,0.805] # schizophrenia
plt.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.04, hspace=0.38, wspace=0.35)
start_drawing()

# 显示图形
plt.savefig(r'D:\python\result\EEG\fig\fig3/fig3.pdf')
plt.show()
