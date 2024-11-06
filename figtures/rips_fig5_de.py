import sys
import os
import itertools
import numpy as np
import networkx as nx
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["font.family"] = "Times New Roman"
np.random.seed(4)

def plotBarcode(ax, simplexes, radii, eigs):
    from matplotlib import collections as mc
    ax_ = ax.twinx()
    ax_.tick_params(axis='y', colors='#287cb7',labelsize=28)
    ax_.plot(radii, eigs, '#287cb7',linewidth=3)
    lines = []
    start = 0
    for simplex in simplexes:
        #lines.append([(simplex[0], start), (simplex[1], start)])
        start += 0.01
    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    #ax.set_yticks([])
    #ax.set_yticks(np.arange(0, 31, 5))
    return

class SimplicialComplex:
    def __init__(self, plottype='1', numVertex=30):
        self.numVertex = numVertex

        # Load vertices from topologicaldynamics.xyz
        self.vertices = np.loadtxt('topologicaldynamics.xyz')

        # for networkx
        self.pos = {}
        for i in range(self.numVertex):
            self.pos.update({i: (self.vertices[i,:])})

    def filtration(self, interval=1, death=5, flag_plotBarcode=False):
        import gudhi
        matrixA = np.zeros((self.numVertex, self.numVertex))
        for i in range(self.numVertex):
            for j in range(i + 1, self.numVertex):
                dis = np.linalg.norm(self.vertices[i] - self.vertices[j])
                matrixA[i, j] = dis
                matrixA[j, i] = dis
        rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=death)
        PH = rips_complex.create_simplex_tree(max_dimension=2).persistence()
        simplexes0 = [];
        simplexes1 = [];
        radii = [0.]
        for simplex in PH:
            dim, b, d = simplex[0], simplex[1][0], simplex[1][1]
            if simplex[1][1] > 100: d = death
            if dim == 0:
                simplexes0.append([b, d])
                radii.append(np.round(d, 3))
                radii.append(np.round(d, 3) + 0.001)
            if dim == 1:
                simplexes1.append([b, d])
                radii.append(np.round(b, 3))
                radii.append(np.round(b, 3) + 0.001)
                radii.append(np.round(d, 3))
                radii.append(np.round(d, 3) + 0.001)
        radii = np.sort(radii)
        radii = np.linspace(0, 5, 40)
        L0eigs = [];
        L1eigs = []
        for idx, diameter in enumerate(radii):
            G = nx.random_geometric_graph(self.numVertex, diameter, pos=self.pos)
            self.plotGraph(G, idx, radius=diameter / 2)
            L0eig, L1eig = self.persistenceLaplacian(G, diameter)
            L0eigs.append(L0eig)
            L1eigs.append(L1eig)  # this is not right.

        if flag_plotBarcode:
            # setup figures
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 8.5))
            plotBarcode(ax0, simplexes0, radii, L0eigs)
            segments = [[(0.0, 1), (5, 1)], [(0.0, 1), (2.447865016461431, 1)], [(0.0, 3), (2.4252600616260453, 3)], [(0.0, 4), (2.334637880275518, 4)], [(0.0, 5), (2.2687036898289605, 5)], [(0.0, 6), (2.1572970952722823, 6)], [(0.0, 7), (2.156602303833397, 7)], [(0.0, 8), (2.121667872302978, 8)], [(0.0, 9), (2.1069747029285155, 9)], [(0.0, 10), (2.056143182293519, 10)], [(0.0, 11), (2.054184532708517, 11)], [(0.0, 12), (2.0023384163356632, 12)], [(0.0, 13), (1.9876270934847464, 13)], [(0.0, 14), (1.9811632155335315, 14)], [(0.0, 15), (1.9457295284733749, 15)], [(0.0, 16), (1.9421213253821064, 16)], [(0.0, 17), (1.930483779859983, 17)], [(0.0, 18), (1.9008081669603196, 18)], [(0.0, 19), (1.8394169055869125, 19)], [(0.0, 20), (1.8085196585634264, 20)], [(0.0, 21), (1.798608229726581, 21)], [(0.0, 22), (1.56146060559924, 22)], [(0.0, 23), (1.5203713411337687, 23)], [(0.0, 24), (1.5153252168830897, 24)], [(0.0, 25), (1.4771435624491012, 25)], [(0.0, 26), (1.4163097830347493, 26)], [(0.0, 27), (1.3311982071972732, 27)], [(0.0, 28), (1.2343701158575804, 28)], [(0.0, 29), (1.169986722913593, 29)], [(0.0, 30), (1.070049675283933, 30)]]

            top_coordinates = [(seg[1][0], seg[1][1]) for seg in segments]
            ax0.plot([coord[0] for coord in top_coordinates], [coord[1] for coord in top_coordinates], color='#ff1111',linewidth=3)
            ax0.plot([segments[-1][0][0], segments[-1][1][0]], [segments[-1][0][1], segments[-1][1][1]],
                     color='#ff1111',linewidth=3)
            #ax0.set_xlabel(r'radius $\alpha$', fontsize=40)
            ax0.set_ylabel(r'$\beta_0^{\alpha,0}$', fontsize=40, color='#ff1111')
            ax0_right = ax0.twinx()
            ax0_right.set_ylabel(r'$\lambda_0^{\alpha,0}$', fontsize=40, color='#287cb7')
            ax0_right.set_yticks([])
            ax0_right.yaxis.set_label_coords(1.07, 0.5)
            ax0.tick_params(axis='y', colors='#ff1111',labelsize=28)
            ax0.tick_params(axis='x', labelsize=28)
            ax0.set_xticks([0, 1, 2, 3, 4, 5,],['0', '0.5', '1', '1.5','2', '2.5'])
            ax0.set_yticks(np.arange(0,31,10))
            # ax0.grid(True, linestyle='--', linewidth=0.5, color='gray')
            ax0.set_xlim([0, 5])
            #plt.xlim([0, death])
            # plt.savefig('figure1.png')
            # plt.close(fig)

            plotBarcode(ax1, simplexes1, radii, L1eigs)
            def piecewise_function(x):
                if x < 2.616437893926333:
                    return 0
                elif 2.616437893926333 <= x < 2.6272828666034713:
                    return 1
                elif 2.6272828666034713 <= x < 2.845095615668788:
                    return 2
                elif 2.845095615668788 <= x < 2.8977032548724875:
                    return 3
                elif 2.8977032548724875 <= x < 3.1937308838885397:
                    return 4
                elif 3.1937308838885397 <= x < 3.406855172635758:
                    return 3
                elif 3.406855172635758 <= x < 3.989901006719163:
                    return 2
                elif 3.989901006719163 <= x <= 5:
                    return 1
                else:
                    return None

            x_values = np.linspace(0, death)

            # 计算对应的 y 值
            y_values = [piecewise_function(x) for x in x_values]

            ax1.plot(x_values, y_values, color='#ff1111',linewidth=3)
            ax1.set_xlabel(r'radius $\alpha$', fontsize=40)
            #ax1.set_ylabel(r'$\lambda_1^{\alpha,0}$', fontsize=32, color='#287cb7')
            ax1.set_ylabel(r'$\beta_1^{\alpha,0}$', fontsize=40, color='#ff1111')
            ax1.tick_params(axis='y', colors='#ff1111',labelsize=28)
            ax1.tick_params(axis='x',labelsize=28)
            ax1_right = ax0.twinx()
            #ax1_right.set_ylabel(r'$\beta_1^{\alpha,0}$', fontsize=32, color='#ff1111')
            ax1_right.set_ylabel(r'$\lambda_1^{\alpha,0}$', fontsize=40, color='#287cb7')
            ax1_right.set_yticks([])
            #ax1_right.set_yticks(np.arange(0,0.7,0.2))
            ax1_right.yaxis.set_label_coords(1.07, -0.6)
            ax1.set_yticks(np.arange(0, 5, 1))
            ax1.set_xticks(np.arange(0,6,1))
            ax1.set_xticks([0, 1, 2, 3, 4, 5], ['0', '0.5', '1', '1.5', '2', '2.5'])
            ax1.set_xlim([0, 5])
            #plt.xlim([0, death])
            #plt.xlim([0, 5])
            # plt.savefig('figure2.png')
            # plt.close(fig)
            plt.savefig("大图.png",dpi=1200)
            plt.savefig("大图.svg",dpi=1200)
            plt.show()
        return

    def persistenceLaplacian(self, G, radius):
        """
        Compute the persistence Laplacian.

        Parameters:
        - G: Graph
        - radius: Radius for the persistence Laplacian

        Returns:
        - L0eig: L0 eigenvalue
        - L1eig: L1 eigenvalue
        """
        edges = [x for x in nx.enumerate_all_cliques(G) if len(x) == 2]
        triangles = [x for x in nx.enumerate_all_cliques(G) if len(x) == 3]

        matrixA = np.zeros((self.numVertex, self.numVertex), dtype=np.int16)
        for idx, pos in enumerate(self.vertices):
            for jdx in range(idx + 1, self.numVertex):
                dis = np.linalg.norm(pos - self.vertices[jdx])
                if dis < radius: matrixA[idx, jdx] = 1
        matrixL = matrixA + matrixA.T
        matrixL = np.diag(sum(matrixL)) - matrixL
        eigs, eigvs = np.linalg.eig(matrixL)
        eigs = np.sort(eigs)
        index = np.sum(eigs < 1e-6)
        if index != self.numVertex:
            L0eig = eigs[index]
        else:
            L0eig = 0.

        ## NOTE: the following is wrong: there is no orientation on each edge (correct)
        D1T = np.zeros((len(edges), self.numVertex), dtype=np.int16)
        for idx, edge in enumerate(edges):
            D1T[idx, edge[0]] = 1
            D1T[idx, edge[1]] = -1
        D1 = D1T.T
        L0 = D1 @ D1T
        eigs, eigvs = np.linalg.eig(L0)

        ## NOTE: edge part is still not right
        D2T = np.zeros((len(triangles), len(edges)), dtype=np.int16)
        for idx, tri in enumerate(triangles):
            D2T[idx, edges.index([tri[1], tri[2]])] = 1
            D2T[idx, edges.index([tri[0], tri[2]])] = -1
            D2T[idx, edges.index([tri[0], tri[1]])] = 1
        D2 = D2T.T
        L1 = D2 @ D2T + D1T @ D1
        eigs, eigvs = np.linalg.eig(L1)
        eigs = np.sort(eigs)
        index = np.sum(eigs < 1e-6)
        if index != len(triangles):
            L1eig = eigs[index]
        else:
            L1eig = 0.

        return L0eig, L1eig

    def plotGraph(self, G, save_idx=0, radius=0.1):
        k2Simplex = [x for x in nx.enumerate_all_cliques(G) if len(x) == 3]
        k3Simplex = [x for x in nx.enumerate_all_cliques(G) if len(x) == 4]

        # define subplots as a square window
        fig, ax = plt.subplots(figsize=(5, 5))
        # same to pos = nx.get_node_attributes(G, 'pos')
        pos = nx.get_node_attributes(G, 'pos')

        patches = []
        for s in k2Simplex:
            polygon = Polygon([[pos[s[0]][0], pos[s[0]][1]],
                               [pos[s[1]][0], pos[s[1]][1]],
                               [pos[s[2]][0], pos[s[2]][1]]], closed=True)
            patches.append(polygon)
        p = PatchCollection(patches, alpha=0.4)
        colors = 100 * np.random.rand(len(patches))
        p.set_array(colors)
        ax.add_collection(p)

        # plot connecting lines, only
        for idx in G.edges():
            x = np.array((pos[idx[0]][0], pos[idx[1]][0]))
            y = np.array((pos[idx[0]][1], pos[idx[1]][1]))
            # z = np.array((pos[idx[0]][2], pos[idx[1]][2]))

            # plot the connecting lines # zorder to control the layers
            ax.plot(x, y, c='k', zorder=5)

        # Draw circles around each vertex with transparency
        for vertex in G.nodes():
            circle = plt.Circle((pos[vertex][0], pos[vertex][1]), radius, alpha=0.2, color='lightblue')
            ax.add_patch(circle)

        ax.scatter(self.vertices[:, 0], self.vertices[:, 1], c='r', alpha=0.8, edgecolors='k', zorder=10)
        # remove ticks and grid
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        # plt.show()
        ax.set_title(fr"$\alpha$ = {radius:.2f}",fontsize=28)
        #plt.savefig(f'figures/simplex_{save_idx}.pdf')
        #plt.savefig(f'figures/simplex_{save_idx}.png',dpi=1200)
        #plt.savefig(f'figures/simplex_{save_idx}.svg',dpi=1200)
        return


def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    scomplex = SimplicialComplex()
    scomplex.filtration(flag_plotBarcode=True)

if __name__ == "__main__":
    main()
