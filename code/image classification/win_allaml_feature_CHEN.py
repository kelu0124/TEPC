import numpy as np
import sys
import time

class OscillatorCollection:
    def __init__(self):
        self.n = 0
        self.h = 0.0
        self.name = ""
        self.x = []
        self.y = []
        self.z = []
        self.u = []
        self.couple_matrix = []
        self.atoms = []

    def initialize(self):
        self.name = "coil_20_reduced"  # 文件名
        self.h = 0.01
        sigma = 8.0
        k = 2.0
        atoms = np.load("/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/coil_20_reduced.npy")
        self.atoms = atoms
        self.n = len(atoms)
        self.u = np.random.rand(self.n, 3)

        couple_matrix = []
        for i in range(self.n):
            tmp_vec = []
            for j in range(self.n):
                tmp = np.sqrt(np.sum((atoms[i] - atoms[j])**2))  # 计算距离
                if i != j:
                    tmp_vec.append(np.exp(-np.power(tmp, k) / np.power(sigma, k)))
                else:
                    tmp_vec.append(0.0)
            for j in range(self.n):
                if i != j:
                    tmp_vec[i] -= tmp_vec[j]
            couple_matrix.append(tmp_vec)
        self.couple_matrix = couple_matrix

    def run(self):
        # 4th Order Runge-Kutta
        k1 = self.f(self.u)
        tmp_vec = self.u + self.h * 0.5 * k1
        k2 = self.f(tmp_vec)
        tmp_vec = self.u + self.h * 0.5 * k2
        k3 = self.f(tmp_vec)
        tmp_vec = self.u + self.h * k3
        k4 = self.f(tmp_vec)
        self.u += (1.0 / 6.0) * self.h * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def perturb(self, ip):
        self.u[ip * 3 + 2] += self.u[ip * 3 + 2]
        return 0  # Assuming the return type is for potential future use

    def f(self, uu):
        delta = 1.0
        gamma = 12.0
        beta = 8.0 / 3.0
        epsilon = 0.12
        fuu = np.zeros((self.n, 3))

        for i in range(self.n):
            couplex = 0.0
            coupley = 0.0
            couplez = 0.0
            for j in range(self.n):
                if self.couple_matrix[i][j] != 0.0:
                    couplex += epsilon * self.couple_matrix[i][j] * uu[j][0]
                    coupley += epsilon * self.couple_matrix[i][j] * uu[j][1]
                    couplez += epsilon * self.couple_matrix[i][j] * uu[j][2]

            fuu[i][0] = delta * (uu[i][1] - uu[i][0]) + couplex
            fuu[i][1] = gamma * uu[i][0] - uu[i][1] - uu[i][0] * uu[i][2] + coupley
            fuu[i][2] = uu[i][0] * uu[i][1] - beta * uu[i][2] + couplez

        return fuu

def generate_feature(x, y, z):
    # 六个统计量特征
    f1 = np.mean(x)  # 平均值
    f2 = np.max(x)   # 最大值
    f3 = np.min(x)   # 最小值
    f4 = np.median(x)  # 中位数
    f5 = np.var(x)    # 方差
    f6 = np.std(x)    # 标准差
    return np.array([f1, f2, f3, f4, f5, f6])

def main():
    np.random.seed(int(time.time()))
    threshold = 0.001
    timelength = 1000
    relax_iterations = 10000

    OC = OscillatorCollection()
    OC.initialize()
    N = OC.n

    orbit = np.zeros((N, timelength))

    # Relaxation
    for _ in range(relax_iterations):
        OC.run()

    fx = OC.u[0]
    fy = OC.u[1]
    fz = OC.u[2]

    features = []

    with open(OC.name + ".orbit", "w") as orbit_file:
        orbit_file.write(f"{OC.n} {timelength}\n")

        for i in range(OC.n):
            OC.u[i][0] = fx
            OC.u[i][1] = fy
            OC.u[i][2] = fz

            OC.perturb(i)

            for j in range(timelength):
                for k in range(OC.n):
                    tmp = np.sqrt(np.sum((fx - OC.u[k][0])**2 + (fy - OC.u[k][1])**2 + (fz - OC.u[k][2])**2))
                    orbit[k][j] = tmp
                OC.run()

            features.append(generate_feature(OC.u[:, 0], OC.u[:, 1], OC.u[:, 2]))

            for k in range(OC.n):
                for j in range(timelength):
                    orbit_file.write(f"{orbit[k][j]}\n")

    # Save features
    features = np.array(features)
    np.save(f"{OC.name}_4RK_feature_cutoff_{threshold}_couple_{OC.couple_matrix[0][0]}.npy", features)

if __name__ == "__main__":
    main()
