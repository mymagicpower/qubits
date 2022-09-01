# from scipy import make_blobs
# https://blog.csdn.net/qq_42902997/article/details/109198424
from sklearn.datasets import make_blobs
import pyqpanda as pq
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
import numpy as np
import math
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

# 根据数据的数据量n，聚类中心k和数据标准差std返回对应数据点和聚类中心点
def get_data(n, k, std):
    data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)
    points = data[0]
    centers = data[1]
    return points, centers

# 根据输入的坐标点d(x,y)来计算输入的量子门旋转角度
def get_theta(d):
    x = d[0]
    y = d[1]
    theta = 2 * math.acos((x.item() + y.item()) / 2.0)
    return theta

# 根据输入的量子数据点构建量子线路
def qkmeans_circuits(x, y):

    theta_1 = get_theta(x)
    theta_2 = get_theta(y)

    num_qubits = 3
    machine = pq.CPUQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)
    cbits = machine.cAlloc_many(num_qubits)
    circuit = pq.QCircuit()

    circuit.insert(pq.H(qubits[0]))
    circuit.insert(pq.H(qubits[1]))
    circuit.insert(pq.H(qubits[2]))

    circuit.insert(pq.U3(qubits[1], theta_1, np.pi, np.pi))
    circuit.insert(pq.U3(qubits[2], theta_2, np.pi, np.pi))

    circuit.insert(pq.SWAP(qubits[1], qubits[2]).control([qubits[0]]))

    circuit.insert(pq.H(qubits[0]))

    prog = pq.QProg()
    prog.insert(circuit)
    prog << pq.Measure(qubits[0], cbits[0])
    prog.insert(pq.Reset(qubits[0]))
    prog.insert(pq.Reset(qubits[1]))
    prog.insert(pq.Reset(qubits[2]))

    result = machine.run_with_configuration(prog, cbits, 1024)

    data = result

    if len(data) == 1:
        return 0.0
    else:
        return data['001'] / 1024.0

# 对散点和聚类中心进行可视化
def draw_plot(points, centers, label=True):
    points = np.array(points)
    centers = np.array(centers)
    if label==False:
        plt.scatter(points[:,0], points[:,1])
    else:
        plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

# 随机生成聚类中心点
def initialize_centers(points,k):
    return points[np.random.randint(points.shape[0],size=k),:]


def find_nearest_neighbour(points, centroids):
    n = points.shape[0]
    k = centroids.shape[0]

    centers = tensor.zeros([n])

    for i in range(n):
        min_dis = 10000
        ind = 0
        for j in range(k):

            temp_dis = qkmeans_circuits(points[i, :], centroids[j, :])

            if temp_dis < min_dis:
                min_dis = temp_dis
                ind = j
        centers[i] = ind

    return centers

def find_centroids(points, centers):

    k = int(tensor.max(centers).item()) + 1
    centroids = tensor.zeros([k, 2])
    for i in range(k):

        cur_i = centers == i

        x = points[:,0]
        x = x[cur_i]
        y = points[:,1]
        y = y[cur_i]
        centroids[i, 0] = tensor.mean(x)
        centroids[i, 1] = tensor.mean(y)

    return centroids

def preprocess(points):
    n = len(points)
    x = 30.0 * np.sqrt(2)
    for i in range(n):
        points[i, :] += 15
        points[i, :] /= x

    return points


def qkmean_run():
    n = 100  # number of data points
    k = 3  # Number of centers
    std = 2  # std of datapoints

    points, o_centers = get_data(n, k, std)  # dataset

    points = preprocess(points)  # Normalize dataset

    centroids = initialize_centers(points, k)  # Intialize centroids

    epoch = 9
    points = QTensor(points)
    centroids = QTensor(centroids)
    plt.figure()
    draw_plot(points.data, o_centers,label=False)

    # 运行算法
    for i in range(epoch):
            centers = find_nearest_neighbour(points, centroids)  # find nearest centers
            centroids = find_centroids(points, centers)  # find centroids

    plt.figure()
    draw_plot(points.data, centers.data)

# 运行程序入口
if __name__ == "__main__":
    qkmean_run()