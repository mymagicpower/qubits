from pyqpanda import *
import pyqpanda as pq
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.nn.conv import Conv2D, ConvT2D
from pyvqnet.nn.pooling import MaxPool2D
from pyvqnet.nn.linear import Linear
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.data.data import data_generator

def circuit(weights):
    num_qubits = 1
    #pyQPanda 创建模拟器
    machine = pq.CPUQVM()
    machine.init_qvm()
    #pyQPanda 分配量子比特
    qubits = machine.qAlloc_many(num_qubits)
    #pyQPanda 分配经典比特辅助测量
    cbits = machine.cAlloc_many(num_qubits)
    #构建线路
    circuit = pq.QCircuit()
    circuit.insert(pq.H(qubits[0]))
    circuit.insert(pq.RY(qubits[0], weights[0]))

    prog = pq.QProg()
    prog.insert(circuit)
    prog << measure_all(qubits, cbits)

    #运行量子程序
    result = machine.run_with_configuration(prog, cbits, 100)

    counts = np.array(list(result.values()))
    states = np.array(list(result.keys())).astype(float)
    probabilities = counts / 100
    expectation = np.sum(states * probabilities)
    return expectation

#量子计算层的前传和梯度计算函数的定义，其需要继承于抽象类Module
class Hybrid(Module):
    """ Hybrid quantum - Quantum layer definition """
    def __init__(self, shift):
        super(Hybrid, self).__init__()
        self.shift = shift
    def forward(self, input):
        self.input = input
        expectation_z = circuit(np.array(input.data))
        result = [[expectation_z]]
        requires_grad = input.requires_grad and not QTensor.NO_GRAD
        def _backward(g, input):
            """ Backward pass computation """
            input_list = np.array(input.data)
            shift_right = input_list + np.ones(input_list.shape) * self.shift
            shift_left = input_list - np.ones(input_list.shape) * self.shift

            gradients = []
            for i in range(len(input_list)):
                expectation_right = circuit(shift_right[i])
                expectation_left = circuit(shift_left[i])

                gradient = expectation_right - expectation_left
                gradients.append(gradient)
            gradients = np.array([gradients]).T
            return gradients * np.array(g)

        nodes = []
        if input.requires_grad:
            nodes.append(QTensor.GraphNode(tensor=input, df=lambda g: _backward(g, input)))
        return QTensor(data=result, requires_grad=requires_grad, nodes=nodes)

#模型定义
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
        self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
        self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
        self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
        self.fc1 = Linear(input_channels=256, output_channels=64)
        self.fc2 = Linear(input_channels=64, output_channels=1)
        self.hybrid = Hybrid(np.pi / 2)
        self.fc3 = Linear(input_channels=1, output_channels=2)

    def forward(self, x):
        x = F.ReLu()(self.conv1(x))  # 1 6 24 24
        x = self.maxpool1(x)
        x = F.ReLu()(self.conv2(x))  # 1 16 8 8
        x = self.maxpool2(x)
        x = tensor.flatten(x, 1)   # 1 256
        x = F.ReLu()(self.fc1(x))  # 1 64
        x = self.fc2(x)    # 1 1
        x = self.hybrid(x)
        x = self.fc3(x)
        return x

#实例化
model = Net()
#使用Adam完成此任务就足够了，model.parameters（）是模型需要计算的参数。
optimizer = Adam(model.parameters(), lr=0.005)
#分类任务使用交叉熵函数
loss_func = CategoricalCrossEntropy()

#训练次数
epochs = 10
train_loss_list = []
val_loss_list = []
train_acc_list =[]
val_acc_list = []


for epoch in range(1, epochs):
    total_loss = []
    model.train()
    batch_size = 1
    correct = 0
    n_train = 0
    for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):

        x = x.reshape(-1, 1, 28, 28)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(y, output)
        loss_np = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_train += batch_size
        loss.backward()
        optimizer._step()
        total_loss.append(loss_np)

    train_loss_list.append(np.sum(total_loss) / len(total_loss))
    train_acc_list.append(np.sum(correct) / n_train)
    print("{:.0f} loss is : {:.10f}".format(epoch, train_loss_list[-1]))

    model.eval()
    correct = 0
    n_eval = 0

    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        loss = loss_func(y, output)
        loss_np = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_eval += 1

        total_loss.append(loss_np)
    print(f"Eval Accuracy: {correct / n_eval}")
    val_loss_list.append(np.sum(total_loss) / len(total_loss))
    val_acc_list.append(np.sum(correct) / n_eval)

import os
plt.figure()
xrange = range(1,len(train_loss_list)+1)
figure_path = os.path.join(os.getcwd(), 'HQCNN LOSS.png')
plt.plot(xrange,train_loss_list, color="blue", label="train")
plt.plot(xrange,val_loss_list, color="red", label="validation")
plt.title('HQCNN')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(1, epochs +1,step = 2))
plt.legend(loc="upper right")
plt.savefig(figure_path)
plt.show()

plt.figure()
figure_path = os.path.join(os.getcwd(), 'HQCNN Accuracy.png')
plt.plot(xrange,train_acc_list, color="blue", label="train")
plt.plot(xrange,val_acc_list, color="red", label="validation")
plt.title('HQCNN')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, epochs +1,step = 2))
plt.legend(loc="lower right")
plt.savefig(figure_path)
plt.show()

n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
model.eval()
for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
    if count == n_samples_show:
        break
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        pred = QTensor.argmax(output, [1])
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
        plt.show()