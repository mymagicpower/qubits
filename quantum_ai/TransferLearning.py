"""
Quantum Classic Nerual Network Transfer Learning demo
"""

import os
import sys
sys.path.insert(0,'../')
import numpy as np
import matplotlib.pyplot as plt

from pyvqnet.nn.module import Module
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.conv import Conv2D
from pyvqnet.utils.storage import load_parameters, save_parameters

from pyvqnet.nn import activation as F
from pyvqnet.nn.pooling import MaxPool2D

from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.nn.loss import SoftmaxCrossEntropy

from pyvqnet.optim.sgd import SGD
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
import pyqpanda as pq
from pyqpanda import *
import matplotlib
from pyvqnet.nn.module import *
from pyvqnet.utils.initializer import *
from pyvqnet.qnn.quantumlayer import QuantumLayer

try:
    matplotlib.use('TkAgg')
except:
    pass

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def _download(dataset_dir,file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        with gzip.GzipFile(file_path) as f:
            file_path_ungz = file_path[:-3].replace('\\', '/')
            if not os.path.exists(file_path_ungz):
                open(file_path_ungz,"wb").write(f.read())
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
    print("Done")

def download_mnist(dataset_dir):
    for v in key_file.values():
        _download(dataset_dir,v)


IF_PLOT = False
if not os.path.exists("./result"):
    os.makedirs("./result")
else:
    pass
# classical CNN
class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = Conv2D(input_channels=1, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d1 = BatchNorm2d(16)
        self.Relu1 = F.ReLu()

        self.conv2 = Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d2 = BatchNorm2d(32)
        self.Relu2 = F.ReLu()
        self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")

        self.conv3 = Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d3 = BatchNorm2d(64)
        self.Relu3 = F.ReLu()

        self.conv4 = Conv2D(input_channels=64, output_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.BatchNorm2d4 = BatchNorm2d(128)
        self.Relu4 = F.ReLu()
        self.maxpool4 = MaxPool2D([2, 2], [2, 2], padding="valid")

        self.fc1 = Linear(input_channels=128 * 4 * 4, output_channels=1024)
        self.fc2 = Linear(input_channels=1024, output_channels=128)
        self.fc3 = Linear(input_channels=128, output_channels=10)

    def forward(self, x):

        x = self.Relu1(self.conv1(x))
        x = self.maxpool2(self.Relu2(self.conv2(x)))
        x = self.Relu3(self.conv3(x))
        x = self.maxpool4(self.Relu4(self.conv4(x)))
        x = tensor.flatten(x, 1)
        x = F.ReLu()(self.fc1(x))  # 1 64
        x = F.ReLu()(self.fc2(x))  # 1 64
        x = self.fc3(x)  # 1 1
        return x

def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
    import os, struct
    from array import array as pyarray
    download_mnist(path)
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
        fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
        fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    images = np.zeros((N, rows, cols))
    labels = np.zeros((N, 1), dtype=int)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def data_select(train_num, test_num):
    x_train, y_train = load_mnist("training_data")  # 下载训练数据

    x_test, y_test = load_mnist("testing_data")

    # Train Leaving only labels 0 and 1
    idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                          np.where(y_train == 1)[0][:train_num])

    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_train = x_train / 255
    y_train = np.eye(2)[y_train].reshape(-1, 2)

    # Test Leaving only labels 0 and 1
    idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                         np.where(y_test == 1)[0][:test_num])

    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    x_test = x_test / 255
    y_test = np.eye(2)[y_test].reshape(-1, 2)

    return x_train, y_train, x_test, y_test

"""
to get cnn model parameters for transfer learning
"""

train_size = 50
eval_size = 50
EPOCHES = 10
def classcal_cnn_model_making():
    # load train data
    x_train, y_train = load_mnist("training_data", digits=np.arange(10))
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:eval_size]
    y_test = y_test[:eval_size]

    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.eye(10)[y_train].reshape(-1, 10)
    y_test = np.eye(10)[y_test].reshape(-1, 10)

    model = CNN()

    optimizer = SGD(model.parameters(), lr=0.005)
    loss_func = SoftmaxCrossEntropy()

    epochs = EPOCHES
    loss_list = []
    model.train()

    SAVE_FLAG = True
    temp_loss = 0
    for epoch in range(1, epochs):
        total_loss = []
        for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):

            x = x.reshape(-1, 1, 28, 28)
            optimizer.zero_grad()
            # Forward pass
            output = model(x)

            # Calculating loss
            loss = loss_func(y, output)  # target output
            loss_np = np.array(loss.data)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer._step()

            total_loss.append(loss_np)

        loss_list.append(np.sum(total_loss) / len(total_loss))
        print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")


    model.eval()
    correct = 0
    n_eval = 0

    for x, y in data_generator(x_test, y_test, batch_size=4, shuffle=True):
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        loss = loss_func(y, output)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_eval += 1
    print(f"Eval Accuracy: {correct / n_eval}")

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

def classical_cnn_TransferLearning_predict():
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

    x_test = x_test[:eval_size]
    y_test = y_test[:eval_size]
    x_test = x_test / 255
    y_test = np.eye(10)[y_test].reshape(-1, 10)
    model = CNN()

    model_parameter = load_parameters("./result/QCNN_TL_1.model")
    model.load_state_dict(model_parameter)
    model.eval()
    correct = 0
    n_eval = 0

    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)

        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y.argmax(1))
        correct += np.sum(np.array(mask))
        n_eval += 1

    print(f"Eval Accuracy: {correct / n_eval}")

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

def quantum_cnn_TransferLearning():

    n_qubits = 4  # Number of qubits
    q_depth = 6  # Depth of the quantum circuit (number of variational layers)

    def Q_H_layer(qubits, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        circuit = pq.QCircuit()
        for idx in range(nqubits):
            circuit.insert(pq.H(qubits[idx]))
        return circuit

    def Q_RY_layer(qubits, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        circuit = pq.QCircuit()
        for idx, element in enumerate(w):
            circuit.insert(pq.RY(qubits[idx], element))
        return circuit

    def Q_entangling_layer(qubits, nqubits):
        """Layer of CNOTs followed by another shifted layer of CNOT.
        """
        # In other words it should apply something like :
        # CNOT  CNOT  CNOT  CNOT...  CNOT
        #   CNOT  CNOT  CNOT...  CNOT
        circuit = pq.QCircuit()
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
        """
        The variational quantum circuit.
        """
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(n_qubits)
        circuit = pq.QCircuit()

        # Reshape weights
        q_weights = q_weights_flat.reshape([q_depth, n_qubits])

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        circuit.insert(Q_H_layer(qubits, n_qubits))

        # Embed features in the quantum node
        circuit.insert(Q_RY_layer(qubits, q_input_features))

        # Sequence of trainable variational layers
        for k in range(q_depth):
            circuit.insert(Q_entangling_layer(qubits, n_qubits))
            circuit.insert(Q_RY_layer(qubits, q_weights[k]))

        # Expectation values in the Z basis
        prog = pq.QProg()
        prog.insert(circuit)

        exp_vals = []
        for position in range(n_qubits):
            pauli_str = "Z" + str(position)
            pauli_map = pq.PauliOperator(pauli_str, 1)
            hamiltion = pauli_map.toHamiltonian(True)
            exp = machine.get_expectation(prog, hamiltion, qubits)
            exp_vals.append(exp)

        return exp_vals

    class Q_DressedQuantumNet(Module):

        def __init__(self):
            """
            Definition of the *dressed* layout.
            """

            super().__init__()
            self.pre_net = Linear(128, n_qubits)
            self.post_net = Linear(n_qubits, 10)
            self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

        def forward(self, input_features):
            """
            Defining how tensors are supposed to move through the *dressed* quantum
            net.
            """

            # obtain the input features for the quantum circuit
            # by reducing the feature dimension from 512 to 4
            pre_out = self.pre_net(input_features)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            q_out_elem = self.temp_Q(q_in)

            result = q_out_elem
            # return the two-dimensional prediction from the postprocessing layer
            return self.post_net(result)

    x_train, y_train = load_mnist("training_data", digits=np.arange(10))  # 下载训练数据
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:eval_size]
    y_test = y_test[:eval_size]

    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.eye(10)[y_train].reshape(-1, 10)
    y_test = np.eye(10)[y_test].reshape(-1, 10)

    model = CNN()
    model_param = load_parameters("./result/QCNN_TL_1.model")
    model.load_state_dict(model_param)

    loss_func = SoftmaxCrossEntropy()

    epochs = EPOCHES
    loss_list = []
    eval_losses = []

    model_hybrid = model
    print(model_hybrid)

    for param in model_hybrid.parameters():
        param.requires_grad = False
    #替换经典神经网络层为量子层
    model_hybrid.fc3 = Q_DressedQuantumNet()
    #优化器只训练量子层
    optimizer_hybrid = Adam(model_hybrid.fc3.parameters(), lr=0.001)
    model_hybrid.train()

    SAVE_FLAG = True
    temp_loss = 0
    for epoch in range(1, epochs):
        total_loss = []
        for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            optimizer_hybrid.zero_grad()
            # Forward pass
            output = model_hybrid(x)

            loss = loss_func(y, output)  # target output
            loss_np = np.array(loss.data)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer_hybrid._step()
            total_loss.append(loss_np)

        loss_list.append(np.sum(total_loss) / len(total_loss))
        print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            save_parameters(model_hybrid.fc3.state_dict(), "./result/QCNN_TL_FC3.model")
            save_parameters(model_hybrid.state_dict(), "./result/QCNN_TL_ALL.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                save_parameters(model_hybrid.fc3.state_dict(), "./result/QCNN_TL_FC3.model")
                save_parameters(model_hybrid.state_dict(), "./result/QCNN_TL_ALL.model")

        correct = 0
        n_eval = 0
        loss_temp =[]
        for x1, y1 in data_generator(x_test, y_test, batch_size=4, shuffle=True):
            x1 = x1.reshape(-1, 1, 28, 28)
            output = model_hybrid(x1)
            loss = loss_func(y1, output)
            np_loss = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y1.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1
            loss_temp.append(np_loss)
        eval_losses.append(np.sum(loss_temp) / n_eval)
        print("{:.0f} eval loss is : {:.10f}".format(epoch, eval_losses[-1]))


    plt.title('model loss')
    plt.plot(loss_list, color='green', label='train_losses')
    plt.plot(eval_losses, color='red', label='eval_losses')
    plt.ylabel('loss')
    plt.legend(["train_losses", "eval_losses"])
    plt.savefig("qcnn_transfer_learning_classical")
    plt.show()
    plt.close()

    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model_hybrid.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model_hybrid(x)
        pred = QTensor.argmax(output, [1])
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()

def quantum_cnn_TransferLearning_predict():

    n_qubits = 4  # Number of qubits
    q_depth = 6  # Depth of the quantum circuit (number of variational layers)

    def Q_H_layer(qubits, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        circuit = pq.QCircuit()
        for idx in range(nqubits):
            circuit.insert(pq.H(qubits[idx]))
        return circuit

    def Q_RY_layer(qubits, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        circuit = pq.QCircuit()
        for idx, element in enumerate(w):
            circuit.insert(pq.RY(qubits[idx], element))
        return circuit

    def Q_entangling_layer(qubits, nqubits):
        """Layer of CNOTs followed by another shifted layer of CNOT.
        """
        # In other words it should apply something like :
        # CNOT  CNOT  CNOT  CNOT...  CNOT
        #   CNOT  CNOT  CNOT...  CNOT
        circuit = pq.QCircuit()
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
        """
        The variational quantum circuit.
        """
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(n_qubits)
        circuit = pq.QCircuit()

        # Reshape weights
        q_weights = q_weights_flat.reshape([q_depth, n_qubits])

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        circuit.insert(Q_H_layer(qubits, n_qubits))

        # Embed features in the quantum node
        circuit.insert(Q_RY_layer(qubits, q_input_features))

        # Sequence of trainable variational layers
        for k in range(q_depth):
            circuit.insert(Q_entangling_layer(qubits, n_qubits))
            circuit.insert(Q_RY_layer(qubits, q_weights[k]))

        # Expectation values in the Z basis
        prog = pq.QProg()
        prog.insert(circuit)
        exp_vals = []
        for position in range(n_qubits):
            pauli_str = "Z" + str(position)
            pauli_map = pq.PauliOperator(pauli_str, 1)
            hamiltion = pauli_map.toHamiltonian(True)
            exp = machine.get_expectation(prog, hamiltion, qubits)
            exp_vals.append(exp)

        return exp_vals

    class Q_DressedQuantumNet(Module):

        def __init__(self):
            """
            Definition of the *dressed* layout.
            """

            super().__init__()
            self.pre_net = Linear(128, n_qubits)
            self.post_net = Linear(n_qubits, 10)
            self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

        def forward(self, input_features):
            """
            Defining how tensors are supposed to move through the *dressed* quantum
            net.
            """

            # obtain the input features for the quantum circuit
            # by reducing the feature dimension from 512 to 4
            pre_out = self.pre_net(input_features)
            q_in = tensor.tanh(pre_out) * np.pi / 2.0
            q_out_elem = self.temp_Q(q_in)

            result = q_out_elem
            # return the two-dimensional prediction from the postprocessing layer
            return self.post_net(result)


    x_train, y_train = load_mnist("training_data", digits=np.arange(10))  # 下载训练数据
    x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
    x_train = x_train[:2000]
    y_train = y_train[:2000]
    x_test = x_test[:500]
    y_test = y_test[:500]

    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.eye(10)[y_train].reshape(-1, 10)
    y_test = np.eye(10)[y_test].reshape(-1, 10)


    # The second method: unified storage and unified reading
    model = CNN()
    model_hybrid = model
    model_hybrid.fc3 = Q_DressedQuantumNet()
    for param in model_hybrid.parameters():
        param.requires_grad = False
    model_param_quantum = load_parameters("QCNN_TL_ALL.model")

    model_hybrid.load_state_dict(model_param_quantum)
    model_hybrid.eval()

    loss_func = SoftmaxCrossEntropy()
    eval_losses = []

    correct = 0
    n_eval = 0
    loss_temp =[]
    eval_batch_size = 4
    for x1, y1 in data_generator(x_test, y_test, batch_size=eval_batch_size, shuffle=True):
        x1 = x1.reshape(-1, 1, 28, 28)
        output = model_hybrid(x1)
        loss = loss_func(y1, output)
        np_loss = np.array(loss.data)
        np_output = np.array(output.data, copy=False)
        mask = (np_output.argmax(1) == y1.argmax(1))
        correct += np.sum(np.array(mask))

        n_eval += 1
        loss_temp.append(np_loss)

    eval_losses.append(np.sum(loss_temp) / n_eval)
    print(f"Eval Accuracy: {correct / (eval_batch_size*n_eval)}")

    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model_hybrid.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model_hybrid(x)
        pred = QTensor.argmax(output, [1])
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()

if __name__ == "__main__":
    # save classic model parameters
    if not os.path.exists('./result/QCNN_TL_1.model'):
        classcal_cnn_model_making()
        classical_cnn_TransferLearning_predict()
    #train quantum circuits.
    print("use exist cnn model param to train quantum parameters.")
    quantum_cnn_TransferLearning()
    #eval quantum circuits.
    quantum_cnn_TransferLearning_predict()