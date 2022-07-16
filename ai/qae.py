"""
Quantum AutoEncoder demo



"""

import os
import sys
sys.path.insert(0,'../')
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.nn.loss import  fidelityLoss
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.qnn.qae.qae import QAElayer
import matplotlib.pyplot as plt
import matplotlib
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


class Model(Module):

    def __init__(self, trash_num: int = 2, total_num: int = 7):
        super().__init__()
        self.pqc = QAElayer(trash_num, total_num)

    def forward(self, x):

        x = self.pqc(x)
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

def run2():
    ##load dataset
    #x_train,x_test,y_train,y_test = load_mnist("training_data")                      # 下载训练数据

    x_train, y_train = load_mnist("training_data")                      # 下载训练数据
    x_train = x_train / 255                                             # 将数据进行归一化处理[0,1]

    x_test, y_test = load_mnist("testing_data")

    x_test = x_test / 255

    x_train = x_train.reshape([-1, 1, 28, 28])
    x_test = x_test.reshape([-1, 1, 28, 28])
    x_train = x_train[:100, :, :, :]
    x_train = np.resize(x_train, [x_train.shape[0], 1, 2, 2])

    x_test = x_test[:10, :, :, :]
    x_test = np.resize(x_test, [x_test.shape[0], 1, 2, 2])
    encode_qubits = 4
    latent_qubits = 2
    trash_qubits = encode_qubits - latent_qubits
    total_qubits = 1 + trash_qubits + encode_qubits
    print("model start")
    model = Model(trash_qubits, total_qubits)

    optimizer = Adam(model.parameters(), lr=0.005)
    model.train()
    F1 = open("rlt.txt", "w")
    loss_list = []
    loss_list_test = []
    fidelity_train = []
    fidelity_val = []

    for epoch in range(1, 10):
        running_fidelity_train = 0
        running_fidelity_val = 0
        print(f"epoch {epoch}")
        model.train()
        full_loss = 0
        n_loss = 0
        n_eval = 0
        batch_size = 1
        correct = 0
        iter = 0
        if epoch %5 ==1:
            optimizer.lr  = optimizer.lr *0.5
        for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True): #shuffle batch rather than data

            x = x.reshape((-1, encode_qubits))
            x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]), x), 1)
            optimizer.zero_grad()
            output = model(x)
            iter += 1
            np_out = np.array(output.data)
            floss = fidelityLoss()
            loss = floss(output)
            loss_data = np.array(loss.data)
            loss.backward()

            running_fidelity_train += np_out[0]
            optimizer._step()
            full_loss += loss_data[0]
            n_loss += batch_size
            np_output = np.array(output.data, copy=False)
            mask = np_output.argmax(1) == y.argmax(1)

            correct += sum(mask)

        loss_output = full_loss / n_loss
        print(f"Epoch: {epoch}, Loss: {loss_output}")
        loss_list.append(loss_output)

        # F1.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

        # Evaluation
        model.eval()
        correct = 0
        full_loss = 0
        n_loss = 0
        n_eval = 0
        batch_size = 1
        for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
            x = x.reshape((-1, encode_qubits))
            x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]),x),1)
            output = model(x)

            floss = fidelityLoss()
            loss = floss(output)
            loss_data = np.array(loss.data)
            full_loss += loss_data[0]
            running_fidelity_val += np.array(output.data)[0]

            n_eval += 1
            n_loss += 1

        loss_output = full_loss / n_loss
        print(f"Epoch: {epoch}, Loss: {loss_output}")
        loss_list_test.append(loss_output)

        fidelity_train.append(running_fidelity_train / 64)
        fidelity_val.append(running_fidelity_val / 64)

    figure_path = os.path.join(os.getcwd(), 'QAE-rate1.png')
    plt.plot(loss_list, color="blue", label="train")
    plt.plot(loss_list_test, color="red", label="validation")
    plt.title('QAE')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(figure_path)
    plt.show()

    F1.write(f"done\n")
    F1.close()
    del model

if __name__ == '__main__':
    run2()