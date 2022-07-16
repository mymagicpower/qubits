import os
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.nn.conv import Conv2D, ConvT2D
from pyvqnet.nn import activation as F
from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.nn.loss import BinaryCrossEntropy
from pyvqnet.optim.adam import Adam
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
import pyqpanda as pq
from pyvqnet.utils.storage import load_parameters, save_parameters

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

import cv2
#预处理数据
class PreprocessingData:
    def __init__(self, path):
        self.path = path
        self.x_data = []
        self.y_label = []


    def processing(self):
        list_path = os.listdir((self.path+"/train"))
        for i in range(len(list_path)):

            temp_data = cv2.imread(self.path+"/train" + '/' + list_path[i], cv2.IMREAD_COLOR)
            temp_data = cv2.resize(temp_data, (128, 128))
            grayimg = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
            temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1)
            self.x_data.append(temp_data)

            label_data = cv2.imread(self.path+"/label" + '/' +list_path[i].split(".")[0] + ".png", cv2.IMREAD_COLOR)
            label_data = cv2.resize(label_data, (128, 128))

            label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
            label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1)
            self.y_label.append(label_data)

        return self.x_data, self.y_label

    def read(self):
        self.x_data, self.y_label = self.processing()
        x_data = np.array(self.x_data)
        y_label = np.array(self.y_label)

        return x_data, y_label

#进行量子编码的线路
class QCNN_:
    def __init__(self, image):
        self.image = image

    def encode_cir(self, qlist, pixels):
        cir = pq.QCircuit()
        for i, pix in enumerate(pixels):
            theta = np.arctan(pix)
            phi = np.arctan(pix**2)
            cir.insert(pq.RY(qlist[i], theta))
            cir.insert(pq.RZ(qlist[i], phi))
        return cir

    def entangle_cir(self, qlist):
        k_size = len(qlist)
        cir = pq.QCircuit()
        for i in range(k_size):
            ctr = i
            ctred = i+1
            if ctred == k_size:
                ctred = 0
            cir.insert(pq.CNOT(qlist[ctr], qlist[ctred]))
        return cir

    def qcnn_circuit(self, pixels):
        k_size = len(pixels)
        machine = pq.MPSQVM()
        machine.init_qvm()
        qlist = machine.qAlloc_many(k_size)
        cir = pq.QProg()

        cir.insert(self.encode_cir(qlist, np.array(pixels) * np.pi / 2))
        cir.insert(self.entangle_cir(qlist))

        result0 = machine.prob_run_list(cir, [qlist[0]], -1)
        result1 = machine.prob_run_list(cir, [qlist[1]], -1)
        result2 = machine.prob_run_list(cir, [qlist[2]], -1)
        result3 = machine.prob_run_list(cir, [qlist[3]], -1)

        result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
        machine.finalize()
        return result

def quanconv_(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((64, 64, 1))

    for j in range(0, 128, 2):
        for k in range(0, 128, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = QCNN_(image).qcnn_circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )

            for c in range(1):
                out[j // 2, k // 2, c] = q_results[c]
    return out


#下采样神经网络层的定义
class DownsampleLayer(Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d1 = BatchNorm2d(out_ch)
        self.Relu1 = F.ReLu()
        self.conv2 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d2 = BatchNorm2d(out_ch)
        self.Relu2 = F.ReLu()
        self.conv3 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                            padding="same")
        self.BatchNorm2d3 = BatchNorm2d(out_ch)
        self.Relu3 = F.ReLu()

    def forward(self, x):
        """
        :param x:
        :return: out(Output to deep)，out_2(enter to next level)，
        """
        x1 = self.conv1(x)
        x2 = self.BatchNorm2d1(x1)
        x3 = self.Relu1(x2)
        x4 = self.conv2(x3)
        x5 = self.BatchNorm2d2(x4)
        out = self.Relu2(x5)
        x6 = self.conv3(out)
        x7 = self.BatchNorm2d3(x6)
        out_2 = self.Relu3(x7)
        return out, out_2

#上采样神经网络层的定义
class UpSampleLayer(Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()

        self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d1 = BatchNorm2d(out_ch * 2)
        self.Relu1 = F.ReLu()
        self.conv2 = Conv2D(input_channels=out_ch * 2, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d2 = BatchNorm2d(out_ch * 2)
        self.Relu2 = F.ReLu()

        self.conv3 = ConvT2D(input_channels=out_ch * 2, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                             padding="same")
        self.BatchNorm2d3 = BatchNorm2d(out_ch)
        self.Relu3 = F.ReLu()

    def forward(self, x):
        '''
        :param x: input conv layer
        :param out: connect with UpsampleLayer
        :return:
        '''
        x = self.conv1(x)
        x = self.BatchNorm2d1(x)
        x = self.Relu1(x)
        x = self.conv2(x)
        x = self.BatchNorm2d2(x)
        x = self.Relu2(x)
        x = self.conv3(x)
        x = self.BatchNorm2d3(x)
        x_out = self.Relu3(x)
        return x_out

#Unet整体网络架构
class UNet(Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 4) for i in range(5)]

        # DownSampleLayer
        self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # UpSampleLayer
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # output
        self.conv1 = Conv2D(input_channels=out_channels[1], output_channels=out_channels[0], kernel_size=(3, 3),
                            stride=(1, 1), padding="same")
        self.BatchNorm2d1 = BatchNorm2d(out_channels[0])
        self.Relu1 = F.ReLu()
        self.conv2 = Conv2D(input_channels=out_channels[0], output_channels=out_channels[0], kernel_size=(3, 3),
                            stride=(1, 1), padding="same")
        self.BatchNorm2d2 = BatchNorm2d(out_channels[0])
        self.Relu2 = F.ReLu()
        self.conv3 = Conv2D(input_channels=out_channels[0], output_channels=1, kernel_size=(3, 3),
                            stride=(1, 1), padding="same")
        self.Sigmoid = F.Sigmoid()

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)

        out5 = self.u1(out4)
        cat_out5 = tensor.concatenate([out5, out_4], axis=1)
        out6 = self.u2(cat_out5)
        cat_out6 = tensor.concatenate([out6, out_3], axis=1)
        out7 = self.u3(cat_out6)
        cat_out7 = tensor.concatenate([out7, out_2], axis=1)
        out8 = self.u4(cat_out7)
        cat_out8 = tensor.concatenate([out8, out_1], axis=1)
        out = self.conv1(cat_out8)
        out = self.BatchNorm2d1(out)
        out = self.Relu1(out)
        out = self.conv2(out)
        out = self.BatchNorm2d2(out)
        out = self.Relu2(out)
        out = self.conv3(out)
        out = self.Sigmoid(out)
        return out

class MyDataset():
    def __init__(self, x_data, x_label):
        self.x_set = x_data
        self.label = x_label

    def __getitem__(self, item):
        img, target = self.x_set[item], self.label[item]
        img_np = np.uint8(img).transpose(2, 0, 1)
        target_np = np.uint8(target).transpose(2, 0, 1)

        img = img_np
        target = target_np
        return img, target

    def __len__(self):
        return len(self.x_set)

if not os.path.exists("./result"):
    os.makedirs("./result")
else:
    pass
if not os.path.exists("./Intermediate_results"):
    os.makedirs("./Intermediate_results")
else:
    pass

# prepare train/test data and label
path0 = './data/dataset/Unet_data_src'
path1 = './data/dataset/Unet_data_test'
train_images, train_labels = PreprocessingData(path0).read()
test_images, test_labels = PreprocessingData(path1).read()

print('train: ', train_images.shape, '\ntest: ', test_images.shape)
print('train: ', train_labels.shape, '\ntest: ', test_labels.shape)
train_images = train_images / 255
test_images = test_images / 255

# use quantum encoder to preprocess data
PREPROCESS = True
# PREPROCESS = False

if PREPROCESS == True:
    print("Quantum pre-processing of train images:")
    q_train_images = QuantumDataPreprocessing(train_images)
    q_test_images = QuantumDataPreprocessing(test_images)
    q_train_label = QuantumDataPreprocessing(train_labels)
    q_test_label = QuantumDataPreprocessing(test_labels)

    # Save pre-processed images
    print('Quantum Data Saving...')
    np.save("./data/dataset/q_train.npy", q_train_images)
    np.save("./data/dataset/q_test.npy", q_test_images)
    np.save("./data/dataset/q_train_label.npy", q_train_label)
    np.save("./data/dataset/q_test_label.npy", q_test_label)
    print('Quantum Data Saving Over!')

# loading quantum data
SAVE_PATH = "./data/dataset/"
train_x = np.load(SAVE_PATH + "q_train.npy")
train_labels = np.load(SAVE_PATH + "q_train_label.npy")
test_x = np.load(SAVE_PATH + "q_test.npy")
test_labels = np.load(SAVE_PATH + "q_test_label.npy")

train_x = train_x.astype(np.uint8)
test_x = test_x.astype(np.uint8)
train_labels = train_labels.astype(np.uint8)
test_labels = test_labels.astype(np.uint8)
train_y = train_labels
test_y = test_labels

trainset = MyDataset(train_x, train_y)
testset = MyDataset(test_x, test_y)

x_train = []
y_label = []
model = UNet()
optimizer = Adam(model.parameters(), lr=0.01)
loss_func = BinaryCrossEntropy()
epochs = 200

loss_list = []
SAVE_FLAG = True
temp_loss = 0
file = open("./result/result.txt", 'w').close()
for epoch in range(1, epochs):
    total_loss = []
    model.train()
    for i, (x, y) in enumerate(trainset):
        x_img = QTensor(x)
        x_img_Qtensor = tensor.unsqueeze(x_img, 0)
        y_img = QTensor(y)
        y_img_Qtensor = tensor.unsqueeze(y_img, 0)
        optimizer.zero_grad()
        img_out = model(x_img_Qtensor)

        print(f"=========={epoch}==================")
        loss = loss_func(y_img_Qtensor, img_out)  # target output
        if i == 1:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title("predict")
            img_out_tensor = tensor.squeeze(img_out, 0)

            if matplotlib.__version__ >= '3.4.2':
                plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
            else:
                plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
            plt.subplot(1, 2, 2)
            plt.title("label")
            y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
            if matplotlib.__version__ >= '3.4.2':
                plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
            else:
                plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))

            plt.savefig("./Intermediate_results/" + str(epoch) + "_" + str(i) + ".jpg")

        loss_data = np.array(loss.data)
        print("{} - {} loss_data: {}".format(epoch, i, loss_data))
        loss.backward()
        optimizer._step()
        total_loss.append(loss_data)

    loss_list.append(np.sum(total_loss) / len(total_loss))
    out_read = open("./result/result.txt", 'a')
    out_read.write(str(loss_list[-1]))
    out_read.write(str("\n"))
    out_read.close()
    print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
    if SAVE_FLAG:
        temp_loss = loss_list[-1]
        save_parameters(model.state_dict(), "./result/Q-Unet_End.model")
        SAVE_FLAG = False
    else:
        if temp_loss > loss_list[-1]:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "./result/Q-Unet_End.model")

out_read = open("./result/result.txt", 'r')
plt.figure()
lines_read = out_read.readlines()
data_read = []
for line in lines_read:
    float_line = float(line)
    data_read.append(float_line)
out_read.close()
plt.plot(data_read)
plt.title('Unet Training')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.savefig("./result/traing_loss.jpg")

modela = load_parameters("./result/Q-Unet_End.model")
print("----------------PREDICT-------------")
model.train()

for i, (x1, y1) in enumerate(testset):
    x_img = QTensor(x1)
    x_img_Qtensor = tensor.unsqueeze(x_img, 0)
    y_img = QTensor(y1)
    y_img_Qtensor = tensor.unsqueeze(y_img, 0)
    img_out = model(x_img_Qtensor)
    loss = loss_func(y_img_Qtensor, img_out)
    loss_data = np.array(loss.data)
    print("{} loss_eval: {}".format(i, loss_data))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("predict")
    img_out_tensor = tensor.squeeze(img_out, 0)
    if matplotlib.__version__ >= '3.4.2':
        plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
    else:
        plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
    plt.subplot(1, 2, 2)
    plt.title("label")
    y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
    if matplotlib.__version__ >= '3.4.2':
        plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
    else:
        plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))
    plt.savefig("./result/" + str(i) + "_1" + ".jpg")
print("end!")




