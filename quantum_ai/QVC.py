
from pyvqnet.nn.module import Module
from pyvqnet.optim.sgd import SGD
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.tensor.tensor import QTensor
import pyqpanda as pq
from pyvqnet.qnn.quantumlayer import QuantumLayer

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.qvc = QuantumLayer(qvc_circuits,24,"cpu",4)

    def forward(self, x):
        return self.qvc(x)

def qvc_circuits(input,weights,qlist,clist,machine):

    def get_cnot(nqubits):
        cir = pq.QCircuit()
        for i in range(len(nqubits)-1):
            cir.insert(pq.CNOT(nqubits[i],nqubits[i+1]))
        cir.insert(pq.CNOT(nqubits[len(nqubits)-1],nqubits[0]))
        return cir

    def build_circult(weights, xx, nqubits):

        def Rot(weights_j, qubits):
            circult = pq.QCircuit()
            circult.insert(pq.RZ(qubits, weights_j[0]))
            circult.insert(pq.RY(qubits, weights_j[1]))
            circult.insert(pq.RZ(qubits, weights_j[2]))
            return circult
        def basisstate():
            circult = pq.QCircuit()
            for i in range(len(nqubits)):
                if xx[i] == 1:
                    circult.insert(pq.X(nqubits[i]))
            return circult

        circult = pq.QCircuit()
        circult.insert(basisstate())

        for i in range(weights.shape[0]):

            weights_i = weights[i,:,:]
            for j in range(len(nqubits)):
                weights_j = weights_i[j]
                circult.insert(Rot(weights_j,nqubits[j]))
            cnots = get_cnot(nqubits)
            circult.insert(cnots)

        circult.insert(pq.Z(nqubits[0]))

        prog = pq.QProg()
        prog.insert(circult)
        return prog

    weights = weights.reshape([2,4,3])
    prog = build_circult(weights,input,qlist)
    prob = machine.prob_run_dict(prog, qlist[0], -1)
    prob = list(prob.values())

    return prob


import numpy as np
import os
qvc_train_data = [0,1,0,0,1,
0, 1, 0, 1, 0,
0, 1, 1, 0, 0,
0, 1, 1, 1, 1,
1, 0, 0, 0, 1,
1, 0, 0, 1, 0,
1, 0, 1, 0, 0,
1, 0, 1, 1, 1,
1, 1, 0, 0, 0,
1, 1, 0, 1, 1,
1, 1, 1, 0, 1,
1, 1, 1, 1, 0]
qvc_test_data= [0, 0, 0, 0, 0,
0, 0, 0, 1, 1,
0, 0, 1, 0, 1,
0, 0, 1, 1, 0]
def dataloader(data,label,batch_size, shuffle = True)->np:
    if shuffle:
        for _ in range(len(data)//batch_size):
            random_index = np.random.randint(0, len(data), (batch_size, 1))
            yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
    else:
        for i in range(0,len(data)-batch_size+1,batch_size):
            yield data[i:i+batch_size], label[i:i+batch_size]

def get_data(dataset_str):
    if dataset_str == "train":
        datasets = np.array(qvc_train_data)

    else:
        datasets = np.array(qvc_test_data)

    datasets = datasets.reshape([-1,5])
    data = datasets[:,:-1]
    label = datasets[:,-1].astype(int)
    label = np.eye(2)[label].reshape(-1,2)
    return data, label

def get_accuary(result,label):
    result,label = np.array(result.data), np.array(label.data)
    score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
    return score

#示例化Model类
model = Model()
#定义优化器，此处需要传入model.parameters()表示模型中所有待训练参数，lr为学习率
optimizer = SGD(model.parameters(),lr =0.1)
#训练时候可以修改批处理的样本数
batch_size = 3
#训练最大迭代次数
epoch = 20
#模型损失函数
loss = CategoricalCrossEntropy()

model.train()
datas,labels = get_data("train")

for i in range(epoch):
    count=0
    sum_loss = 0
    accuary = 0
    t = 0
    for data,label in dataloader(datas,labels,batch_size,False):
        optimizer.zero_grad()
        data,label = QTensor(data), QTensor(label)

        result = model(data)

        loss_b = loss(label,result)
        loss_b.backward()
        optimizer._step()
        sum_loss += loss_b.item()
        count+=batch_size
        accuary += get_accuary(result,label)
        t = t + 1

    print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")

model.eval()
count = 0
test_data,test_label = get_data("test")
test_batch_size = 1
accuary = 0
sum_loss = 0
for testd,testl in dataloader(test_data,test_label,test_batch_size):
    testd = QTensor(testd)
    test_result = model(testd)
    test_loss = loss(testl,test_result)
    sum_loss += test_loss
    count+=test_batch_size
    accuary += get_accuary(test_result,testl)
print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")