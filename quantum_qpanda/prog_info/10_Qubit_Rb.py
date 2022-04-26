from pyqpanda import *

if __name__=="__main__":
    # 构建噪声虚拟机，调整噪声模拟真实芯片
    qvm = NoiseQVM()
    qvm.init_qvm()
    qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.CZ_GATE, 0.005)
    qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.PAULI_Y_GATE, 0.005)
    qv = qvm.qAlloc_many(4)

    # 同样可以申请云计算机器（采用真实芯片）
    # qvm =  QCloud()
    # qvm.init_qvm("898D47CF515A48CEAA9F2326394B85C6")

    # 设置随机线路中clifford门集数量
    range = [ 5,10,15 ]

    # 测量单比特随机基准
    res = single_qubit_rb(qvm, qv[0], range, 10, 1000)

    # 同样可以测量两比特随机基准
    #res = double_qubit_rb(qvm, qv[0], qv[1], range, 10, 1000)

    # 对应的数值随噪声影响，噪声数值越大，所得结果越小，且随clifford门集数量增多，结果数值越小。
    print(res)

    qvm.finalize()