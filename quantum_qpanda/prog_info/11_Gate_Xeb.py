from pyqpanda import *

if __name__=="__main__":

    # 构建噪声虚拟机，调整噪声模拟真实芯片
    qvm = NoiseQVM()
    qvm.init_qvm()
    qv = qvm.qAlloc_many(4)

    # 设置噪声参数
    qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.CZ_GATE, 0.1)

    # 同样可以申请云计算机器（采用真实芯片）
    # qvm =  QCloud()
    # qvm.init_qvm("898D47CF515A48CEAA9F2326394B85C6")

    # 设置不同层数组合
    range = [2,4,6,8,10]
    # 现在可测试双门类型主要为CZ CNOT SWAP ISWAP SQISWAP
    res = double_gate_xeb(qvm, qv[0], qv[1], range, 10, 1000, GateType.CZ_GATE)
    # 对应的数值随噪声影响，噪声数值越大，所得结果越小，且层数增多，结果数值越小。

    print(res)

    qvm.finalize()