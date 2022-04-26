from pyqpanda import *

if __name__=="__main__":
    #构建噪声虚拟机，设置噪声参数
    qvm = NoiseQVM()
    qvm.init_qvm()
    qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.CZ_GATE, 0.005)
    #同样可以申请云计算机器（采用真实芯片），采用真实芯片要考虑芯片构造
    #qvm = QCloud()
    #qvm.init_qvm("898D47CF515A48CEAA9F2326394B85C6")

    #构建待测量的量子比特组合， 这里比特组合为2组，其中 量子比特3、4为一组；量子比特2，3，5为一组
    qubit_lists = [[3,4], [2,3,5]]

    #设置随机迭代次数
    ntrials = 100

    #设置测量次数,即真实芯片或者噪声虚拟机shots数值
    shots = 2000
    qv_result = calculate_quantum_volume(qvm, qubit_lists, ntrials, shots)
    print("Quantum Volume : ", qv_result)
    qvm.finalize()