from pyqpanda import *

if __name__ == "__main__":
    init(QMachineType.CPU)
    qubits = qAlloc_many(4)
    cbits = cAlloc_many(4)

    # 构建量子程序
    prog = QProg()
    prog << H(qubits[0])\
         << H(qubits[1])\
         << H(qubits[2])\
         << H(qubits[3])\
         << measure_all(qubits, cbits)

    # 量子程序运行1000次，并返回测量结果
    result = run_with_configuration(prog, cbits, 1000)

    # 打印测量结果
    print(result)
    finalize()