from pyqpanda import *
import numpy as np
if __name__ == "__main__":
    init(QMachineType.CPU)
    qubits = qAlloc_many(2)
    cbits = cAlloc_many(2)
    # 构建量子程序
    prog = QProg()
    prog << X(qubits[0]) \
         << SWAP(qubits[0], qubits[1]) \
         << Measure(qubits[0], cbits[0]) \
         << Measure(qubits[1], cbits[1])
    # 量子程序运行1000次，并返回测量结果
    result = run_with_configuration(prog, cbits, 1000)
    print(result)
    finalize()