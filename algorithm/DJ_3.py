from pyqpanda import *
import numpy as np
if __name__ == "__main__":
    init(QMachineType.CPU)
    qubits = qAlloc_many(2)
    cbits = cAlloc_many(2)
    # 构建量子程序
    prog = QProg()
    prog << H(qubits[0]) \
         << X(qubits[1])<< H(qubits[1]) \
         << CNOT(qubits[1], qubits[0]) \
         << X(qubits[0]) << H(qubits[0])\
         << H(qubits[1]) \
         << measure_all(qubits, cbits)

    # 量子程序运行1000次，并返回测量结果
    result = run_with_configuration(prog, cbits, 1000)
    print(result)
    finalize()