from pyqpanda import *
import numpy as np

if __name__ == "__main__":
    qvm = init_quantum_machine(QMachineType.CPU)
    qubits = qvm.qAlloc_many(4)
    cbits = qvm.cAlloc_many(4)
    # 构建量子程序
    prog = QProg()
    prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])\
         << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], np.pi / 4)

    # 统计量子程序时钟周期
    clock_cycle = get_qprog_clock_cycle(prog, qvm)
    print(clock_cycle)
    destroy_quantum_machine(qvm)