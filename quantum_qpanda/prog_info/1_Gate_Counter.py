from pyqpanda import *
if __name__ == "__main__":
    qvm = init_quantum_machine(QMachineType.CPU)
    qubits = qvm.qAlloc_many(2)
    cbits = qvm.cAlloc_many(2)
    prog = QProg()
    # 构建量子程序
    prog << X(qubits[0]) << Y(qubits[1])\
        << H(qubits[0]) << RX(qubits[0], 3.14)\
        << Measure(qubits[0], cbits[0])
    # 统计逻辑门个数
    number = get_qgate_num(prog)
    print("QGate number: " + str(number))
    qvm.finalize()