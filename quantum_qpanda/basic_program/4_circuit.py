from pyqpanda import *
if __name__ == "__main__":
    init(QMachineType.CPU)
    qubits = qAlloc_many(4)
    cbits = cAlloc_many(4)
    # 构建量子程序
    prog = QProg()
    circuit = create_empty_circuit()
    circuit << H(qubits[0]) \
            << CNOT(qubits[0], qubits[1]) \
            << CNOT(qubits[1], qubits[2]) \
            << CNOT(qubits[2], qubits[3])
    prog << circuit << Measure(qubits[0], cbits[0])
    # 量子程序运行1000次，并返回测量结果
    result = run_with_configuration(prog, cbits, 1000)
    # 打印量子态在量子程序多次运行结果中出现的次数
    print(result)
    finalize()