from pyqpanda import *
if __name__ == "__main__":
    init(QMachineType.CPU)
    qubits = qAlloc_many(1)
    cbits = cAlloc_many(1)
    # 构建量子程序
    prog = QProg()
    prog << Z(qubits[0]) \
         << Measure(qubits[0], cbits[0])
    # 量子程序运行1000次，并返回测量结果
    result = run_with_configuration(prog, cbits, 1000)
    print(result)
    finalize()