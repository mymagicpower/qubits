from pyqpanda import *
if __name__=="__main__":
    machine = init_quantum_machine(QMachineType.CPU)
    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)
    # 构建量子程序
    prog = QProg()
    prog << H(q[0])\
        << H(q[2])\
        << H(q[3])\
        << CNOT(q[1], q[0])\
        << H(q[0])\
        << CNOT(q[1], q[2])\
        << H(q[2])\
        << CNOT(q[2], q[3])\
        << H(q[3])
    # 构建查询线路
    query_cir = QCircuit()
    query_cir << H(q[0])\
            << CNOT(q[1], q[0])\
            << H(q[0])
    # 构建替换线路
    replace_cir = QCircuit()
    replace_cir << CZ(q[1], q[0])
    print("查询替换前：")
    print(convert_qprog_to_originir(prog,machine))
    # 搜索量子程序中的查询线路，并用替换线路替代
    update_prog = circuit_optimizer(prog, [[query_cir, replace_cir]])
    print("查询替换后：")
    print(convert_qprog_to_originir(update_prog,machine))