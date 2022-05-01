import pyqpanda as pq

if __name__ == "__main__":

    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    qubits = machine.qAlloc_many(3)
    prog = pq.create_empty_qprog()

    # 构建量子程序
    prog.insert(pq.H(qubits[0])) \
        .insert(pq.H(qubits[1])) \
        .insert(pq.H(qubits[2]))

    # 对量子程序进行概率测量
    result = pq.prob_run_dict(prog, qubits, -1)
    pq.destroy_quantum_machine(machine)

    # 打印测量结果
    for key in result:
        print(key+":"+str(result[key]))