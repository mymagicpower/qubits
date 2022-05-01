import pyqpanda as pq
from numpy import pi

if __name__ == "__main__":

    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    qvec = machine.qAlloc_many(3)
    prog = pq.create_empty_qprog()

    # 构建量子程序
    prog.insert(pq.QFT(qvec))

    # 对量子程序进行概率测量
    result = pq.prob_run_dict(prog, qvec, -1)
    pq.destroy_quantum_machine(machine)

    # 打印测量结果
    for key in result:
         print(key+":"+str(result[key]))