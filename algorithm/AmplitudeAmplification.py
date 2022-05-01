import pyqpanda as pq
from numpy import pi

if __name__ == "__main__":

    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    qvec = machine.qAlloc_many(1)
    prog = pq.create_empty_qprog()

    # 构建量子程序
    prog.insert(pq.RY(qvec[0], pi/3))
    prog.insert(pq.Z(qvec[0]))
    prog.insert(pq.RY(qvec[0], pi*4/3))

    # 对量子程序进行概率测量
    result = pq.prob_run_dict(prog, qvec, -1)
    pq.destroy_quantum_machine(machine)

    # 打印测量结果
    for key in result:
         print(key+":"+str(result[key]))