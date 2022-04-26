import pyqpanda.pyQPanda as pq
import math
class InitQMachine:
    def __init__(self, quBitCnt, cBitCnt, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_qlist = self.m_machine.qAlloc_many(quBitCnt)
        self.m_clist = self.m_machine.cAlloc_many(cBitCnt)
        self.m_prog = pq.QProg()
    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def get_matrix(q, c):
    prog = pq.QProg()
    # 构建量子程序
    prog << pq.H(q[0]) \
        << pq.S(q[2]) \
        << pq.CNOT(q[0], q[1]) \
        << pq.CZ(q[1], q[2]) \
        << pq.CR(q[1], q[2], math.pi/2)
    # 获取线路对应矩阵
    result_mat = pq.get_matrix(prog)
    # 打印矩阵信息
    pq.print_matrix(result_mat)

if __name__=="__main__":
    init_machine = InitQMachine(16, 16)
    qlist = init_machine.m_qlist
    clist = init_machine.m_clist
    machine = init_machine.m_machine
    get_matrix(qlist, clist)
    print("Test over.")