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

def is_match_topology(q, c):
    cx = pq.CNOT(q[1], q[3])

    # 构建拓扑结构
    qubits_topology = [[0,1,0,0,0],[1,0,1,1,0],[0,1,0,0,0],[0,1,0,0,1],[0,0,0,1,0]]

    #判断逻辑门是否符合量子拓扑结构
    if (pq.is_match_topology(cx,qubits_topology)) == True:
        print('Match !\n')
    else:
        print('Not match.')

if __name__=="__main__":
    init_machine = InitQMachine(16, 16)
    qlist = init_machine.m_qlist
    clist = init_machine.m_clist
    machine = init_machine.m_machine
    is_match_topology(qlist, clist)
    print("Test over.")