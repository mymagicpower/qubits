import pyqpanda.pyQPanda as pq

class InitQMachine:
    def __init__(self, quBitCnt, cBitCnt, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_qlist = self.m_machine.qAlloc_many(quBitCnt)
        self.m_clist = self.m_machine.cAlloc_many(cBitCnt)
        self.m_prog = pq.QProg()
    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def support_qgate_type():
    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    q = machine.qAlloc_many(8)
    prog = pq.QProg()
    prog << pq.H(q[1])
    result = pq.is_supported_qgate_type(prog.begin())
    if result == True:
        print('Support !\n')
    else:
        print('Unsupport !')

if __name__=="__main__":
    init_machine = InitQMachine(16, 16)
    qlist = init_machine.m_qlist
    clist = init_machine.m_clist
    machine = init_machine.m_machine
    support_qgate_type()
    print("Test over.")