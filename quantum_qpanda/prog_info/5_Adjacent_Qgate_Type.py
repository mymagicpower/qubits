import pyqpanda.pyQPanda as pq

class InitQMachine:
    def __init__(self, quBitCnt, cBitCnt, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_qlist = self.m_machine.qAlloc_many(quBitCnt)
        self.m_clist = self.m_machine.cAlloc_many(cBitCnt)
        self.m_prog = pq.QProg()

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def get_adjacent_qgate_type(qlist, clist):
    prog = pq.QProg()

    # 构建量子程序
    prog << pq.T(qlist[0]) \
        << pq.CNOT(qlist[1], qlist[2]) \
        << pq.Reset(qlist[1]) \
        << pq.H(qlist[3]) \
        << pq.H(qlist[4])

    iter = prog.begin()
    iter = iter.get_next()
    type =iter.get_node_type()
    if pq.NodeType.GATE_NODE == type:
        gate = pq.QGate(iter)
        print(gate.gate_type())

    # 获取指定位置前后逻辑门类型
    list = pq.get_adjacent_qgate_type(prog,iter)
    print(len(list))
    print(len(list[0].m_target_qubits))
    print(list[1].m_is_dagger)

    node_type = list[0].m_node_type
    print(node_type)
    if node_type == pq.NodeType.GATE_NODE:
        gateFront = pq.QGate(list[0].m_iter)
        print(gateFront.gate_type())

    node_type = list[1].m_node_type
    print(node_type)
    if node_type == pq.NodeType.GATE_NODE:
        gateBack = pq.QGate(list[1].m_iter)
        print(gateBack.gate_type())

if __name__=="__main__":
    init_machine = InitQMachine(16, 16)
    qlist = init_machine.m_qlist
    clist = init_machine.m_clist
    machine = init_machine.m_machine
    get_adjacent_qgate_type(qlist, clist)
    print("Test over.")