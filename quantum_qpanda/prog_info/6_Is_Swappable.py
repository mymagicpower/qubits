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

#测试接口： 判断指定的两个逻辑门是否可以交换位置
def is_swappable(q, c):
    prog = pq.QProg()
    cir = pq.QCircuit()
    cir2 = pq.QCircuit()
    cir2 << pq.H(q[3]) << pq.RX(q[1], math.pi/2) << pq.T(q[2]) << pq.RY(q[3], math.pi/2) << pq.RZ(q[2], math.pi/2)
    cir2.set_dagger(True)
    cir << pq.H(q[1]) << cir2 << pq.CR(q[1], q[2], math.pi/2)
    prog << pq.H(q[0]) << pq.S(q[2]) \
    << cir\
    << pq.CNOT(q[0], q[1]) << pq.CZ(q[1], q[2]) << pq.measure_all(q,c)

    iter_first = cir.begin()

    iter_second = cir2.begin()
    #iter_second = iter_second.get_next()
    #iter_second = iter_second.get_next()
    #iter_second = iter_second.get_next()

    type =iter_first.get_node_type()
    if pq.NodeType.GATE_NODE == type:
        gate = pq.QGate(iter_first)
        print(gate.gate_type())

    type =iter_second.get_node_type()
    if pq.NodeType.GATE_NODE == type:
        gate = pq.QGate(iter_second)
        print(gate.gate_type())

    if (pq.is_swappable(prog, iter_first, iter_second)) == True:
        print('Could be swapped !\n')
    else:
        print('Could NOT be swapped.')

if __name__=="__main__":
    init_machine = InitQMachine(16, 16)
    qlist = init_machine.m_qlist
    clist = init_machine.m_clist
    machine = init_machine.m_machine

    is_swappable(qlist, clist)
    print("Test over.")