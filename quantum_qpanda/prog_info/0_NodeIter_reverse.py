import pyqpanda.pyQPanda as pq
import math

machine = pq.init_quantum_machine(pq.QMachineType.CPU)
q = machine.qAlloc_many(8)
c = machine.cAlloc_many(8)
prog = pq.QProg()

prog << pq.H(q[0]) << pq.S(q[2]) << pq.CNOT(q[0], q[1]) \
     << pq.CZ(q[1], q[2]) << pq.CR(q[1], q[2], math.pi/2)
iter = prog.begin()
iter_end = prog.end()
while  iter != iter_end:
    if pq.NodeType.GATE_NODE == iter.get_node_type():
        gate = pq.QGate(iter)
        print(gate.gate_type())
    iter = iter.get_next()
else:
    print('Traversal End.\n')

pq.destroy_quantum_machine(machine)
