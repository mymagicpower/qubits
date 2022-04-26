from pyqpanda import *

if __name__=="__main__":

    machine = init_quantum_machine(QMachineType.CPU)
    q = machine.qAlloc_many(2)

    x = var(1)
    y = var(2)


    vqc = VariationalQuantumCircuit()
    vqc.insert(VariationalQuantumGate_H(q[0]))
    vqc.insert(VariationalQuantumGate_RX(q[0], x))
    vqc.insert(VariationalQuantumGate_RY(q[1], y))
    vqc.insert(VariationalQuantumGate_RZ(q[0], 0.12))
    vqc.insert(VariationalQuantumGate_CZ(q[0], q[1]))
    vqc.insert(VariationalQuantumGate_CNOT(q[0], q[1]))

    circuit1 = vqc.feed()

    prog = QProg()
    prog.insert(circuit1)

    print(convert_qprog_to_originir(prog, machine))

    x.set_value([[3.]])
    y.set_value([[4.]])

    circuit2 = vqc.feed()
    prog2 = QProg()
    prog2.insert(circuit2)
    print(convert_qprog_to_originir(prog2, machine))