from pyqpanda import *

if __name__ == "__main__":
    init(QMachineType.CPU)
    qubits = qAlloc_many(2)
    cbits = cAlloc_many(2)

    prog = QProg()
    prog << H(qubits[0])\
        << CNOT(qubits[0], qubits[1])

    print("prob_run_dict: ")
    result1 = prob_run_dict(prog, qubits, -1)
    print(result1)

    print("prob_run_tuple_list: ")
    result2 = prob_run_tuple_list(prog, qubits, -1)
    print(result2)

    print("prob_run_list: ")
    result3 = prob_run_list(prog, qubits, -1)
    print(result3)

    finalize()