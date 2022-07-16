from pyqpanda import *
if __name__=="__main__":
    a = var(2, True)
    b = var(3, True)
    fermion_op = VarFermionOperator('1+ 0', a)
    pauli_op = VarPauliOperator('Z1 Z0', b)