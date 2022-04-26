from pyqpanda import *

if __name__=="__main__":

    a = PauliOperator("Z0 Z1", 2)
    b = PauliOperator("X5 Y6", 3)

    plus = a + b
    minus = a - b
    muliply = a * b

    print("a + b = ", plus)
    print("a - b = ", minus)
    print("a * b = ", muliply)

    print("Index : ", muliply.getMaxIndex())

    index_map = {}
    remap_pauli = muliply.remapQubitIndex(index_map)

    print("remap_pauli : ", remap_pauli)
    print("Index : ", remap_pauli.getMaxIndex())