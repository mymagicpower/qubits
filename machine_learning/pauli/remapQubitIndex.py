from pyqpanda import *
if __name__=="__main__":
    a = PauliOperator('Z0 Z1', 2)
    b = PauliOperator('X5 X6', 3)
    muliply = a * b
    index_map = {}
    remap_pauli = muliply.remapQubitIndex(index_map)
    print("remap_pauli = {}".format(remap_pauli))
    print("Index : {}".format(remap_pauli.getMaxIndex()))