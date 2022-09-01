from pyqpanda import *
if __name__=="__main__":
    a = PauliOperator('Z0 Z1', 2)
    b = PauliOperator('X5 X6', 3)
    muliply = a * b
    print("a * b = {}".format(muliply))
    print("Index : {}".format(muliply.getMaxIndex()))
