from pyqpanda import *

if __name__=="__main__":
    a = PauliOperator('Z0 Z1', 2)
    b = PauliOperator('X5 X6', 3)
    plus = a + b
    minus = a - b
    multiply = a * b
    print("a + b = {}".format(plus))
    print("a - b = {}".format(minus))
    print("a * b = {}".format(multiply))