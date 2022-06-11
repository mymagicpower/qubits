from pyqpanda import *
if __name__=="__main__":
    p1 = PauliOperator()
    p2 = PauliOperator({'Z0 Z1': 2, 'X1 Y2': 3})
    p3 = PauliOperator('Z0 Z1', 2)
    p4 = PauliOperator(2)
    p5 = p2