from pyqpanda import *
if __name__=="__main__":
    a = FermionOperator('1+ 0', 2)
    b = FermionOperator('3+ 2', 3)
    plus=a+b
    minus=a-b
    multiply=a*b
    print("a + b = {}".format(plus))
    print("a - b = {}".format(minus))
    print("a * b = {}".format(multiply))