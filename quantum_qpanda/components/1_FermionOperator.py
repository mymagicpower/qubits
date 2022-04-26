from pyqpanda import *

if __name__=="__main__":

    a = FermionOperator("0 1+", 2)
    b = FermionOperator("2+ 3", 3)

    plus = a + b
    minus = a - b
    muliply = a * b

    print("a + b = ", plus)
    print("a - b = ", minus)
    print("a * b = ", muliply)

    print("normal_ordered(a + b) = ", plus.normal_ordered())
    print("normal_ordered(a - b) = ", minus.normal_ordered())
    print("normal_ordered(a * b) = ", muliply.normal_ordered())