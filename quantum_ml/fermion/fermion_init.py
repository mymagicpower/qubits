from pyqpanda import *
if __name__=="__main__":
    p1 = FermionOperator()
    p2 = FermionOperator({'1+ 0': 2,'3+ 2+ 1 0': 3})
    p3 = FermionOperator('1+ 0', 2)
    p4 = FermionOperator(2)
    p5=p2