from pyqpanda import *
import numpy as np

if __name__=="__main__":

    m1 = np.array([[1., 2.],[3., 4.]])
    v1 = var(m1)

    m2 = np.array([[5., 6.],[7., 8.]])
    v2 = var(m2)

    sum = v1 + v2
    minus = v1 - v2
    multiply = v1 * v2

    print("v1: ", v1.get_value())
    print("v2: ", v2.get_value())
    print("sum: " , eval(sum))
    print("minus: " , eval(minus))
    print("multiply: " , eval(multiply))

    m3 = np.array([[4., 3.],[2., 1.]])
    v1.set_value(m3)

    print("sum: " , eval(sum))
    print("minus: " , eval(minus))
    print("multiply: " , eval(multiply))