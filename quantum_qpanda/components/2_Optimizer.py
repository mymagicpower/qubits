from pyqpanda import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
             2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27,3.1])
y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
             1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94,1.3])

def lossFunc(para,grad,inter,fcall):
    y_ = np.zeros(len(y))

    for i in range(len(y)):
        y_[i] = para[0] * x[i] + para[1]

    loss = 0
    for i in range(len(y)):
        loss += (y_[i] - y[i])**2/len(y)

    return ("", loss)

optimizer = OptimizerFactory.makeOptimizer('NELDER_MEAD')

init_para = [0, 0]
optimizer.registerFunc(lossFunc, init_para)
optimizer.setXatol(1e-6)
optimizer.setFatol(1e-6)
optimizer.setMaxIter(200)
optimizer.exec()

result = optimizer.getResult()
print(result.message)
print(" Current function value: ", result.fun_val)
print(" Iterations: ", result.iters)
print(" Function evaluations: ", result.fcalls)
print(" Optimized para: W: ", result.para[0], " b: ", result.para[1])

w = result.para[0]
b = result.para[1]

plt.plot(x, y, 'o', label = 'Training data')
plt.plot(x, w*x + b, 'r', label = 'Fitted line')
plt.legend()
plt.show()