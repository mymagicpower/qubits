import pyqpanda as pq
import numpy as np

if __name__ == "__main__":
   A=[1,0,0,1]
   b=[0.6,0.8]
   result = pq.HHL_solve_linear_equations(A,b,1)

   #打印测量结果
   for key in result:
      print(key)