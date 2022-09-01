import pyqpanda as pq
from numpy import pi

if __name__ == "__main__":

   machine = pq.init_quantum_machine(pq.QMachineType.CPU)
   qvec = machine.qAlloc_many(1)
   cqv = machine.qAlloc_many(4)
   prog = pq.create_empty_qprog()

   # 构建量子程序
   prog.insert(pq.H(cqv[0]))\
       .insert(pq.H(cqv[1]))\
       .insert(pq.H(cqv[2]))\
       .insert(pq.H(cqv[3]))\
       .insert(pq.H(qvec[0]))\
       .insert(pq.S(qvec[0]))\
       .insert(pq.RY(qvec[0], pi/4).control(cqv[0]))\
       .insert(pq.RY(qvec[0], pi/2).control(cqv[1]))\
       .insert(pq.RY(qvec[0], pi).control(cqv[2]))\
       .insert(pq.RY(qvec[0], pi*2).control(cqv[3])) \
       .insert(pq.QFT(cqv).dagger())

   # 对量子程序进行概率测量
   result = pq.prob_run_dict(prog, cqv, -1)
   pq.destroy_quantum_machine(machine)

   # 打印测量结果
   for key in result:
       print(key+":"+str(result[key]))