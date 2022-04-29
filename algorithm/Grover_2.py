#!/usr/bin/env python

import pyqpanda as pq
import numpy as np

if __name__ == "__main__":

   machine = pq.init_quantum_machine(pq.QMachineType.CPU)
   x = machine.cAlloc()
   prog = pq.create_empty_qprog()

   data=[3, 6, 6, 9, 10, 15, 11, 6]
   grover_result = pq.Grover_search(data, x==6, machine, 1)

   print(grover_result[1])