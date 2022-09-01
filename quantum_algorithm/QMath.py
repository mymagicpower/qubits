import pyqpanda as pq
# from numpy import pi

if __name__ == "__main__":
    # 为了节约比特数，辅助比特将会互相借用
    qvm = pq.init_quantum_machine(pq.QMachineType.CPU)

    qdivvec = qvm.qAlloc_many(10)
    qmulvec = qdivvec[:7]
    qsubvec = qmulvec[:-1]
    qvec1 = qvm.qAlloc_many(4)
    qvec2 = qvm.qAlloc_many(4)
    qvec3 = qvm.qAlloc_many(4)
    cbit = qvm.cAlloc()
    prog = pq.create_empty_qprog()

    # (4/1+1-3)*5=10
    prog.insert(pq.bind_data(4,qvec3)) \
       .insert(pq.bind_data(1,qvec2)) \
       .insert(pq.QDivider(qvec3, qvec2, qvec1, qdivvec, cbit)) \
       .insert(pq.bind_data(1,qvec2)) \
       .insert(pq.bind_data(1,qvec2)) \
       .insert(pq.QAdd(qvec1, qvec2, qsubvec)) \
       .insert(pq.bind_data(1,qvec2)) \
       .insert(pq.bind_data(3,qvec2)) \
       .insert(pq.QSub(qvec1, qvec2, qsubvec)) \
       .insert(pq.bind_data(3,qvec2)) \
       .insert(pq.bind_data(5,qvec2)) \
       .insert(pq.QMul(qvec1, qvec2, qvec3, qmulvec)) \
       .insert(pq.bind_data(5,qvec2))

    # 对量子程序进行概率测量
    result = pq.prob_run_dict(prog, qmulvec,1)
    pq.destroy_quantum_machine(qvm)

    # 打印测量结果
    for key in result:
       print(key+":"+str(result[key]))