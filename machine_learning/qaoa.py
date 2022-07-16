from pyqpanda import *
import numpy as np

def oneCircuit(qlist, Hamiltonian, beta, gamma):
    vqc=VariationalQuantumCircuit()
    for i in range(len(Hamiltonian)):
        tmp_vec=[]
        item=Hamiltonian[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z'!= dict_p[iter]:
                pass
            tmp_vec.append(qlist[iter])

        coef = item[1]

        if 2 != len(tmp_vec):
            pass

        vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
        vqc.insert(VariationalQuantumGate_RZ(tmp_vec[1], 2*gamma*coef))
        vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))

    for j in qlist:
        vqc.insert(VariationalQuantumGate_RX(j,2.0*beta))
    return vqc


if __name__=="__main__":
    problem = {'Z0 Z4':0.73,'Z0 Z5':0.33,'Z0 Z6':0.5,'Z1 Z4':0.69,'Z1 Z5':0.36,
           'Z2 Z5':0.88,'Z2 Z6':0.58,'Z3 Z5':0.67,'Z3 Z6':0.43}
    Hp = PauliOperator(problem)
    qubit_num = Hp.getMaxIndex()

    machine=init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(qubit_num)

    step = 4

    beta = var(np.ones((step,1),dtype = 'float64'), True)
    gamma = var(np.ones((step,1),dtype = 'float64'), True)

    vqc=VariationalQuantumCircuit()

    for i in qlist:
        vqc.insert(VariationalQuantumGate_H(i))

    for i in range(step):
        vqc.insert(oneCircuit(qlist,Hp.toHamiltonian(1),beta[i], gamma[i]))


    loss = qop(vqc, Hp, machine, qlist)
    optimizer = MomentumOptimizer.minimize(loss, 0.02, 0.9)

    leaves = optimizer.get_variables()

    for i in range(100):
        optimizer.run(leaves, 0)
        loss_value = optimizer.get_loss()
        print("i: ", i, " loss:",loss_value )

    # 验证结果
    prog = QProg()
    qcir = vqc.feed()
    prog.insert(qcir)
    directly_run(prog)

    result = quick_measure(qlist, 100)
    print(result)