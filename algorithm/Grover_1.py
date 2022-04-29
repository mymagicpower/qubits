from pyqpanda import *

if __name__ == "__main__":
    machine = CPUQVM()
    machine.initQVM()
    x = machine.cAlloc()
    data = [0,3,2,1]
    measure_qubits = QVec()

    # 构建Grover算法量子线路
    grover_cir = Grover(data, x==2, machine, measure_qubits, 1)
    cbits = machine.cAlloc_many(len(measure_qubits))
    prog = QProg()
    prog << grover_cir << measure_all(measure_qubits, cbits)

    # 量子程序运行1000次，并返回测量结果
    result = machine.run_with_configuration(prog, cbits, 1000)
    print(result)
    finalize()