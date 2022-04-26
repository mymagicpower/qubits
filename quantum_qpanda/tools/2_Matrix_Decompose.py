import pyqpanda as pq
import numpy as np

if __name__=="__main__":

    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    source_matrix = [(0.6477054522122977+0.1195417767870219j), (-0.16162176706189357-0.4020495632468249j), (-0.19991615329121998-0.3764618308248643j), (-0.2599957197928922-0.35935248873007863j),
                    (-0.16162176706189363-0.40204956324682495j), (0.7303014482204584-0.4215172444390785j), (-0.15199187936216693+0.09733585496768032j), (-0.22248203136345918-0.1383600597660744j),
                    (-0.19991615329122003-0.3764618308248644j), (-0.15199187936216688+0.09733585496768032j), (0.6826630277354306-0.37517063774206166j), (-0.3078966462928956-0.2900897445133085j),
                    (-0.2599957197928923-0.3593524887300787j), (-0.22248203136345912-0.1383600597660744j), (-0.30789664629289554-0.2900897445133085j), (0.6640994547408099-0.338593803336005j)]

    print("source matrix : ")
    print(source_matrix)

    out_cir = pq.matrix_decompose(q, source_matrix)
    circuit_matrix = pq.get_matrix(out_cir)

    print("the decomposed matrix : ")
    print(circuit_matrix)

    source_matrix = np.round(np.array(source_matrix),3)
    circuit_matrix = np.round(np.array(circuit_matrix),3)

    if np.all(source_matrix == circuit_matrix):
        print('matrix decompose ok !')
    else:
        print('matrix decompose false !')