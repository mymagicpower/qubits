import pyqpanda as pq

if __name__ == "__main__":
    N=15
    result = pq.Shor_factorization(N)
    # print(result)
    # 打印测量结果
    for key in result:
        print(key)