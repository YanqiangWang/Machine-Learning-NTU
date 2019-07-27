import numpy as np

with open("matrixA.txt") as m1, open("matrixB.txt") as m2:
    matrixA = []
    for i in m1:
        row = [int(x) for x in i.split(",")]
        matrixA.append(row)
    
    matrixB = []
    for j in m2:
        row = [int(x) for x in j.split(",")]
        matrixB.append(row)
    
    matrixA = np.array(matrixA)
    matrixB = np.array(matrixB)
    
    ans = matrixA.dot(matrixB)
    ans.sort(axis=1)
    
    np.savetxt("Q1_ans.txt", ans, fmt="%d", delimiter="\r\n")
