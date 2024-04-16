import numpy as np
from scipy import linalg

# Method 1 : By using Determinants. This method is not very efficient. Hardcoding Cramer's rule.
def getPolynomial(listOfPoints):
    n = len(listOfPoints)
    A = []
    Y = []
    coeffs = []
    print(listOfPoints)
    for i in listOfPoints:
        A.append([1,i[0]])
        Y.append(i[1])
    for i in A:
        m = i[1]
        for j in range(n-2):
            m = m * i[1]
            i.append(m)
    
    for i in range(n):
        for j in range(i, n):
            (A[i][j], A[j][i]) = (A[j][i], A[i][j])
    
    
    row = [0] * n
    delta = np.linalg.det(A)
    for i in range(n):
        row = A[i]
        A[i] = Y
        det = np.linalg.det(A)
        
        A[i] = row
        coeffs.append(det / delta)
        
    return coeffs


# Method 2 : By using LU Decomposition. Efficient.
def getPolynomial2(listOfPoints):
    n = len(listOfPoints)
    A = []
    Y = []
    coeffs = []
    print(listOfPoints)
    for i in listOfPoints:
        A.append([1,i[0]])
        Y.append(i[1])
    for i in A:
        m = i[1]
        for j in range(n-2):
            m = m * i[1]
            i.append(m)
    print(A,Y) 
    coeffs = linalg.solve(A,Y)

    return coeffs


# Testing the function with a set
print(getPolynomial2([[2,17],[7,82],[-5,10],[-4,5]]))
print(getPolynomial([[2,17],[7,82],[-5,10],[-4,5]]))