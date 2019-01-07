import numpy as np
#69
A = np.array([[1,2],[3,4]])
B = np.array([[6,7],[8,9]])
print(np.dot(A,B))
print("")
print(np.diag(np.dot(A,B)))
print("")

#72
A = np.array([[1,2],[3,4],[5,6]])
A[[0,2]] = A[[2,0]]
print(A)
print("")

#82
r = np.linalg.matrix_rank(A)
print(r)
print("")

#83
X = np.random.randint(10, size=10)
print(X)
print("")
print(np.bincount(X))
print("")
f = np.argmax(np.bincount(X))
print(f)
print("")

#89
X = np.random.randint(10, size=10)
m = np.amax(X)
print(m)
print("")
