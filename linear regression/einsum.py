import numpy as np

# transpose
A = np.array([[1,2,3], [4,5,6]])
T = np.einsum("ij->ji", A)
print(T)

# diagonal, trace
A = np.identity(5)
print(A)
diag = np.einsum('ii->i', A)
trace =np.einsum('ii->', A)
print(diag)
print(trace)

# sum all element.
A = np.array([[1,2,3], [4,5,6]])
R = np.einsum("ij->", A)
print(R)

#dot product
x = np.array([[1,2],[4,5]])
y = np.array([[4],[6]])
dot1=np.dot(x,y)
print(dot1)

dot = np.einsum('ij,jk->ik', x, y )
print(dot)
