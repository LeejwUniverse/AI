import numpy as np
from numpy.linalg import inv

A = np.array([[1,2],[4,4]])
x = np.array([[0],[0]])
b = np.array([[4],[8]])

x=np.dot(inv(A),b)
print(x)
print("")
print(np.dot(A,x))
