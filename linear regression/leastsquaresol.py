import numpy as np
from numpy.linalg import inv

A = np.array([[1,2,4],[3,2,8]])
x = np.array([[0],[0]])
b = np.array([[4],[8]])

A_T_A = np.dot(A.T,A)
print("A_T_A")
print(A_T_A)
print("-------------")
print("Inverse_A_T_A")
print(inv(A_T_A))

x=np.dot(np.dot(inv(A_T_A),A.T),b)
print("-------------")
print("result x")
print(x)
print("-------------")
print("verify")
print(np.dot(A,x))
