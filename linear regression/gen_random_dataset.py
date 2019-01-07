import numpy as np
import pickle as pi

N = 100000
alpa = 0.1
R = 10
y = []

w = np.random.uniform(-R,R,2) # dimension 2 W
b = np.random.uniform(-R,R,1) # just one value
x1 = np.random.uniform(-R,R,N)
x2 = np.random.uniform(-R,R,N) # dimension 2 X
x_list = np.array([x1,x2]) #row vector 2 by N

w = w.reshape(-1,1) # column vector d by 1

for i in range(N):
    y.append(np.random.normal(np.dot(w.T,x_list[:,i])+b,(alpa*R)**2,1))
# correspond to each X column is calculated with W + b
# generate one y
# iterate N
y_list = np.array(y) # column vector N by 2
x_list = x_list.T # column vector N by 2


Data = np.hstack((y_list,x_list))
# first column is y_list
# second column is x_list[:,0]
# third column is x_list[:,1]

np.random.shuffle(Data)
# switching row as a random
tr = N*0.85
dv = N*0.05
te = N*0.1
# assign as each percent
training_set = Data[:int(tr)]
dev_set = Data[int(tr):int(tr+dv)]
test_set = Data[int(tr+dv):int(tr+dv+te)]
true_W = w
true_B = b

print("Generate!")
print("Train_set: ",training_set.shape)
print("Dev_set: ",dev_set.shape)
print("Test_set: ",test_set.shape)
print("true_W: ",true_W)
print("true_B: ", true_B)

with open('myrandomdataset_100000.pkl','wb') as file:
    pi.dump(training_set, file)
    pi.dump(dev_set, file)
    pi.dump(test_set, file)
    pi.dump(true_W, file)
    pi.dump(true_B, file)
