import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Load the diabetes dataset
diabetes = datasets.load_diabetes().data
target = datasets.load_diabetes().target
target.reshape(-1,1)
# total 442 data

X_training_set = diabetes[:375,:]
X_dev_set = diabetes[375:397,:]
X_test_set = diabetes[397:,:]

Y_training_set = target[:375]
Y_dev_set = target[375:397]
Y_test_set = target[397:]
# assign each set 375/22/45

X_dev_set = X_dev_set.T
X_test_set = X_test_set.T

Y_dev_set = Y_dev_set.T
Y_test_set = Y_test_set.T
# for matching row,column Transpose.

def cost_function(h, y):
    return (1/(2*len(y))) * np.sum((h-y)**2)

def SGD_gradient_descent(batch, learning_rate, epoch):

    minimum_error = 10000000.0
    Best_W = np.random.normal((10,1))
    Best_B = np.random.normal((1))
   
    stop_count = 0
    B = np.random.randn(1)
    W = np.random.randn(10,1)
    W = W.reshape(-1,1)
  
    for i in range(epoch):
        for j in range(int(375/batch)):
          
            y = Y_training_set[batch*j:batch*(j+1)] # choose random 100 data
            X = X_training_set[batch*j:batch*(j+1),0:11]
            y= y.T
            X= X.T
            if stop_count == 1000:
                print("")
                print("[early stop]")
                print("epoch:",i)
                print("minimum_error:",minimum_error)
                print("best cost:",cost_function(np.dot(W.T,X_dev_set) + B, Y_dev_set))
                print("best_w",Best_W)
                print("best_b",Best_B)
            
                return
            
            
            W = W - (learning_rate / batch) * np.sum(((np.dot(W.T, X)+B) - y) * X)
            B = B - (learning_rate/ batch) * np.sum((np.dot(W.T, X)+B) - y)
            
            current_error = cost_function(np.dot(W.T,X_dev_set) + B, Y_dev_set)
            
            if current_error < minimum_error:
                minimum_error = current_error
                Best_W = np.array(W)
                Best_B = np.array(B)
                stop_count = 0
            else:
                stop_count = stop_count + 1
                
        print("-------------------------------------")
        print("epoch:",i)
        print("training: ",cost_function(np.dot(W.T,X) + B, y))
        print("dev: ",cost_function(np.dot(W.T,X_dev_set) + B, Y_dev_set))
        print("test: ",cost_function(np.dot(W.T,X_test_set) + B, Y_test_set))
        
     
    return

def main():

    minibatch_size = 25
    learning_rate = 0.01
    epoch = 10000

    SGD_gradient_descent(minibatch_size, learning_rate, epoch)

if __name__ == "__main__": 
    main()
