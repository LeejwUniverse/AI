import pickle
import numpy as np

with open('myrandomdataset_10000.pkl','rb') as file:
    training_set = pickle.load(file)
    dev_set = pickle.load(file)
    test_set = pickle.load(file)
    true_W = pickle.load(file)
    true_B = pickle.load(file)
    
def cost_function(h, y):
    return (1/(2*len(y))) * np.sum((h-y)**2)

def W_cost_function(W):
    return (W-true_W)**2

def B_cost_function(B):
    return (B-true_B)**2

def SGD_gradient_descent(batch, learning_rate, epoch):

    minimum_error = 10000000.0
    Best_W = np.random.normal((2,1))
    Best_B = np.random.normal((1))
   
    stop_count = 0
    B = np.random.randn(1)
    W = np.random.randn(len(true_W),1)
    W = W.reshape(-1,1)
  
    for i in range(epoch):
        for j in range(int(len(training_set)/batch)):
          
            y = training_set[batch*j:batch*(j+1),0] # choose random 100 data
            X = training_set[batch*j:batch*(j+1),1:3]
            y= y.T
            X= X.T
            if stop_count == 10000:
                print("")
                print("[early stop]")
                print("epoch:",i)
                print("minimum_error:",minimum_error)
                print("best cost:",cost_function(np.dot(Best_W.T,dev_set[:,1:3].T) + Best_B, dev_set[:,0].T))
                print("best_w",Best_W)
                print("best_b",Best_B)
            
                return
            
            W = W - (learning_rate / batch) * np.sum(((np.dot(W.T, X)+B) - y) * X)
            B = B - (learning_rate/ batch) * np.sum((np.dot(W.T, X)+B) - y)
            
            current_error = cost_function(np.dot(W.T,dev_set[:,1:3].T) + B, dev_set[:,0].T)
            
            if current_error < minimum_error:
                minimum_error = current_error
                Best_W = np.array(W)
                Best_B = np.array(B)
                stop_count = 0
            else:
                stop_count = stop_count + 1
        print("-------------------------------------")
        print("epoch:",i)
        print("training: ",cost_function(np.dot(W.T,training_set[:,1:3].T) + B, training_set[:,0].T))
        print("dev: ",cost_function(np.dot(W.T,dev_set[:,1:3].T) + B, dev_set[:,0].T))
        print("test: ",cost_function(np.dot(W.T,test_set[:,1:3].T) + B, test_set[:,0].T))
        print("W squared error: ",W_cost_function(W))        
        print("B squared error: ",B_cost_function(B))
    return

def main():

    minibatch_size = 50
    learning_rate = 0.003
    epoch = 10000

    SGD_gradient_descent(minibatch_size, learning_rate, epoch)
if __name__ == "__main__": 
    main()



