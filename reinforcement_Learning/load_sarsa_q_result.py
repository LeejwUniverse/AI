import pickle
 
with open('sarsa_q_result.pkl','rb') as file:
    sarsa_q = pickle.load(file)
    sarsa_optimal = pickle.load(file)
    qlearn_q = pickle.load(file)
    qlearn_optimal = pickle.load(file)
    

print(sarsa_q)
print(sarsa_optimal)
print(qlearn_q)
print(qlearn_optimal)
