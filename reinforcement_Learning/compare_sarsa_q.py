import numpy as np
import pickle as pi
import matplotlib
import matplotlib.pyplot as plt
from sarsa import *
from q_learning import *

sarsa_time_step, sarsa_q = sarsa(50)
sarsa_optimal = compute_optimal_policy(sarsa_q)
qlearn_time_step, qlearn_q = Q_learning(50)
qlearn_optimal = compute_optimal_policy(qlearn_q)

print(sarsa_time_step)
print(qlearn_time_step)

print(sarsa_optimal)
print(qlearn_optimal)

with open('sarsa_q_result.pkl','wb') as file:
    pi.dump(sarsa_q, file)
    pi.dump(sarsa_optimal, file)
    pi.dump(qlearn_q, file)
    pi.dump(qlearn_optimal, file)

episode = range(1,len(sarsa_time_step)+1)
plt.plot(sarsa_time_step,episode, color='b')
plt.plot(qlearn_time_step,episode, color ='r')
plt.axis(xmin=0,ymin=0)
plt.xlabel('Sarsa/Q-learning time step')
plt.ylabel('Episode')
plt.show()
