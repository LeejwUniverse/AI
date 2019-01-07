import numpy as np
import random

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

reward = -1.0

epsilon = 0.1
alpha = 0.5

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
action = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT] # left,right,down,up

def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], 6), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, 9)]
    else:
        assert False

def epsilon_greedy(Q,state):

    if np.random.binomial(1, epsilon) == 1:
        act = np.random.choice(action)
    else:
        Q_values = Q[:,state[0], state[1]]
        act = np.random.choice([act for act, Q_values in enumerate(Q_values) if Q_values == np.max(Q_values)])    
        
    return act


def sarsa(episode):
    time_step = []
    Q = np.zeros((4, 7, 10))
    time = 0
    for k in range(episode):
        # init data
        state = [3, 0]
        GOAL = [3, 7]

        # choose action through epsilon-greedy
        action = epsilon_greedy(Q,state)
   
        # start learning
        while state != GOAL:
            next_state = step(state, action)
            next_action = epsilon_greedy(Q,next_state)
     
            # update
            Q[action, state[0], state[1]] += alpha * (reward + Q[next_action,next_state[0], next_state[1]] -
                     Q[action,state[0], state[1]])
            state = next_state
            action = next_action
            time += 1
        time_step.append(time)
        
    return time_step, Q

def compute_optimal_policy(Q):
    optimal_policy = []
    for i in range(0,7):
        optimal_policy.append([])
        for j in range(0, 10):
            if [i, j] == [3,7]:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(Q[:, i, j])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy:')
    for row in optimal_policy:
        print(row)

    return optimal_policy

    
def main():
    episode = 10
    
    time_step, Q = sarsa(episode)
    
    print("time_step: ",time_step)
    
    optimal_policy = []
    for i in range(0,7):
        optimal_policy.append([])
        for j in range(0, 10):
            if [i, j] == [3,7]:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(Q[:, i, j])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy:')
    for row in optimal_policy:
        print(row)
    
if __name__ == '__main__':
    main()
