import matplotlib.pyplot as plt
import numpy as np 
import random

def generate_problem(k):
    return np.random.normal(loc=0.0, scale=1, size=10)

def generate_reward(problem, action):
    return np.random.normal(loc=problem[action], scale=0.2)

def k_bandit(problem, k, steps, exploration_rate):
    Q = {i: 0 for i in range(k)}
    N = {i: 0 for i in range(k)}
    avgReward = 0
    rewards = []

    for i in range(steps):
        
        explore = random.uniform(0,1) < exploration_rate
        if explore:
            action = random.randint(0, k-1)
        else:
            action = max(Q, key=Q.get)    

        reward = generate_reward(problem, action)
        N[action] += 1
        Q[action] += 0.1 * (reward - Q[action])

        avgReward += 0.1 * (reward - avgReward)        
        rewards.append(avgReward)

    return rewards


epidoseCount = 10000

avgexp01   = [0 for i in range(epidoseCount)]

for j in range(epidoseCount):
    problem = generate_problem(10)
    exp01   = k_bandit(problem,10,epidoseCount,0.1)
    for x in range(epidoseCount):
        avgexp01[x]  += (1.0 / (j+1)) * (exp01[x] - avgexp01[x])

p3 = plt.plot(range(epidoseCount), avgexp01,linewidth=2, color='y')
plt.show()
