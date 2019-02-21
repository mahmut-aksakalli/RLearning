import gym
import numpy as np 
from gym import spaces
from collections import defaultdict

episode_n   = 100000
t_max       = 1000
gamma       = 0.2

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx     = 0

    for _ in range(t_max):
        if render:
            env.render()

        if policy is None:
            action = env.action_space.sample()
        else:
            action = select_action(obs, q_table)

        print('{}\t{}'.format(obs,action))
        obs,reward,done, _ = env.step(action)
        total_reward += (gamma**step_idx) * reward        
        step_idx += 1

        if done:
            break
        
    return total_reward

def obs_to_state(env, obs):
    return obs

def select_action(s, q_table):
    a = np.argmax(q_table[s])
    return a

def learn_q(env, q_table):
    obs = env.reset()
    step_idx     = 0
    alpha        = 0.1

    for _ in range(t_max):
        s       = obs_to_state(env, obs)
        action  = select_action(s, q_table)

        obs,reward,done, _ = env.step(action)

        s_ = obs_to_state(env, obs)

        q_table[s][action] += alpha * (reward + gamma * max(q_table[obs]) - q_table[s][action] )        


        step_idx += 1
        if done:
            break    

if __name__ == '__main__':
    
    env = gym.make('Blackjack-v0')

    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for i in range(episode_n):
        learn_q(env, q_table)
    
    rwds = []
    count = 0
    for _ in range(1000):
        r = run_episode(env, policy=q_table)
        rwds.append(r)
        if int(r) == 1:
            count += 1

    print('avg : {}\twin : {}'.format(np.mean(rwds),count))        
    env.close()
