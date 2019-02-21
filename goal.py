from IPython.display import clear_output
import matplotlib.pyplot as plt
from time import sleep
import numpy as np 
import random


def print_board(agent_position):
    fields = list(range(16))
    board = "-----------------\n"
    for i in range(0, 16, 4):
        line = fields[i:i+4]
        for field in line:
            if field == agent_position:
                board += "| A "
            elif field == fields[0] or field == fields[-1]:
                board += "| X "
            else:
                board += "|   "
        board += "|\n"
        board += "-----------------\n"     
    print(board)

def create_state_to_state_prime_verbose_map():
    l = list(range(16))
    state_to_state_prime = {}
    for i in l:
        if i == 0 or i == 15:
            state_to_state_prime[i] = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
        elif i % 4 == 0:
            state_to_state_prime[i] = {'N': i - 4 if i - 4 in l else i, 'E': i + 1 if i + 1 in l else i, 'S': i + 4 if i + 4 in l else i, 'W': i}
        elif i % 4 == 3:
            state_to_state_prime[i] = {'N': i - 4 if i - 4 in l else i, 'E': i, 'S': i + 4 if i + 4 in l else i, 'W': i - 1 if i - 1 in l else i}
        else:
            state_to_state_prime[i] = {'N': i - 4 if i - 4 in l else i, 'E': i + 1 if i + 1 in l else i, 'S': i + 4 if i + 4 in l else i, 'W': i - 1 if i - 1 in l else i}

    return state_to_state_prime

def create_probability_map():
    states = list(range(16))
    state_to_state_prime = create_state_to_state_prime_verbose_map()

    probability_map = {}

    for state in states:
        for move in ['N', 'S', 'E', 'W']:
            for prime in states:
                probability_map[(prime, -1, state, move)] = 0 if prime != state_to_state_prime[state][move] else 1
    
    return probability_map

def create_random_policy():
    return {i: {'N': 0.0, 'E': 0.0, 'S': 0.0, 'W': 0.0} if i == 0 or i == 15 else {'N': 0.25, 'E': 0.25, 'S': 0.25, 'W': 0.25} for i in range(16)} # [N, E, S, W]

def create_greedy_policy(V_s):
    s_to_sprime = create_state_to_state_prime_verbose_map()
    policy = {}

    for state in range(16):
        state_values = {a : V_s[s_to_sprime[state][a]] for a in ['N', 'S', 'E', 'W']}

        if state == 0 or state == 15:
            policy[state] = {'N':0.0, 'E':0.0, 'S':0.0, 'W':0.0}
        else:
            max_actions = [k for k,v in state_values.items() if v == max(state_values.values())]
            policy[state] = {a: 1/len(max_actions) if a in max_actions else 0.0 for a in ['N', 'S', 'E', 'W']}

    return policy

def iterative_policy_evaluation(policy, theta=0.01, discount_rate=0.5):
    V_s = {i:0 for i in range(16)}
    probability_map = create_probability_map()

    delta = 100
    while not (delta < theta):
        delta = 0
        for state in range(16):
            v = V_s[state]
            total = 0
            
            for action in ['N', 'E', 'S', 'W']:
                action_total = 0
                for state_prime in range(16):
                    action_total += probability_map[(state_prime, -1, state, action)] * \
                        (-1 + discount_rate*V_s[state_prime])
                
                total += policy[state][action] * action_total
            
            V_s[state] = round(total, 1)
            delta = max(delta, abs(v - V_s[state]))
    
    return V_s

def agent(policy, starting_position=None, verbose=False):
    l = list(range(16))
    state_to_state_prime = create_state_to_state_prime_verbose_map()
    agent_position = random.randint(1, 14) if starting_position is None else starting_position
        
    step_number = 1
    action_taken = None
    
    if verbose:
        print("Move: {} Position: {} Action: {}".format(step_number, agent_position, action_taken))
        print_board(agent_position)
        print("\n")
        sleep(2)
    
    while not (agent_position == 0 or agent_position == 15):
        if verbose:
            clear_output(wait=True)
            print("Move: {} Position: {} Action: {}".format(step_number, agent_position, action_taken))
            print_board(agent_position)
            print("\n")
            sleep(1)
        
        current_policy = policy[agent_position]
        ##next_move = random.random()
        ##lower_bound = 0
        max_actions = [k for k,v in current_policy.items() if v == max(current_policy.values())]
        agent_position = state_to_state_prime[agent_position][max_actions[0]]
        action_taken = max_actions[0]
                
        step_number += 1   
                
    if verbose:
        clear_output(wait=True)
        print("Move: {} Position: {} Action: {}".format(step_number, agent_position, action_taken))
        print_board(agent_position)
        print("Win!")
    
    return step_number


##clear_output(wait=True)
##agent(create_random_policy(),verbose=True)
policy = create_random_policy()
V_s = iterative_policy_evaluation(policy)
policy = create_greedy_policy(V_s)
agent(policy, verbose=True)