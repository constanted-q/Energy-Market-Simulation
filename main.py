# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from client import energy_client
from game_env import energy_game
from util import update_rho,update_rho_SLSQP

n = 5
# Number of clients
capacity = 7
# Maximum capacity of each client's storage
G =  np.array(np.arange(n))+4
#G =  np.zeros(n).astype(int)+capacity+1
# Maximum generation of energy for each client
S = capacity+1
# Number of possibilities of the states
A = capacity+1
# Number of possibilities of the actions
lambda1 = 1.5
# Coefficient for computing energy price
eta_ini = 0.02
# Initial step size
Y = []
# Dual Scores
rho_ini = []
for i in range(n):
    rho_ini.append(np.ones(A*S)/(A*S))
    Y.append(np.zeros(A*S))
# Initial occupation measure for each client is uniform
delta = 1/(5*A*S)
# Minimum value of occupation; shrunk occupation polytope.
K = 10000
# Number of total iterations of Dual Averaging
d = 5000
# Threshold d, warm-up steps
T = 1000
# Length of each
reward_list = []
#Recording average rewards
rho = rho_ini
for k in range(K):
    # Initializing the current episode/batch
    eta = eta_ini/((k+1)**0)
    client = []
    env = []
    visited_states = []
    rewards = []
    for i in range(n):
        c = energy_client(A,S,rho[i])
        e = energy_game(G[i],capacity,S,A)
        v = np.zeros(S)
        r = []
        client.append(c)
        env.append(e)
        visited_states.append(v)
        rewards.append(r)
    asking = np.zeros(n) # Number of energy each client is asking for
    price = 0 # Current price of energy

    # d steps to get sufficiently mixed states
    for p in range(d):
        for i in range(n):
            a = client[i].get_action(env[i].s)
            env[i].step(a)
            asking[i] = env[i].asked
        price = lambda1 * sum(asking)
        for i in range(n):
            r = env[i].get_reward(price)
            rewards[i].append(r)

    # Run and compute R
    flag = 0
    flags = [0]*n
    while flag == 0:
        for i in range(n):
            if_first_visit = 1-int(visited_states[i][env[i].s])
            visited_states[i][env[i].s] = 1
            a = client[i].get_action(env[i].s)
            env[i].step(a)
            asking[i] = env[i].asked
        price = lambda1 * sum(asking)
        for i in range(n):
            r = env[i].get_reward(price)
            rewards[i].append(r)
            if if_first_visit:
                client[i].update_R(r,env[i].s_prev,env[i].a)
            if sum(visited_states[i]) == S:
                flags[i] = 1
        flag = min(flags)
    #Update dual score and rho
    for i in range(n):
        Y[i] += eta * client[i].R.reshape(A*S)
        P = env[i].get_P_matrix()
        rho[i] = update_rho(rho[i], Y[i], P, delta)
        #rho[i] = update_rho_SLSQP(rho[i],Y[i],P,delta)
    for i in range(n):
        print(f"Average reward for client {i} at iterateion {k+1}: {sum(rewards[i])/len(rewards[i])} with {len(rewards[i])} steps.")
        #print(min(rho[i])-delta)
    reward_avg = np.sum(rewards,1)/len(rewards[0])
    print(reward_avg)
    reward_list.append(reward_avg)

with open('1.npy', 'wb') as f:
    np.save(f, np.array(reward_list))

    #print(np.sum(env[2].get_P_matrix(),0))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
