#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import gym
from gym import wrappers
import sys
from itertools import chain
from matplotlib import pyplot as plt


# In[21]:


########################################################
######### FROZEN LAKE 4x4 POLICY ITERATION #############
########################################################
"""
Solving FrozenLake environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, gamma, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.observation_space.n)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.env.nA, size=(env.observation_space.n))  # initialize a random policy
    max_iterations = 200000
    gamma = gamma
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            #print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy, i+1


# In[22]:


#comment in the environemnt you want to test
env_name = 'FrozenLake-v0'
#env_name = 'FrozenLake8x8-v0'

converged = []
avg_scores = []
policies = []


for g in np.linspace(0.1,1.0,10):

    if __name__ == '__main__':
        env_name  = 'FrozenLake-v0'
        env = gym.make(env_name)
        optimals = policy_iteration(env, gamma = g)
        optimal_policy = optimals[0]
        converged.append(optimals[1])
        scores = evaluate_policy(env, optimal_policy, gamma = g)
        avg_scores.append(np.mean(scores))
        policies.append(optimal_policy.tolist())
        
#plot average scores by gamma
plt.plot(np.linspace(0.1,1.0,10),converged)


# In[14]:


#plot average scores by gamma
plt.plot(np.linspace(0.1,1.0,10),avg_scores)


# In[7]:


for i in policies:
    print(i)


# In[15]:


converged


# In[16]:


avg_scores


# In[17]:


########### RUN VALUE ITERATION ################

def run_episode(env, policy, gamma, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma ,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

def value_iteration(env, gamma):
    """ Value-iteration algorithm """
    v = np.zeros(env.observation_space.n)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            #print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v, i+1


# In[36]:


#run desired value iteration

#comment in the 4x4 or 8x8, whichever you want to run
#env_name = 'FrozenLake-v0'
env_name = 'FrozenLake8x8-v0'

converged = []
avg_scores = []
policies = []

for g in np.linspace(0.1,1.0,10):

    if __name__ == '__main__':
        env_name  = env_name
        gamma = 1.0
        env = gym.make(env_name)
        optimal = value_iteration(env, g)
        optimal_v = optimal[0]
        converged.append(optimal[1])
        policy = extract_policy(optimal_v, g)
        avg_scores.append(evaluate_policy(env, policy, g, n=1000))
        policies.append(policy)

#plot converged steps by gamma
plt.plot(np.linspace(0.1,1.0,10),converged)


# In[28]:


#plot average scores by gamma
plt.plot(np.linspace(0.1,1.0,10),avg_scores)


# In[29]:


avg_scores


# In[53]:


import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
for episode in range(total_episodes):
    state = env.reset()
    t = 0
    
    while t < max_steps:
        env.render()

        action = choose_action(state)  

        state2, reward, done, info = env.step(action)  

        learn(state, state2, reward, action)

        state = state2

        t += 1
       
        if done:
            break

        time.sleep(0.1)

print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)


# In[75]:


Q = np.array([[0.58344198, 0.5756656 , 0.56035759, 0.56389952],
       [0.10431598, 0.01510228, 0.42447421, 0.58066108],
       [0.5451708 , 0.49764148, 0.52256539, 0.49350022],
       [0.01517503, 0.00401254, 0.08817466, 0.46177721],
       [0.61860446, 0.00478643, 0.52086984, 0.50469879],
       [0.5        , 0.5        , 0.5        , 0.5        ],
       [0.53175872, 0.02054064, 0.10935776, 0.42813382],
       [0.5        , 0.5        , 0.5        , 0.5        ],
       [0.68322524, 0.78047617, 0.50673618, 0.58628453],
       [0.02019012, 0.69136293, 0.54920675, 0.83934735],
       [0.58422045, 0.61445893, 0.09681056, 0.4396542 ],
       [0.5        , 0.5        , 0.5        , 0.5        ],
       [0.5        , 0.5        , 0.5        , 0.5        ],
       [0.12685092, 0.65381332, 0.75223221, 0.02350563],
       [0.84124261, 0.90114059, 0.97595985, 0.98328631],
       [0.5        , 0.5        , 0.5        , 0.5        ]])


# In[76]:


env = gym.make('FrozenLake-v0')

#with open("frozenLake_qTable.pkl", 'rb') as f:
    #Q = pickle.load(f)

def choose_action(state):
    action = np.argmax(Q[state, :])
    return action

# start
total_rew = []
for episode in range(50):

    state = env.reset()
    rewards = []
    #print("*** Episode: ", episode)
    t = 0
    while t < 100:
        #env.render()

        action = choose_action(state)  

        state2, reward, done, info = env.step(action)  
        
        rewards.append(reward)

        state = state2

        if done:
            total_rew.append(rewards)
            break

        #time.sleep(0.5)
        os.system('clear')
print (np.sum(total_rew))


# In[ ]:




