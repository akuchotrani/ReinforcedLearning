# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:13:27 2017

@author: aakash.chotrani
"""

import gym
import numpy as np


#Hill Climbing
#initialize weights randomly, utilize memory to save

def run_episode(env,parameters):
    
    observation = env.reset()
    totalreward = 0
    
    for i in range(200):
        env.render()
        action = 0 if np.matmul(parameters,observation)<0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward 
    

#HIll CLIMBING
def train(submit):
    env = gym.make('CartPole-v0')
    
    episodes_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4)*2 -1
    
    bestreward = 0
    
    for i in range(2000):
        newparams = parameters + (np.random.rand(4)*2 -1) * noise_scaling
        reward = run_episode(env,newparams)
        print("reward %d best %d"% (reward,bestreward))
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward >= 190:
                print("train break")
                break

r = train(submit = False)
print (r)
        