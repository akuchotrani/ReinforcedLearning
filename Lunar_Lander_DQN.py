# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:30:38 2017

@author: aakash.chotrani
"""

import gym
import numpy as np
import tensorflow as tf


'''
class Agent:
    
    def Act():
    
    def Remember():
'''
        

class Environment:
    
     def __init__(self,problem):
        self.problem = problem
        self.env = gym.make(problem)
        
     def run_random_action(self):
        for i_episode in range(10):
            reward = 0
            for t in range(100):
                self.env.render()
                action = self.env.action_space.sample()
                newState, reward, done, info = self.env.step(action)
                if done:
                    break
                
     def closeEnvironment(self):
        self.env.close()
                
                
def main():
    env = Environment("LunarLander-v2")
    stateCnt  = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n
   # env.closeEnvironment()
    
    
    
    
main()
