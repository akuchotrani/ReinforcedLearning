# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:30:38 2017

@author: aakash.chotrani
"""

import gym
import numpy as np
import tensorflow as tf


REPLAY_SIZE = 100000
BATCH_SIZE = 512

class DQN:
    
    def __init__(self,state_dim,action_dim):
        self.stateDimension = state_dim
        self.actionDimension = action_dim
        
        self.Neurons_Layer_1 = 128
        self.Neurons_Layer_2 = 256
        self.Neurons_Layer_3 = 512
        self.Create_Neural_Network()
        
        self.replay_buffer = deque()
        
    
    def Create_Neural_Network(self):
        self.state_input = tf.placeholder(tf.float32,[None,self.state_dim], name = 'state_inputs')
        
        self.W1 = tf.get_variable("W1",[self.stateDimension,self.Neurons_Layer_1])
        self.b1 = tf.get_variable(tf.constant(0.01,shape = [self.Neurons_Layer_1, ]))
        layer1 = tf.nn.relu(tf.matmul(self.state_input,self.W1) + self.b1)
        
        self.W2 = tf.get_variable("W2",[self.Neurons_Layer_1,self.Neurons_Layer_2])
        self.b2 = tf.get_variable(tf.constant(0.01,shape = [self.Neurons_Layer_2, ]))
        layer2 = tf.nn.relu(tf.matmul(layer1,self.W2) + self.b2)
        
        self.W3 = tf.get_variable("W3",[self.Neurons_Layer_2,self.Neurons_Layer_3])
        self.b3 = tf.get_variable(tf.constant(0.01,shape = [self.Neurons_Layer_3, ]))
        layer3 = tf.nn.relu(tf.matmul(layer2,self.W3) + self.b3)
        
        
        self.W4 = tf.get_variable("W4",[self.Neurons_Layer_3,self.actionDimension])
        self.b4 = tf.get_variable(tf.constant(0.01,shape = [self.actionDimension, ]))
        self.Q_value = tf.matmul(layer3,self.W4) + self.b4
    
    
    def Training_Method(self):
        self.action_input = tf.placeholder(tf.float32,[None,self.actionDimension])
        self.y_input = tf.placeholder(tf.float32,[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        
        self.loss = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
        
    
    def perceive(self,state,action,reward,next_state,done):
        one_hot_action_array = np.zeros(self.actionDimension)
        one_hot_action_array[action] = 1
        self.replay_buffer.append((state,one_hot_action_array,reward,next_state,done))
        
        #taking care of data overflow
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_network()
        
    
    def Train_Network(self):
        
        
        
        
        


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
            self.env.reset()
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
    env.run_random_action()
    env.closeEnvironment()
    
    
    
    
main()
