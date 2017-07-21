# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:30:38 2017

@author: aakash.chotrani
"""

import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
#import sys
#from utility.utility import get_path

REPLAY_SIZE = 100000
BATCH_SIZE = 512
GAMMA = 0.99
INITIAL_EPSILON = 1
DECAY_RATE = 0.975


class DQN:
    
    def __init__(self,state_dim,action_dim):
        self.stateDimension = state_dim
        self.actionDimension = action_dim
        
        self.Neurons_Layer_1 = 20
        self.Neurons_Layer_2 = 30
        self.Neurons_Layer_3 = 40
        
        self.replay_buffer = deque()
        
        self.epsilon = INITIAL_EPSILON
        
        
        self.Create_Neural_Network()
        self.Training_Method()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        
    
    def Create_Neural_Network(self):
        print("create neural net called")
        self.state_input = tf.placeholder(tf.float32,[None,self.stateDimension], name = 'state_inputs')
        
        self.W1 = tf.get_variable("W1",[self.stateDimension,self.Neurons_Layer_1])
        self.b1 = tf.Variable(tf.constant(0.01,shape = [self.Neurons_Layer_1, ]))
        layer1 = tf.nn.relu(tf.matmul(self.state_input,self.W1) + self.b1)
        
        self.W2 = tf.get_variable("W2",[self.Neurons_Layer_1,self.Neurons_Layer_2])
        self.b2 = tf.Variable(tf.constant(0.01,shape = [self.Neurons_Layer_2, ]))
        layer2 = tf.nn.relu(tf.matmul(layer1,self.W2) + self.b2)
        
        self.W3 = tf.get_variable("W3",[self.Neurons_Layer_2,self.Neurons_Layer_3])
        self.b3 = tf.Variable(tf.constant(0.01,shape = [self.Neurons_Layer_3, ]))
        layer3 = tf.nn.relu(tf.matmul(layer2,self.W3) + self.b3)
        
        
        self.W4 = tf.get_variable("W4",[self.Neurons_Layer_3,self.actionDimension])
        self.b4 = tf.Variable(tf.constant(0.01,shape = [self.actionDimension, ]))
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
            self.Train_Network()
        
    
    def Train_Network(self):
        randomBatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in randomBatch]
        action_batch = [data[1] for data in randomBatch]
        reward_batch = [data[2] for data in randomBatch]
        next_state_batch = [data[3] for data in randomBatch]
        
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
        
        
        for i in range(0,BATCH_SIZE):
            done = randomBatch[i][4]
            
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA* np.max(Q_value_batch[i]))
            
            feed_dict = {self.y_input: y_batch,
                         self.action_input:action_batch,
                         self.state_input: state_batch}
            
        self.session.run(self.optimizer,feed_dict)
            
            
            
    def explore_action(self,state):
        
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon - 0.001
        else:
            self.epsilon *= DECAY_RATE
            
            
        if random.random() <= self.epsilon:
            return random.randint(0,self.actionDimension -1)
        else:
            return np.argmax(Q_value)
        
    
    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])
    
    
    def store_data(self,state,action,reward,next_state,done):
        one_hot_action_array = np.zeros(self.actionDimension)
        one_hot_action_array[action] = 1
        self.replay_buffer.append((state,one_hot_action_array,reward,next_state,done))
    
        #taking care of data overflow
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
            
    
    def save_network(self,path):
        self.saver.save(self.session,path + '/my_lunar_lander_model.ckpt')
            
            
            
            
        
        
        
        
        


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
                
     def play_games_to_record_data(self,agent):
         for i_episode in range(1000):
             state = self.env.reset()
             for step in range(500):
                 action = agent.action(state)
                 next_state,reward,done,info = self.env.step(action)
                 agent.store_data(state,action,reward,next_state,done)
                 state = next_state
                 if done:
                     break
    
     def  play_games_and_train(self,agent):
         for i_episode in range(1000):
             state = self.env.reset()
             for step in range(500):
                 action = agent.explore_action(state)
                 next_state,reward,done,info = self.env.step(action)
                 agent.perceive(state,action,reward,next_state,done)
                 state = next_state
                 if done:
                     break
                 if i_episode % 100 == 0:
                     #agent.save_network(saveDir)
                     total_reward = 0
                     for i in range(10):
                         state = self.env.reset()
                         for j in range(500):
                             self.env.render()
                             action = agent.action(state)
                             state,reward,done,info = self.env.step(action)
                             total_reward += reward
                             if done:
                                 break
                     average_reward = total_reward/500
                     print("Episode: ",i_episode, " Average reward: ",average_reward)
                    
                     if average_reward >=200:
                         break
                 
                 
                
     def closeEnvironment(self):
        self.env.close()
                
                
def main():
    env = Environment("LunarLander-v2")
    stateCnt  = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n
    agent = DQN(stateCnt,actionCnt)
    env.play_games_to_record_data(agent)
   # saveDir = get_path('tmp/'+'lander%2d'%np.random.randint(99))
    env.play_games_and_train(agent)
    env.closeEnvironment()
    
    
    
    
main()
