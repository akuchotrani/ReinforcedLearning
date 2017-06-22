
import gym
import numpy as np
env = gym.make('LunarLander-v2')

MAX_EPISODES = 1000


def run_episode():
     total_reward = 0
     state = env.reset()
     print("State: ",state)
     for i_episode in range(20):
        observation = env.reset()
        action = np.random.randint(0,4)
        for t in range(500):
            env.render()
           # print(observation)
            #action = env.action_space.sample()
           
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Episode = ", i_episode)
                break
        return total_reward
    


def train_my_ship():
    
    for _ in range(MAX_EPISODES):
        reward = run_episode()
        print(reward)
    
   
            
            
train_my_ship()
