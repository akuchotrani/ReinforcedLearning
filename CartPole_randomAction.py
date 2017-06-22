
import gym
env = gym.make('LunarLander-v2')

MAX_EPISODES = 1000


def run_episode():
     for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
           # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            if done:
                print("Episode = ", i_episode,"reward = ",reward)
                break
    


def train_my_ship():
    
    for _ in range(MAX_EPISODES):
        run_episode()
    
   
            
            
train_my_ship()
