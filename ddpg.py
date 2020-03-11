import gym.spaces
import random
import torch
import torch.nn as nn
import numpy as np
import time
from itertools import count
from collections import deque

from ddpg_agent import Agent

import gym
import math, os, pdb 

MAX_EP = 100000
MAX_EP_STEP = 50000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('LunarLanderContinuous-v2')
env.seed(0)

random_seed = 0
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]


agent = Agent(state_size, action_size, random_seed)

def ddpg(n_episodes=MAX_EP, max_t=MAX_EP_STEP, print_every=50):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        timestep = time.time()
        for t in range(max_t):
            action = agent.act(state)
            action = np.clip(action, -1, 1)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward
            if done:
                break 
                
        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)
        
        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep), end="\n")
                    
        if np.mean(scores_deque) >= 200.0:            
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))            
            break            
            
    return scores