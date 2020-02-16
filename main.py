import gym.spaces
import random
import torch
import numpy as np
import time
from itertools import count
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = gym.make('LunarLanderContinuous-v2')
env.seed(0)

# size of each action
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)

# examine the state space 
states = env.observation_space.shape
state_size = states[0]
print('Size of state:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

def ddpg(n_episodes=100000, max_t=50000, print_every=50):
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

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

