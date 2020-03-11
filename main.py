import gym.spaces
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp 
from syncutils import v_wrap, set_init, push_and_pull, record

import numpy as np
import time
from itertools import count
from collections import deque

from ddpg_agent import Agent
from model import Actor, Critic 
from shared_adam import SharedAdam
import gym
import math, os, pdb 
os.environ["OMP_NUM_THREADS"] = "1"


UPDATE_GLOBAL_ITER = 5
MAX_EP = 10000
MAX_EP_STEP = 5000

env = gym.make('LunarLanderContinuous-v2')
env.seed(0)

random_seed = 0
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w{}'.format(name)
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.agent = Agent(state_size, action_size, gnet['actor'], gnet['critic'] \
        , opt['actor_optimizer'], opt['critic_optimizer'], random_seed)           # local agent
        
        self.env = gym.make('LunarLanderContinuous-v2')

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            state = self.env.reset()
            ep_r = 0.
            self.agent.reset()
            
            for t in range(MAX_EP_STEP):
                # if self.name == 'w1':
                #     self.env.render()

                action = self.agent.act(state)
                action = np.clip(action, -1, 1)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done, t)

                
                if t == MAX_EP_STEP - 1:
                    done = True

                ep_r += reward
                
                if total_step % UPDATE_GLOBAL_ITER == 0  or done: # time to sync

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1
                



if __name__=='__main__':

    # multiple copies of both actor and critic (one pair per worker)
    # updates sent to global model 

    gnet = {'actor': Actor(state_size, action_size, random_seed).to(device), \
            'critic': Critic(state_size, action_size, random_seed).to(device) }

    opt = {} # stores both shared optimizers for critic and actor networks
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3

    print('Networks present are: ')
    for key, value in gnet.items():  # Alternatively if gnet is a class, use gnet.__dict__
        if isinstance(value, nn.Module):
            value.share_memory()
            print('Sharing in memory {}: '.format(key))
            if key == 'actor' or key == 'critic':
                opt[key+'_optimizer'] = SharedAdam(value.parameters(), lr=LR_ACTOR if key == 'actor' else LR_CRITIC)


    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()


    # parallel training
    
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    scores = res

    import matplotlib.pyplot as plt
    import matplotlib
    
    matplotlib.use('TkAgg')
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

