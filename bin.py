import torch
import torch.nn as nn
import torch.multiprocessing as mp 

import numpy as np 
import random
import os, math, pdb


from ddpg_agent import Agent
from shared_adam import SharedAdam
from model import Actor, Critic
from syncutils import v_wrap, set_init, push_and_pull, record


import gym 
import gym.spaces 

random_seed = 0
state_size = 3
action_size = 3
steps = 5

def change_opt(optimizer, arg):
    for group in optimizer.param_groups:
        for p in group['params']:
            # print(group)
            optimizer.state[p][arg] += 1



def print_opt_state(optimizer,arg):
    for group in optimizer.param_groups:
        for p in group['params']:
            print(optimizer.state[p][arg])


# actor = Actor(state_size, action_size, random_seed)
# optimizer = SharedAdam(actor.parameters())

# print_opt_state(optimizer, 'step')
# change_opt(optimizer, 'step')
# print('After change')
# print_opt_state(optimizer, 'step')

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w {}'.format(name)
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Actor(state_size, action_size, random_seed)

    def run(self):
        print('Starting Process {}'.format(self.name))
        # change_opt(self.opt, 'step')
        # print_opt_state(self.opt, 'step')


        for i in range(steps):
            change_opt(self.opt, 'exp_avg')
            self.opt.step()
            # print_opt_state(optimizer, 'step')
            # print_opt_state(optimizer, 'exp_avg')



if __name__=='__main__':

    gnet = Actor(state_size, action_size, random_seed)
    optimizer = SharedAdam(gnet.parameters())

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()


    print('Number of workers is {} '.format(mp.cpu_count()))
    
    workers = [Worker(gnet, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]


