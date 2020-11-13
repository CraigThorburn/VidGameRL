# import math
# import random
# import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import sys
# from collections import namedtuple
from itertools import count

# import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
from ReplayMemory import *
from DQN import *
from Environment import *

import time
from params import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-debug", help="run with debugging output on")
parser.add_argument("host", help="server on which code is run")
parser.add_argument("-overwrite", help="overwrite any existing output files")
parser.add_argument("-modelname", help = 'name of this simulation')
args = parser.parse_args()

if args.host == 'local':
    ROOT = '/mnt/c/files/research/projects/vid_game/data/'+EXPERIMENT
elif args.host == 'clip':
    ROOT = '/fs/clip-realspeech/projects/aud_neuro/models/dqn/WSJ/'

if args.modelname:
    MODELNAME = args.modelname

to_print = args.debug

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    policy_net.train()

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack(tuple([s for s in batch.next_state
                                               if s is not None]))


    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)
    state_action_values = state_action_values.reshape(BATCH_SIZE, n_actions).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values_network = target_net(non_final_next_states)
    # TODO: Update policy_net
    next_state_values[non_final_mask] = next_state_values_network.max(-1)[0]

    # Compute the expected Q values
    expected_state_action_values =  reward_batch + (next_state_values * GAMMA)
    # TODO: Make sure loss is functioning correctly
    # TODO: sort Cuda Movement
    # TODO: Optimization change
    policy_net.train()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())
    loss = loss.double()
    # Optimize the model
    optimizer.zero_grad()
    # print(policy_net.training)
    loss.backward()
#    for param in policy_net.parameters():
 #       param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        network_return= policy_net(state)
        network_return = network_return.max(-1)[1].view(1, 1)
    if sample > eps_threshold:
        action = network_return
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action


# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#    from IPython import display

STATE_PATH = ROOT + STATE_FILE +'.txt'
REWARD_PATH = ROOT + REWARD_FILE+'.txt'
EPISODE_PATH = ROOT + EPISODE_FILE+'.txt'
OUT_PATH = ROOT + OUT_FILE+'_'+MODELNAME+'.txt'


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


env = IndependentEnvironment()


# Get length of file to find number of episodes
print('loading states')
env.load_states(STATE_PATH)
print('loading rewards')
env.load_rewards(REWARD_PATH)
print('loading episodes')
env.load_episodes(EPISODE_PATH)
print('data loaded')
env.validate_environment()
print('environment valid')


num_inputs = env.get_n_dims() # len(phones2vectors.keys())
print('num inputs: ' + str(num_inputs))
num_episodes = env.get_n_episodes() # len(input_data)
print('num episodes: ' + str(num_episodes))
n_actions = env.get_n_actions()
print('num actions: ' + str(n_actions))


policy_net = DQN_NN(num_inputs, n_actions, LAYERS).to(device)
target_net = DQN_NN(num_inputs, n_actions, LAYERS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# CHANGE: from RMSprop to SGD
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
torch.backends.cudnn.enabled = False
# TODO: Check what exactly this is doing ^^^

policy_net.train()
print('model and memory initialized')

env.initiate_environment()
print('environment intialized')


steps_done = 0

if args.overwrite:
    write_method = 'w'
else:
    write_method = 'x'
outfile = open(OUT_PATH, write_method)
outfile.close()

to_output = ''

print('starting simulation')
tic = time.time()
for i_episode in range(num_episodes):

    to_output = to_output + 'eps'+str(i_episode)
    total_reward = 0

    # current_episode = get_episode(input_data, i_episode)
    episode_length = env.get_current_episode_length()
    state = env.get_state().to(device)

    if to_print:
        print('-----------------------------------------------------------')
        print('episode num:  ' + str(i_episode) + ', episode_length:  ' + str(episode_length))

    for t in count():
        # Select and perform an action
        done = t + 1 == episode_length

        action = select_action(state)
        reward = env.step(action)
        total_reward += reward

        to_output = to_output + ' ' + str(float(reward))

        reward = torch.tensor([reward], device=device, dtype=torch.float64)

        # Observe new state
        if not done:
            next_state = env.get_state().to(device)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        policy_net.train()
        optimize_model()

        if done:
            to_output = to_output + '\n'

            if i_episode + 1 != num_episodes:
                env.advance_episode()
            # plot_durations()
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % UPDATES == 0:
        outfile = open(OUT_PATH, 'a+')
        outfile.write(to_output)
        outfile.close()
        to_output = ''

        torch.save(policy_net.state_dict(), ROOT + '/checkpoints/' + MODELNAME + '_model.pt')
        toc = time.time()
        time_passed = toc - tic
        time_remaining = ((time_passed / (i_episode + 1)) * num_episodes - time_passed) / 60

        print(
             'episode: ' + str(i_episode) + ', percent complete: ' + str(
                 math.ceil((i_episode / num_episodes) * 100)) + \
             ', time remaining: ' + str(int(time_remaining)) + ' minutes')

print('model complete')
print('saving data')
outfile = open(OUT_PATH, 'a+')
outfile.write(to_output)
outfile.close()
print('data saved')
print('saving model')
torch.save(policy_net.state_dict(), ROOT + MODELNAME + '_final.pt')
print('model saved')
print('done')
# env.render()
# env.close()
# plt.ioff()
# plt.show()

# TODO: Cuda issues on clip (loaded vocab is not going to device)
# TODO: Memory is only cleared at end of episode
