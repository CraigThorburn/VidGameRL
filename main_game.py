### Set Imports

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

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-debug", help="run with debugging output on")
parser.add_argument("-overwrite", help="overwrite any existing output files")
parser.add_argument("-modelname", help='name of this simulation')
args = parser.parse_args()

### Define Model Name From Arguments
if args.modelname:
    MODELNAME = args.modelname

to_print = args.debug

### Define Optimization Function
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
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)
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

### Define Action Selection Function
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
        network_return = policy_net(state)
        network_return = network_return.max(-1)[1].view(1, 1)
    if sample > eps_threshold:
        action = network_return
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

### Define Memory Transitions
transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

### Set File Locations
ACTION_LIST = ROOT + ACTION_LIST_FILE + '_' + MODELNAME + '.txt'
REWARD_LIST = ROOT + REWARD_LIST_FILE + '_' + MODELNAME + '.txt'
STATE_LIST = ROOT + STATE_LIST_FILE + '_' + MODELNAME + '.txt'

### Create Environment and Set Other File Locations
if GAME_TYPE == 'simple':
    env = SimpleGame(REWARD_PATH, STATE_PATH, EPISODE_PATH)
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST]
    to_output = ['', '', '']

elif GAME_TYPE == 'movement':
    env = MovementGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, STIMULUS_REPS)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST]
    to_output = ['', '', '', '']

elif GAME_TYPE == 'oneshotmovement':
    env = MovementGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, STIMULUS_REPS)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST,]
    to_output = ['', '', '', '']
print("environment created")

### Validate Environment
env.validate_environment()
print('environment valid')

### Report Model Parameters
num_inputs = env.get_n_dims() + env.get_n_location_dims()  # len(phones2vectors.keys())
print('num inputs: ' + str(num_inputs))
num_episodes = env.get_n_episodes()  # len(input_data)
print('num episodes: ' + str(num_episodes))
n_actions = env.get_n_actions()
print('num actions: ' + str(n_actions))

### Create Model Networks
policy_net = DQN_NN(num_inputs, n_actions, LAYERS).to(device)
target_net = DQN_NN(num_inputs, n_actions, LAYERS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

### Define Optimizer
optimizer = optim.RMSprop(policy_net.parameters())
torch.backends.cudnn.enabled = False
# TODO: Check what exactly this is doing ^^^
# CHANGE: from RMSprop to SGD

### Define Model Memory
memory = ReplayMemory(10000)
print('model and memory initialized')

### Initiate Environment
env.initiate_environment()
print('environment initialized')

### Touch Save Files
if args.overwrite:
    write_method = 'w'
else:
    write_method = 'x'
for name in OUTPUTS:
    outfile = open(name, write_method)
    outfile.close()

### Set Model Start
tic = time.time()
steps_done = 0

print('starting simulation')
for i_episode in range(num_episodes):

    ### Set Episode For Output
    for i in range(len(to_output)):
        to_output[i] = to_output[i] + 'eps' + str(i_episode)

    ### Initialize This Episode
    total_reward = 0
    episode_length = env.get_current_episode_length()
    state = env.get_state().to(device)
    done = False

    ### Iterate Over Episode
    for t in count():

        ### Select Action
        action = select_action(state)

        ### Set State For Output
        to_output[2] = to_output[2] + ' ' + env.get_state_str()

        ### Take Game Step
        reward = env.step(action)
        total_reward += reward

        ### Set Remaining Outputs
        to_output[0] = to_output[0] + ' ' + str(float(reward))
        to_output[1] = to_output[1] + ' ' + str(float(action))
        if GAME_TYPE != 'simplegame':
            to_output[3] = to_output[3] + ' ' + env.get_location_str()

        ### Get State Tensor
        reward = torch.tensor([reward], device=device, dtype=torch.float64)

        ### Get Next State
        next_state = env.get_state()
        if next_state == None:
            done = True
        else:
            next_state.to(device)

        ### Store Transition
        memory.push(state, action, next_state, reward)

        ### Move to the next state
        state = next_state

        ### Perform Optimization Step
        policy_net.train()
        optimize_model()

        ### If Episode End
        if done:
            for i in range(len(to_output)):
                to_output[i] += '\n'

            if i_episode + 1 != num_episodes:
                env.advance_episode()
            break

    ### Update Target Network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    ### Save Updates To File
    if i_episode % UPDATES == 0:
        for i in range(len(to_output)):
            outfile = open(OUTPUTS[i], 'a+')
            outfile.write(to_output[i])
            to_output[i] = ''
            outfile.close()

        ### Save Model Checkpoint
        torch.save(policy_net.state_dict(), ROOT + '/checkpoints/' + MODELNAME + '_model.pt')

        ### Print Timing Information
        toc = time.time()
        time_passed = toc - tic
        time_remaining = ((time_passed / (i_episode + 1)) * num_episodes - time_passed) / 60
        print(
            'episode: ' + str(i_episode) + ', percent complete: ' + str(
                math.ceil((i_episode / num_episodes) * 100)) + \
            ', time remaining: ' + str(int(time_remaining)) + ' minutes')

print('model complete')
print('saving data')

### Save Final Outputs
for i in range(len(to_output)):
    outfile = open(OUTPUTS[i], 'a+')
    outfile.write(to_output[i])
    outfile.close()
print('data saved')

### Save Final Model
print('saving model')
torch.save(policy_net.state_dict(), ROOT + MODELNAME + '_final.pt')
print('model saved')
print('done')

# TODO: Cuda issues on clip (loaded vocab is not going to device)
# TODO: Memory is only cleared at end of episode
