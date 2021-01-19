### Set Imports

# import math
# import random
# import numpy as np
# import matplotlib
#import matplotlib.pyplot as plt
import sys
# from collections import namedtuple
from itertools import count

import torch
# import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
from ReplayMemory import *
from DQN import *
from Environment import *

import time
from params import *
import argparse

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
args = parser.parse_args()

### Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)

for key in all_params:
    globals()[key] = all_params[key]

MODEL_LOCATION=MODEL_PATH + MODELNAME + '.pt'
MODELNAME='conv_'+MODELNAME +'_test'
print('parameters loaded from '+args.params_file)




### Define Action Selection Function
def select_action(state, loc):
    global steps_done

    steps_done += 1
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        network_return = policy_net(state, loc)
        network_return = network_return.max(-1)[1].view(1, 1)
        action = network_return
    return action

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

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

elif GAME_TYPE == 'convmovement':
    env = ConvMovementGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, MOVE_SEPERATION, WAITTIME, GAME_MODE, CONV_SIZE, STIMULUS_REPS)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST,]
    to_output = ['', '', '', '']

else:
    raise(TypeError, 'Game type on implemented in this script')
print("environment created")

to_output.append('')
OUTPUTS.append(ROOT + 'exp/loss' + '_' + MODELNAME + '.txt')

### Validate Environment
env.validate_environment()
print('environment valid')

### Report Model Parameters
num_inputs = env.get_n_location_dims()  # len(phones2vectors.keys())
print('num inputs: ' + str(num_inputs))
num_episodes = env.get_n_episodes()  # len(input_data)
print('num episodes: ' + str(num_episodes))
n_actions = env.get_n_actions()
print('num actions: ' + str(n_actions))
w,h = env.get_aud_dims()

### Create Model Networks
policy_net = torch.load(MODEL_LOCATION, map_location=device)

### Initiate Environment
env.initiate_environment()
print('environment initialized')

### Touch Save Files
if OVERWRITE:
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
    state, loc = env.get_state()
    state = state.to(device)
    loc = loc.to(device)
    done = False

    ### Iterate Over Episode
    for t in count():

        ### Select Action
        action = select_action(state, loc)

        ### Set State For Output.
        out_str = env.get_state_str()

        ### Take Game Step
        reward = env.step(action)
        total_reward += reward

        ### Set Remaining Outputs

        if reward==1:
            to_output[0] = to_output[0] + ' ' + str(float(reward))
            to_output[1] = to_output[1] + ' ' + str(float(action))
            to_output[2] = to_output[2] + ' ' + out_str
            if GAME_TYPE != 'simplegame':
                to_output[3] = to_output[3] + ' ' + env.get_location_str()





        ### Get State Tensor
        reward = torch.tensor([reward], device=device, dtype=torch.float64)

        ### Get Next State
        next_state, next_loc = env.get_state()
        if next_state == None:
            done = True
        else:
            next_state = next_state.to(device)
            next_loc = next_loc.to(device)

        ### Move to the next state
        state = next_state
        loc = next_loc

        ### If Episode End
        if done:
            for i in range(len(to_output)):
                to_output[i] += '\n'

            if i_episode + 1 != num_episodes:
                env.advance_episode()
            break

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
print('done')

# TODO: Cuda issues on clip (loaded vocab is not going to device)
# TODO: Memory is only cleared at end of episode
