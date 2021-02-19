### Set Imports

import torch
from itertools import count
import torch.optim as optim
from ReplayMemory import *
from DQN import *
from Environment import *

import os
import time
import argparse
import json

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
args = parser.parse_args()

### Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)

for key in all_params:
    globals()[key] = all_params[key]


MODELNAME='lstm_'+MODELNAME
print('parameters loaded from '+args.params_file)

os.rename(args.params_file, ROOT + MODELNAME + '.params')

print('parameter file moved to results location')

torch_cat = torch.cat
torch_stack = torch.stack
torch_tensor = torch.tensor


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = [LSTMTransition(*zip(*b)) for b in transitions]
    states, actions, next_states, rewards, hiddens, next_hiddens = zip(*batch)

    seq_lengths = [len(s) for s in states]
    pad_number = 99
    longest_seq = max(seq_lengths)
    cat_dim_length = BATCH_SIZE*longest_seq
    padded_state = torch.ones(BATCH_SIZE, longest_seq, num_inputs, device=device) * pad_number
    padded_action = torch.zeros(BATCH_SIZE, longest_seq, device=device)
    padded_next_state = torch.ones(BATCH_SIZE, longest_seq, num_inputs, device=device) * pad_number
    padded_reward = torch.ones(BATCH_SIZE, longest_seq, device=device) * pad_number

    for ind, seq_len in enumerate(seq_lengths):
        this_state = states[ind]
        this_action = actions[ind]
        this_next_state = next_states[ind]
        this_reward = rewards[ind]
        padded_state[ind, 0:seq_len, :] = torch.stack(this_state)
        padded_action[ind, 0:seq_len] = torch.tensor(this_action)
        padded_next_state[ind, 0:seq_len, :] = torch.stack(this_next_state)
        padded_reward[ind, 0:seq_len] = torch.tensor(this_reward)

    #should now have padded states

    # Reshape action and reward tensors
    padded_action = padded_action.reshape(1,cat_dim_length).type(torch.int64)
    padded_reward = padded_reward.reshape(cat_dim_length)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    h_initial = (torch.zeros(1, BATCH_SIZE, 40, device=device), torch.zeros(1, BATCH_SIZE, 40, device=device))
    state_action_values,_ = policy_net(padded_state, seq_lengths, h_initial)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    mask = (padded_reward > pad_number).float()

    state_action_values = state_action_values.reshape(BATCH_SIZE*longest_seq,n_actions).gather(1,padded_action).flatten()

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    #next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values_network, _ = target_net(padded_next_state, seq_lengths, h_initial)
    ## TODO: This is where filtering of non-final states would happen
    next_state_values = next_state_values_network.reshape(cat_dim_length,n_actions).max(-1)[0]

    # Create Mask
    mask = (padded_reward != pad_number).float()

    next_state_values = next_state_values * mask
    state_action_values = state_action_values * mask
    padded_reward = padded_reward * mask

    # pick the values for the label and zero out the rest with the mask
    next_state_values = next_state_values * mask
    state_action_values = state_action_values * mask
    padded_reward = padded_reward * mask


    # Compute the expected Q values
    expected_state_action_values = padded_reward + (next_state_values * GAMMA)


    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values, reduction = 'sum')/sum(seq_lengths)
    loss = loss.double()
    # Optimize the model
    optimizer.zero_grad()
    # print(policy_net.training)
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    ### BEGIN DEBUG
    to_output[-1].append(round(float(loss),4))

rand = random.random
### Define Action Selection Function
def select_action(state, hidden):
    global steps_done
    sample = rand()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * i_episode / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        network_state = state#.reshape(1,len(state),1)
        network_return, hidden = policy_net(network_state.float(), 1, hidden)
        network_return = network_return.max(-1)[1].view(1, 1)
    if sample > eps_threshold:
        action = network_return
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action, hidden

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

### Define Memory Transitions
transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden'))

### Set File Locations
ACTION_LIST = ROOT + ACTION_LIST_FILE + '_' + MODELNAME + '.txt'
REWARD_LIST = ROOT + REWARD_LIST_FILE + '_' + MODELNAME + '.txt'
STATE_LIST = ROOT + STATE_LIST_FILE + '_' + MODELNAME + '.txt'

### Create Environment and Set Other File Locations
if GAME_TYPE == 'simple':
    env = SimpleGame(REWARD_PATH, STATE_PATH, EPISODE_PATH)
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST]
    to_output = [[], [], []]

elif GAME_TYPE == 'movement':
    env = MovementGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, STIMULUS_REPS)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST]
    to_output = [[], [], [], []]

elif GAME_TYPE == 'oneshotmovement':
    env = MovementGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, STIMULUS_REPS)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST,]
    to_output = [[], [], [], []]

elif GAME_TYPE == 'continuousmovement':
    env = AcousticsGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, MOVE_SEPERATION, WAITTIME, GAME_MODE)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST,]
    to_output = [[], [], [], []]

elif GAME_TYPE == 'convolutionalmovement':
    env = ConvAcousticsGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, MOVE_SEPERATION, WAITTIME, GAME_MODE, CONV_SIZE, STRIDE)
    LOCATION_LIST = ROOT + LOCATION_LIST_FILE + '_' + MODELNAME + '.txt'
    OUTPUTS = [REWARD_LIST, ACTION_LIST, STATE_LIST, LOCATION_LIST,]
    to_output = [[], [], [], []]

else:
    raise ModuleNotFoundError

    ### BEGIN DEBUG
to_output.append([])
OUTPUTS.append(ROOT + 'exp/loss' + '_' + MODELNAME + '.txt')



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
policy_net = DQN_LSTM(num_inputs, n_actions, LAYERS,device).to(device) # .to(device)
target_net = DQN_LSTM(num_inputs, n_actions, LAYERS, device).to(device) #to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

### Define Optimizer
optimizer = optim.RMSprop(policy_net.parameters())
torch.backends.cudnn.enabled = False
# TODO: Check what exactly this is doing ^^^
# CHANGE: from RMSprop to SGD

### Define Model Memory
memory = SequentialUpdatesReplayMemory(10000)
print('model and memory initialized')

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



tensor = torch.tensor
step = env.step
get_state_str = env.get_state_str
get_location_str = env.get_location_str
get_state = env.get_state
is_new = env.is_new_state()


print('starting simulation')
for i_episode in range(num_episodes):

    ### Set Episode For Output
    for i in range(len(to_output)):
        to_output[i].append('\neps' + str(i_episode))

    ### Initialize hidden states
    h0 = torch.randn(1, 1, 40, device=device)
    c0 = torch.randn(1, 1, 40, device=device)
    hidden = (h0, c0)

    ### Initialize This Episode
    total_reward = 0
    episode_length = env.get_current_episode_length()
    state = env.get_state().to(device)
    done = False

    ### Iterate Over Episode
    for t in count():


        ### Select Action
        action, next_hidden = select_action(state.double(), hidden)

        ### Set State For Output
        current_state = get_state_str()


        ### Take Game Step
        reward = step(action)
        total_reward += reward

        ### Set Remaining Outputs
        if reward==1:
            to_output[0].append(float(reward))
            to_output[1].append(float(action))
            to_output[2].append(current_state)
            try:
                to_output[3].append(env.get_location_str())
            except KeyError:
                pass
     #   if GAME_TYPE != 'simplegame':
     #       to_output[3] = to_output[3] + ' ' + get_location_str()

        ### Get State Tensor
        reward = tensor([reward], device=device, dtype=torch.float64)

        ### Get Next State
        next_state = get_state()
        if next_state == None:
            done = True
            next_hidden=None
        else:
            next_state = next_state.to(device)
            ### Store Transition
            memory.push(state, action, next_state, reward, hidden, next_hidden)

        ### Move to the next state
        state = next_state
        if is_new:
            h0 = torch.randn(1, 1, 40, device=device)
            c0 = torch.randn(1, 1, 40, device=device)
        else:
            hidden = next_hidden

        ### Perform Optimization Step
        optimize_model()

        ### If Episode End
        if done:

            if i_episode + 1 != num_episodes:
                env.advance_episode()
                memory.advance_episode()
            break

    ### Update Target Network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    ### Save Updates To File
    if i_episode % UPDATES == 0:
        for i in range(len(to_output)):
            outfile = open(OUTPUTS[i], 'a+')
            outfile.write(''.join(str(j) + ' ' for j in to_output[i]))
            to_output[i] = []
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
    outfile.write(''.join(str(j) + ' ' for j in to_output[i]))
    outfile.close()
print('data saved')

### Save Final Model
print('saving model')
torch.save(policy_net.state_dict(),  MODEL_PATH + MODELNAME + '_final.pt')
print('model saved')
print('done')

# TODO: Cuda issues on clip (loaded vocab is not going to device)
# TODO: Memory is only cleared at end of episode
