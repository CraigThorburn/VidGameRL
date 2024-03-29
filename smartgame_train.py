### Set Imports

from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
from libs.ReplayMemory import *
from libs.DQN import *
from libs.Environment import *
import json
import time
import argparse
import os
import shutil
import libs.Loss


### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("-run_num")
args = parser.parse_args()

### Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)

for key in all_params:
    globals()[key] = all_params[key]

if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run'+str(RUN_NUM)
    #PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run'+str(RUN_NUM)

try:
    os.makedirs(ROOT + OUT_FOLDER + TRAIN_MODELNAME)
    print('created experiment output folder')
except OSError:
    print('experiment output folder exists')

OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'


print('parameters loaded from '+args.params_file)
shutil.copyfile(args.params_file, ROOT + OUT_FOLDER + 'params_'+TRAIN_MODELNAME + '.params')

print('parameter file moved to results location')

### Define Optimization Function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = ConvTransition(*zip(*transitions))
    policy_net.train()

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    try:
        non_final_next_states = torch.stack(tuple([s for s in batch.next_state
                                               if s is not None]))
    except RuntimeError:
        pass
    non_final_next_locs = torch.stack(tuple([s for s in batch.next_location
                                               if s is not None]))

    state_batch = torch.stack(batch.state)
    locs_batch = torch.stack(batch.loc)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    batch_size, dim1, dim2 = state_batch.size()
    state_action_values = policy_net(state_batch.reshape(batch_size, 1, dim1, dim2), locs_batch)
    state_action_values = state_action_values.reshape(BATCH_SIZE, n_actions).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    batch_size, dim1, dim2 = non_final_next_states.size()
    next_state_values_network = target_net(non_final_next_states.reshape(batch_size, 1, dim1, dim2), non_final_next_locs)
    # TODO: Update policy_net
    next_state_values[non_final_mask] = next_state_values_network.max(-1)[0]

    # Compute the expected Q values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)
    # TODO: Make sure loss is functioning correctly
    # TODO: sort Cuda Movement
    # TODO: Optimization change
    policy_net.train()

    # Compute loss
    loss = loss_func(state_action_values.double(), expected_state_action_values.unsqueeze(1).double(), policy_net.named_parameters())
    loss = loss.double()

    # Optimize the model
    optimizer.zero_grad()
    # print(policy_net.training)
    loss.backward()
    #    for param in policy_net.parameters():
    #       param.grad.data.clamp_(-1, 1)
    optimizer.step()
    to_output[-1] = to_output[-1] + ' ' + str(round(float(loss), 4))

### Define Action Selection Function
def select_action(state, loc):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * eps_episode / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        network_return = policy_net(state.reshape(1,1,state.size()[0], state.size()[1]), loc.reshape(1,loc.size()[0]))
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
                        ('state', 'loc', 'action', 'next_state', 'next_location', 'reward'))

### Set File Locations
ACTION_OUT_PATH = ROOT + OUT_FOLDER + 'train_' + ACTION_OUT_FILE + '_' + TRAIN_MODELNAME + '.txt'
REWARD_OUT_PATH = ROOT + OUT_FOLDER +  'train_' +REWARD_OUT_FILE + '_' + TRAIN_MODELNAME + '.txt'
STATE_OUT_PATH = ROOT + OUT_FOLDER +  'train_' +STATE_OUT_FILE + '_' + TRAIN_MODELNAME + '.txt'
LOCATION_OUT_PATH = ROOT + OUT_FOLDER + 'train_' +LOCATION_OUT_FILE + '_' + TRAIN_MODELNAME + '.txt'

### Create Environment and Set Other File Locations
if GAME_TYPE == 'convmovement':

    env = AcousticsGame2DConvVisualFromFile(ROOT + REWARD_FILE + '.txt', ROOT +STATE_FILE + '.txt', ROOT +EPISODE_FILE + '.txt', ROOT +LOCATION_FILE + '.txt', ROOT +TRANSITION_FILE + '.txt',  ROOT +  GAME_WAVS_FOLDER, ROOT +VISUAL_FILE + '.txt', MOVE_SEPERATION, WAITTIME, GAME_MODE, STIMULUS_REPS, device,
                                            acoustic_params =('mfcc', 22050, 0.2, 400, 400, 160, 13, True))
    OUTPUTS = [REWARD_OUT_PATH, ACTION_OUT_PATH, STATE_OUT_PATH, LOCATION_OUT_PATH]
    to_output = ['', '', '', '']
else:
    raise(TypeError, 'Game type on implemented in this script')
print("environment created")

to_output.append('')
OUTPUTS.append(ROOT + OUT_FOLDER + LOSS_OUT_FILE + '_' + TRAIN_MODELNAME + '.txt')

### Validate Environment
env.validate_environment()
print('environment valid')

### Report Model Parameters
num_inputs = env.get_n_visual_dims()  # len(phones2vectors.keys())
print('num inputs: ' + str(num_inputs))
num_episodes = env.get_n_episodes()  # len(input_data)
print('num episodes: ' + str(num_episodes))
n_actions = env.get_n_actions()
print('num actions: ' + str(n_actions))
w,h = env.get_aud_dims()

### Create Model Networks
# if FREEZE_LAYER_TIME > 0:
#     freeze_layer_end = 0
# else:
#     freeze_layer_end = CONV_FREEZE_LAYER

if CONNECTION_LAYER == 'phone':
    policy_net = DQN_NN_conv_pretrain_phonelayer(h, w,num_inputs, n_actions, KERNEL, STRIDE, LAYERS, CONV_FREEZE, n_phone_layer = NUM_PHONES, freeze_layer=CONV_FREEZE_LAYER).to(device)
    target_net = DQN_NN_conv_pretrain_phonelayer(h, w, num_inputs, n_actions, KERNEL, STRIDE, LAYERS, CONV_FREEZE, n_phone_layer = NUM_PHONES, freeze_layer=CONV_FREEZE_LAYER).to(device)
    MODEL_LOCATION = ROOT + MODEL_FOLDER + 'model_' + PRETRAIN_MODELNAME + '_final.pt'
    policy_net.load_state_dict(torch.load(MODEL_LOCATION, map_location=device), strict=False)
    target_net.load_state_dict(policy_net.state_dict())
elif CONNECTION_LAYER == 'conv':
    policy_net = DQN_NN_conv_pretrain_convlayer(h, w,num_inputs, n_actions, KERNEL, STRIDE, LAYERS, CONV_FREEZE, n_phone_layer = NUM_PHONES, freeze_layer=CONV_FREEZE_LAYER).to(device)
    target_net = DQN_NN_conv_pretrain_convlayer(h, w, num_inputs, n_actions, KERNEL, STRIDE, LAYERS, CONV_FREEZE, n_phone_layer = NUM_PHONES, freeze_layer=CONV_FREEZE_LAYER).to(device)
    MODEL_LOCATION = ROOT + MODEL_FOLDER + 'model_' + PRETRAIN_MODELNAME + '_final.pt'
    policy_net.load_state_dict(torch.load(MODEL_LOCATION, map_location=device), strict=False)
    target_net.load_state_dict(policy_net.state_dict())
elif CONNECTION_LAYER == 'none':
    policy_net = DQN_NN_conv_pretrain_convlayer(h, w,num_inputs, n_actions, KERNEL, STRIDE, LAYERS, CONV_FREEZE).to(device)
    target_net = DQN_NN_conv_pretrain_convlayer(h, w, num_inputs, n_actions, KERNEL, STRIDE, LAYERS, CONV_FREEZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
else:
    raise NotImplementedError



target_net.eval()
policy_net.train()

if LOSS_TYPE == 'ewc':
    precision_matrices,means = torch.load(ROOT + MODEL_FOLDER + FISCHER_FILE + '_' +  PRETRAIN_MODELNAME + '.txt', map_location=device)

    for p, n in policy_net.named_parameters():
        if p not in precision_matrices.keys():
            precision_matrices[p] = torch.zeros(n.size())
            means[p] = torch.zeros(n.size())



    loss_class = libs.Loss.EWCLoss(means, precision_matrices, EWC_IMPORTANCE, device)

elif LOSS_TYPE == 'standard':
    loss_class = libs.Loss.StandardLoss(device)

else:
    raise NotImplementedError

loss_func = loss_class.calculate_loss

### Define Optimizer
optimizer = optim.SGD(policy_net.parameters(), lr = TRAIN_LR) ## TODO: Changed from RMSprop
torch.backends.cudnn.enabled = False
# TODO: Check what exactly this is doing ^^^
# CHANGE: from RMSprop to SGD

### Define Model Memory
memory = ConvReplayMemory(10000)
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

eps_episode = 0
print('starting simulation')
for i_episode in range(num_episodes):
    eps_episode+=1

    ### Set Episode For Output
    for i in range(len(to_output)):
        to_output[i] = to_output[i] + 'eps' + str(i_episode)

    ### Initialize This Episode
    total_reward = 0
    episode_length = env.get_current_episode_length()
    state, loc = env.get_state()

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

        #if reward>=1:
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

        ### Store Transition
        memory.push(state, loc, action, next_state, next_loc, reward)

        ### Move to the next state
        state = next_state
        loc = next_loc

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

    if i_episode > FREEZE_LAYER_TIME:
        policy_net.unfreeze_layers(CONV_FREEZE_LAYER)

    ### Save Updates To File
    if i_episode % UPDATES == 0:
        for i in range(len(to_output)):
            outfile = open(OUTPUTS[i], 'a+')
            outfile.write(to_output[i])
            to_output[i] = ''
            outfile.close()

        ### Save Model Checkpoint
        torch.save(policy_net.state_dict(), ROOT + '/checkpoints/' + TRAIN_MODELNAME + '_model.pt')

        ### Print Timing Information
        toc = time.time()
        time_passed = toc - tic
        time_remaining = ((time_passed / (i_episode + 1)) * num_episodes - time_passed) / 60
        print(
            'episode: ' + str(i_episode) + ', percent complete: ' + str(
                math.ceil((i_episode / num_episodes) * 100)) + \
            ', time remaining: ' + str(int(time_remaining)) + ' minutes')

    if env.simulation_finished():
        break

    if env.is_eps_update():
        eps_episode = env.get_eps_update_num()

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
torch.save(policy_net.state_dict(), ROOT + OUT_FOLDER + 'model_' + TRAIN_MODELNAME + '_final.pt')
print('model saved')
print('done')

# TODO: Cuda issues on clip (loaded vocab is not going to device)
# TODO: Memory is only cleared at end of episode