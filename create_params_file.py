
from params import *
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("param_save_file")
parser.add_argument("-gamma")
parser.add_argument("-eps_start")
parser.add_argument("-eps_end")
parser.add_argument("-eps_decay")
parser.add_argument("-target_update")
parser.add_argument("-lr")
parser.add_argument("-updates")
parser.add_argument("-stimulus_reps")
parser.add_argument("-move_seperation")
parser.add_argument("-waittime")
parser.add_argument("-conv_size")
parser.add_argument("-kernel")
parser.add_argument("-stride")
parser.add_argument("-mem_size")
parser.add_argument("-token_type")
parser.add_argument("-game_mode")
parser.add_argument("-game_type")
parser.add_argument("-experiment")
parser.add_argument("-overwrite")
parser.add_argument("-run_num")
parser.add_argument("-episode_file")
parser.add_argument("-state_file")
parser.add_argument("-reward_file")
parser.add_argument("-modelname")
parser.add_argument("-conv1channels")
parser.add_argument("-conv2channels")
parser.add_argument("-midsize")
parser.add_argument("-conv_freeze")
parser.add_argument("-root")
args = parser.parse_args()
print(args)

if args.gamma:
    GAMMA=float(args.gamma)
    print('gamma change: ' + args.gamma)

if args.eps_start:
    EPS_START = float(args.eps_start)
    print('eps_start change: ' + args.eps_start)

if args.eps_end:
    EPS_END = float(args.eps_end)
    print('eps_end change: ' + args.eps_end)

if args.eps_decay:
    EPS_DECAY = float(args.eps_decay)
    print('eps_decay change: ' + args.eps_decay)

if args.target_update:
    TARGET_UPDATE = int(args.target_update)
    print('target_update change: ' + args.target_update)

if args.lr:
    LR = float(args.lr)
    print('lr change: ' + args.lr)

if args.updates:
   UPDATES = int(args.updates)
   print('updates change: ' + args.updates)

if args.stimulus_reps:
    STIMULUS_REPS = int(args.stimulus_reps)
    print('stimulus_reps change: ' + args.stimulus_reps)

if args.move_seperation:
    MOVE_SEPERATION = int(args.move_seperation)
    print('move_seperation change: ' + args.move_seperation)

if args.waittime:
    WAITTIME = int(args.waittime)
    print('waittime change: ' + args.waittime)

if args.conv_size:
    CONV_SIZE = int(args.conv_size)
    print('conv_size change: '+args.conv_size)

if args.kernel:
    KERNEL = int(args.kernel)
    print('kernel changed: '+args.kernel)

if args.stride:
    STRIDE = int(args.stride)
    print('stride changed: '+args.stride)

if args.conv1channels:
    CONV1CHANNELS = int(args.conv1channels)
    print('conv1channels changed: '+args.conv1channels)

if args.conv2channels:
    CONV2CHANNELS = int(args.conv2channels)
    print('conv2channels changed: '+args.conv2channels)

if args.midsize:
    MIDSIZE = int(args.midsize)
    print('midsize changed: '+args.midsize)

LAYERS = [CONV1CHANNELS, CONV2CHANNELS, MIDSIZE]

if args.conv_freeze:
    if args.conv_freeze.lower()=='true':
        CONV_FREEZE = True
    elif args.conv_freeze.lower()=='false':
        CONV_FREEZE = False
    else:
        print('convolution_freeze not recognized')
        raise NotImplementedError

if args.mem_size:
    MEM_SIZE = int(args.mem_size)
    print('mem_size changed: '+args.mem_size)

if args.token_type:
    TOKEN_TYPE = args.token_type
    print('token_type changed: '+args.token_type)

if args.game_mode:
    GAME_MODE = args.game_mode
    print('game_mode changed: '+args.game_mode)

if args.game_type:
    GAME_TYPE = args.game_type
    print('game_type changed: '+args.game_type)

if args.state_file:
    STATE_FILE = args.state_file
    print('state_file change: '+args.state_file)

if args.episode_file:
    EPISODE_FILE = args.episode_file
    print('episode_file change: '+args.episode_file)

if args.reward_file:
    REWARD_FILE = args.reward_file
    print('reward_file change: '+args.reward_file)

if args.root:
    ROOT = args.root
    print('root changed: '+args.root)

if args.experiment:
    EXPERIMENT = args.experiment
    #ROOT = '/mnt/c/files/research/projects/vid_game/data/' + EXPERIMENT
    print('experiment changed: '+args.experiment)

ROOT = ROOT + EXPERIMENT

if args.overwrite:
    if args.overwrite.lower()=='true':
        OVERWRITE = True
    elif args.overwrite.lower()=='false':
        OVERWRITE = False
    else:
        print('overwrite not recognized')
        raise NotImplementedError

    print('overwrite changed: '+args.overwrite)

if args.modelname:
    MODELNAME_ADITIONS = args.modelname
    print('added to model name: '+args.modelname)

MODELNAME = GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay' + str(EPS_DECAY) + '_targetupdate' + str(TARGET_UPDATE) \
                + '_waittime' + str(WAITTIME) + '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_lr'+str(LR) + \
                '_' + MODELNAME_ADITIONS
PRETRAIN_MODELNAME = 'lr' + str(PRETRAIN_LR)+ '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_batchsize' + \
                     str(PRETRAIN_BATCH_SIZE) + '_epochs' + str(PRETRAIN_EPOCHS) +  '_' + MODELNAME_ADITIONS

if args.run_num:
    RUN_NUM = args.run_num
    MODELNAME = MODELNAME + '_run'+str(RUN_NUM)
    PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run' + str(RUN_NUM)

# Set Variables
STATE_PATH = ROOT + STATE_FILE +'.txt'
SIMPLE_STATE_PATH = ROOT + SIMPLE_STATE_FILE +'.txt'
SIMPLE_STATE_TEST_PATH = ROOT + SIMPLE_STATE_TEST_FILE +'.txt'
REWARD_PATH = ROOT + REWARD_FILE+'.txt'
EPISODE_PATH = ROOT + EPISODE_FILE+'.txt'
LOCATION_PATH = ROOT + LOCATION_FILE+'.txt'
TRANSITION_PATH = ROOT + TRANSITION_FILE+'.txt'
MODEL_PATH = ROOT + MODEL_FOLDER + '/'

TEST_STATE_PATH = ROOT + 'test_' +STATE_FILE +'.txt'
TEST_REWARD_PATH = ROOT + 'test_' +REWARD_FILE+'.txt'
TEST_EPISODE_PATH = ROOT + 'test_' +EPISODE_FILE+'.txt'
TEST_LOCATION_PATH = ROOT + 'test_' +LOCATION_FILE+'.txt'
TEST_TRANSITION_PATH = ROOT + 'test_' + TRANSITION_FILE+'.txt'

filename = args.param_save_file
ps={}
f = open(filename, 'w')
for key in dir():
    try:
        if key not in ['argparse','json','ps','save_params', 'f', 'args', 'parser'] and key[0] != '_':
            ps[key] = globals()[key]
    except:
         pass
json.dump(ps,f)

f.close()
