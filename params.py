BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 5
LR = 0.09

UPDATES = 100
STIMULUS_REPS = 8
MOVE_SEPERATION = 1 #(not implemented yet)
WAITTIME = 10
CONV_SIZE = 10

LAYERS = [[3,3],[3,3]]

MEM_SIZE = 100000
TOKEN_TYPE = 'token'

GAME_TYPE = 'convolutionalmovement'
GAME_MODE = 'correct'
EXPERIMENT = 'experimental_continuous_movement_conv_bettermem/'

OVERWRITE = True

MODELNAME = 'lstm_' + GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay'+ str(EPS_DECAY) + '_targetupdate'+ str(TARGET_UPDATE) \
            + '_waittime'+ str(WAITTIME) + '_convsize'+ str(CONV_SIZE) + '_memreset2'

ROOT = '/mnt/c/files/research/projects/vid_game/data/'+EXPERIMENT
#ROOT = '/fs/clip-realspeech/projects/vid_game/data/'+EXPERIMENT
#MODELNAME = 'test'




STATE_FILE = 'states'
REWARD_FILE = 'rewards'
EPISODE_FILE = 'episodes'
TRANSITION_FILE = 'transitions'
LOCATION_FILE = 'locations'
MODEL_FOLDER = 'models'
TEST_FILE = 'test'

ACTION_LIST_FILE = 'exp/action_out'
STATE_LIST_FILE = 'exp/state_out'
LOCATION_LIST_FILE = 'exp/location_out'
REWARD_LIST_FILE = 'exp/reward_out'


# Set Variables
STATE_PATH = ROOT + STATE_FILE +'.txt'
REWARD_PATH = ROOT + REWARD_FILE+'.txt'
EPISODE_PATH = ROOT + EPISODE_FILE+'.txt'
LOCATION_PATH = ROOT + LOCATION_FILE+'.txt'
TRANSITION_PATH = ROOT + TRANSITION_FILE+'.txt'
MODEL_PATH = ROOT + MODEL_FOLDER + '/'

ALL_PARAMS = ['BATCH_SIZE',  BATCH_SIZE, '\n',
'GAMMA', GAMMA, '\n',
'EPS_START',  EPS_START, '\n',
'EPS_END',  EPS_END, '\n',
'EPS_DECAY',  EPS_DECAY, '\n',
'TARGET_UPDATE',  TARGET_UPDATE, '\n',
'LR', 'LR' '\n',
'UPDATES', UPDATES, '\n',
'STIMULUS_REPS', STIMULUS_REPS,  '\n',
'MOVE_SEPERATION', MOVE_SEPERATION,  '\n',
'WAITTIME', WAITTIME,  '\n',
'GAME_TYPE', GAME_TYPE,  '\n',
'GAME_MODE', GAME_MODE, '\n',
'EXPERIMENT', EXPERIMENT ]

def save_params():
     with open(ROOT + MODELNAME + '_params.txt', 'w') as fi:
         fi.write(''.join(str(p) + ' ' for p in ALL_PARAMS))
