BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 16000
TARGET_UPDATE = 5
LR = 0.09

LAYERS = [[3,3],[3,3]]

MEM_SIZE = 100000
TOKEN_TYPE = 'token'

GAME_TYPE = 'continuous_movement'
GAME_MODE = 'correct'
EXPERIMENT = 'experimental_continuous_movement_parameterexploration/'

OVERWRITE = True

ROOT = '/mnt/c/files/research/projects/vid_game/data/'+EXPERIMENT
ROOT = '/fs/clip-realspeech/projects/vid_game/data/'+EXPERIMENT


UPDATES = 100
STIMULUS_REPS = 8
MOVE_SEPERATION = 1 #(not implemented yet)
WAITTIME = 30

STATE_FILE = 'states'
REWARD_FILE = 'rewards'
EPISODE_FILE = 'episodes'
TRANSITION_FILE = 'transitions'
LOCATION_FILE = 'locations'
MODEL_FOLDER = 'models'

ACTION_LIST_FILE = 'exp/action_out'
STATE_LIST_FILE = 'exp/state_out'
LOCATION_LIST_FILE = 'exp/location_out'
REWARD_LIST_FILE = 'exp/reward_out'

MODELNAME = 'test'

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
