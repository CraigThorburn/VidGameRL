BATCH_SIZE = 100
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 800
TARGET_UPDATE = 10
LR = 0.09

LAYERS = [[3,3],[3,3]]

MEM_SIZE = 50000
TOKEN_TYPE = 'token'

GAME_TYPE = 'movement'
EXPERIMENT = 'control_game/'

OVERWRITE = True

#ROOT = '/mnt/c/files/research/projects/vid_game/data/'+EXPERIMENT
ROOT = '/fs/clip-realspeech/projects/vid_game/data/'+EXPERIMENT


UPDATES = 100
STIMULUS_REPS = 8

STATE_FILE = 'states'
REWARD_FILE = 'rewards'
EPISODE_FILE = 'episodes'
TRANSITION_FILE = 'transitions'
LOCATION_FILE = 'locations'

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
