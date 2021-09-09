### FILE VARIABLES
TRAIN_MODELNAME_ADITIONS = '_1500_exp1'

### General
OVERWRITE = True
EXPERIMENT = 'cht_intonation/'
ROOT = '/mnt/c/files/research/projects/vid_game/data/'
#ROOT = '/fs/clip-realspeech/projects/vid_game/data/'
PARAMS_FOLDER = 'params'



### For Train
GAME_TYPE = 'cht'
GAME_MODE = 'oneshot'
STATE_FILE = 'states'
REWARD_FILE = 'rewards'
EPISODE_FILE = 'episodes_experiment1'
TRANSITION_FILE = 'transitions'
ACTION_FILE = 'actions'
LOCATION_FILE = 'locations'
SIMPLE_STATE_FILE='states_simple'
GAME_WAVS_FOLDER = 'wavs_game/'

MODEL_FOLDER = 'models/'
OUT_FOLDER = 'exp/'
STATE_OUT_FILE = 'state'
REWARD_OUT_FILE = 'reward'
LOCATION_OUT_FILE = 'location'
ACTION_OUT_FILE = 'action'
LOSS_OUT_FILE = 'loss'
RESULTS_FILE = 'results'


#########################################
### MODEL PARAMETERS
### General
BATCH_SIZE = 32
UPDATES = 25
CONV_SIZE = 10
KERNEL = 10
STRIDE = 2
CONV1CHANNELS = 32
CONV2CHANNELS = 32
CONV3CHANNELS = 32
MIDSIZE = 40
CONV_FREEZE = True
CONV_FREEZE_LAYER = 0
FREEZE_LAYER_TIME = 0
LAYERS = [CONV1CHANNELS, CONV2CHANNELS, CONV3CHANNELS, MIDSIZE]


SAMPLE_RATE = 16000
WINDOW_SIZE = 0.2
SPEC_WINDOW_LENGTH = 400
SPEC_WINDOW_HOP = 160
N_FFT = 400


### For Train
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 5
TRAIN_LR = 0.05
MEM_SIZE = 10000

CHANGE_TRIAL = (3, 'r')

STIMULUS_REPS = 1
MOVE_SEPERATION = 1
WAITTIME = 0
