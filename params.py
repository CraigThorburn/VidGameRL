### FILE VARIABLES
MODELNAME_ADITIONS = ''

### General
OVERWRITE = False
EXPERIMENT = 'experimental_convolution_continuous/'
ROOT = '/mnt/c/files/research/projects/vid_game/data/'
ROOT = '/fs/clip-realspeech/projects/vid_game/data/'
PARAMS_FOLDER = 'params'


### For Pretrain
PHONES_FILE = 'pretrain_phones'
PRETRAIN_SEGMENTS_FILE = 'pretrain_segments'
PRETRAIN_ALIGNMENTS_FILE = 'pretrain_alignments'
WAVS_FOLDER = 'wavs/'

### For Validation
VALIDATION_SEGMENTS_FILE = 'validation_segments'
VALIDATION_ALIGNMENTS_FILE = 'validation_alignments'
VALIDATION_COPORA = ['GPJ']

### For Train
STATE_FILE = 'states'
REWARD_FILE = 'rewards'
EPISODE_FILE = 'episodes'
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

### For Test
SIMPLE_STATE_TEST_FILE='test_states_simple'



#########################################
### MODEL PARAMETERS
### General
BATCH_SIZE = 32
UPDATES = 250
CONV_SIZE = 10
KERNEL = 5
STRIDE = 2
CONV1CHANNELS = 32
CONV2CHANNELS = 32
CONV3CHANNELS = 32
MIDSIZE = 20
CONV_FREEZE = False
LAYERS = [CONV1CHANNELS, CONV2CHANNELS, CONV3CHANNELS, MIDSIZE]


SAMPLE_RATE = 16000
WINDOW_SIZE = 0.2
SPEC_WINDOW_LENGTH = 400
SPEC_WINDOW_HOP = 160
N_FFT = 400

### For Pretrain
PRETRAIN_LR = 0.09
GAME_TYPE = 'convmovement'
GAME_MODE = 'oneshot'
PRETRAIN_EPOCHS = 25

### For Validation

### For Train
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 150
TARGET_UPDATE = 5
TRAIN_LR = 0.05
MEM_SIZE = 1000

STIMULUS_REPS = 8
MOVE_SEPERATION = 1
WAITTIME = 0

### For Test


