
BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 150
TARGET_UPDATE = 5
LR = 0.09

UPDATES = 250
STIMULUS_REPS = 8
MOVE_SEPERATION = 1 #(not implemented yet)
WAITTIME = 0

CONV_SIZE = 10
KERNEL = 5
STRIDE = 2

CONV1CHANNELS = 16
CONV2CHANNELS = 32
MIDSIZE = 20


CONV_FREEZE = False

MEM_SIZE = 1000
TOKEN_TYPE = 'token'

GAME_TYPE = 'convmovement'
GAME_MODE = 'oneshot'
EXPERIMENT = 'experimental_convolution_continuous/'

OVERWRITE = False

#MODELNAME = GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay'+ str(EPS_DECAY) + '_targetupdate'+ str(TARGET_UPDATE) \
     #       + '_waittime'+ str(WAITTIME) + '_convsize'+ str(CONV_SIZE)

ROOT = '/mnt/c/files/research/projects/vid_game/data/'
ROOT = '/fs/clip-realspeech/projects/vid_game/data/'
#MODELNAME = 'test'




STATE_FILE = 'states'
SIMPLE_STATE_FILE='states_simple'
SIMPLE_STATE_TEST_FILE='test_states_simple'
REWARD_FILE = 'rewards'
EPISODE_FILE = 'episodes'
TRANSITION_FILE = 'transitions'
LOCATION_FILE = 'locations'
MODEL_FOLDER = 'models'
TEST_FILE = 'test'

PARAMS_FOLDER = 'params'
ACTION_LIST_FILE = 'exp/action_out'
STATE_LIST_FILE = 'exp/state_out'
LOCATION_LIST_FILE = 'exp/location_out'
REWARD_LIST_FILE = 'exp/reward_out'

RESULTS_FILE= 'results/results_reward_out'

PRETRAIN_BATCH_SIZE = 32
SAMPLE_RATE = 16000
WINDOW_SIZE = 0.2
SPEC_WINDOW_LENGTH = 400
SPEC_WINDOW_HOP = 160
N_FFT = 400
PRETRAIN_LR = 0.01
PRETRAIN_EPOCHS = 25

PRETRAIN_SEGMENTS_FILE = 'pretrain_segments'
PRETRAIN_ALIGNMENTS_FILE = 'pretrain_alignments'
VALIDATION_SEGMENTS_FILE = 'validation_segments'
VALIDATION_ALIGNMENTS_FILE = 'validation_alignments'
PHONES_FILE = 'pretrain_phones'
WAVS_FOLDER = 'wavs/'

VALIDATION_COPORA = ['GPJ']

#
#
#
# def save_params():
#      with open(ROOT + MODELNAME + '_params.txt', 'w') as fi:
#          fi.write(''.join(str(p) + ' ' for p in ALL_PARAMS))
