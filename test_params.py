
MODELNAME = 'lstm_' + GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay'+ str(EPS_DECAY) + '_targetupdate'+ str(TARGET_UPDATE) \
            + '_waittime'+ str(WAITTIME) + '_convsize'+ str(CONV_SIZE) + '_memreset'



UPDATES = 100
STIMULUS_REPS = 8
MOVE_SEPERATION = 1 #(not implemented yet)
WAITTIME = 0
CONV_SIZE = 10

TOKEN_TYPE = 'token'

GAME_TYPE = 'convolutionalmovement'
GAME_MODE = 'correct'
EXPERIMENT = 'experimental_continuous_movement_conv/'

OVERWRITE = True


#ROOT = '/mnt/c/files/research/projects/vid_game/data/'+EXPERIMENT
ROOT = '/fs/clip-realspeech/projects/vid_game/data/'+EXPERIMENT
#MODELNAME = 'test'




STATE_FILE = 'states'
REWARD_FILE = 'rewards'
TRANSITION_FILE = 'transitions'
LOCATION_FILE = 'locations'
MODEL_FOLDER = 'models'
TEST_FILE = 'test'


ACTION_TEST_FILE = 'exp/action_out'
STATE_TEST_FILE = 'exp/state_out'
LOCATION_TEST_FILE = 'exp/location_out'
REWARD_TEST_FILE = 'exp/reward_out'

# Set Variables
STATE_PATH = ROOT + STATE_FILE +'.txt'
REWARD_PATH = ROOT + REWARD_FILE+'.txt'
LOCATION_PATH = ROOT + LOCATION_FILE+'.txt'
TRANSITION_PATH = ROOT + TRANSITION_FILE+'.txt'
MODEL_PATH = ROOT + MODEL_FOLDER + '/'

