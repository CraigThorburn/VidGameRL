import os
os.getcwd()

from params import *
from Environment import *

ACTION_LIST = ROOT + ACTION_LIST_FILE + '_' + MODELNAME + '.txt'
REWARD_LIST = ROOT + REWARD_LIST_FILE + '_' + MODELNAME + '.txt'
STATE_LIST = ROOT + STATE_LIST_FILE + '_' + MODELNAME + '.txt'

env = AcousticsGame(REWARD_PATH, STATE_PATH, EPISODE_PATH, LOCATION_PATH, TRANSITION_PATH, STIMULUS_REPS)