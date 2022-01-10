### FILE VARIABLES
PRETRAIN_MODELNAME_ADITIONS = 'pretraining_ewcbatch200newattempt0' #extra suffix for the pretraining folder
TRAIN_MODELNAME_ADITIONS = 'ewcbatch200_trainWSJ_EWC1_750epsnewattempt0' #extra suffix for the training folder
### General
OVERWRITE = False #Default: False | whether any files should be overwritten
EXPERIMENT = 'supervised_pretraining_WSJ/' #Default: supervised_pretraining_<CORPUS> | experiment folder **************************************
#ROOT = '/mnt/c/files/research/projects/vid_game/data/'
ROOT = '/fs/clip-realspeech/projects/vid_game/data/' #Default: /fs/clip-realspeech/projects/vid_game/data | root folder on clip cluster
PARAMS_FOLDER = 'params' #Default: params | folder to store intermediate parameter files

### For Pretrain
PHONES_FILE = 'pretrain_phones' #Default: pretrain_phones | file containing phone list for pretraining
PRETRAIN_SEGMENTS_FILE = 'pretrain_segments' #Default: pretrain_segments | file containing segments list for pretraining
PRETRAIN_ALIGNMENTS_FILE = 'pretrain_alignments' #Default: pretrain_alignments | file containing alignments list for pretraining
WAVS_FOLDER = 'wavs_WSJ/' #Default: wavs_<CORPUS>/ | folder containing pretraining wavs  **************************************

### For Validation
VALIDATION_SEGMENTS_FILE = 'validation_segments' #Default: validation_segments | file containing segments list for pretraining validation
VALIDATION_ALIGNMENTS_FILE = 'validation_alignments' #Default: validation_alignments | file containing alignments list for pretraining validation
VALIDATION_COPORA = ['WSJ'] #Default: Same as pretraining corpus | name of corpus to use for validation  **************************************

### For Train
STATE_FILE = 'states_WSJ_correct' #Default: states_WDJ_correct (was states_new_realspeech) | states file to use for training
REWARD_FILE = 'rewards' #Default: rewards | rewards file to use for training
EPISODE_FILE = 'episodes' #Default: episodes_full | episodes file to use for training
TRANSITION_FILE = 'transitions' #Default: transitions | transitions file to use for training
ACTION_FILE = 'actions' #Default: actions | actions file to use for training
LOCATION_FILE = 'locations' #Default: locations | locations file to use for training
SIMPLE_STATE_FILE = 'states_WSJ_correct' #Default: states_synthesized (was states_simple) | Simple list of files to use during results processing
GAME_WAVS_FOLDER = 'wavs_WSJ/' #Default: wavs_WSJ/ (was wavs for English) | Folder storing audio files for training

MODEL_FOLDER = 'models/' #Default: models/  | name of model output folder
OUT_FOLDER = 'exp/'  #Default: exp/ | name of experiment run folder
LOCATION_OUT_FILE = 'location' #Default: location  | name of location output file
ACTION_OUT_FILE = 'action' #Default: action | name of action output file
LOSS_OUT_FILE = 'loss' #Default: loss | name of loss ouptut file
STATE_OUT_FILE = 'state' #Default: state | name of state output file
REWARD_OUT_FILE = 'reward' #Default: reward | name of reward output file
RESULTS_FILE = 'results' #Default: results | name of result output file

### For Test
SIMPLE_STATE_TEST_FILE='test_states_simple' #Default: test_states_simple | NOTE CURRENTLY REQUIED Simple list of files to use during test results processing
ABX_WAVS_FOLDER = 'wavs_WSJ/'    #Default: wavs_WSJ/ (was wavs for English) | Folder storing audio samples for ABX

#########################################cd /fs/clip
### MODEL PARAMETERS
### General
BATCH_SIZE = 32 #Default: 32 | Size of batch to sample during training
UPDATES = 25 #Default: 25 | How often to print updates
CONV_SIZE = 10 #Deprecated

KERNEL = 5 #Default: 5 | Size of convolution kernel
STRIDE = 1 #Default: 1 | Stride of convolution
CONV1CHANNELS = 32 #Default: 32 | Number of channels in 1st convolution
CONV2CHANNELS = 32 #Default: 32 | Number of channels in 1st convolution
CONV3CHANNELS = 32 #Default: 32 | Number of channels in 1st convolution
MIDSIZE = 40 #Default: 40 | Number of nodes in linear layer
CONV_FREEZE = True #Default: False | Whether to freeze convolutional layers
CONV_FREEZE_LAYER = 0 #Default: 0 | Which convolutional layers should be frozen during training
FREEZE_LAYER_TIME = 0 #Default: 0 | At what episode convolutional layers should be frozen
LAYERS = [CONV1CHANNELS, CONV2CHANNELS, CONV3CHANNELS, MIDSIZE] # Does not need edited
SAMPLE_RATE = 16000 #Default: 16000 | Sample rate of acoustic input
WINDOW_SIZE = 0.2 #Default: 0.2 | Window size of acoustic token
SPEC_WINDOW_LENGTH = 400 #Default: 400 | Window length for spectrogram calculation
SPEC_WINDOW_HOP = 160 #Default: 160 | Spectrogram window hop size
N_FFT = 400 #Default: 400 | N_FFT parameter for spectrogram calculation

### For Pretrain
PRETRAIN_LR = 0.09 #Default" 0.09 | Learning rate during training
GAME_TYPE = 'convmovement' #Deprecated
GAME_MODE = 'oneshot' #Default: oneshot | Training game type
PRETRAIN_EPOCHS = 25 #Default: 25 | Number of epochs during pretraining

### For Validation
LOSS_TYPE = 'ewc' #Default: ewc | (or standard) Type of loss function to use during training
FISCHER_CORPUS='WSJ' #Default: Same as pretraining corpus | Corpus for which to calculate fischer coefficients**************************************
FISCHER_FILE = 'fischercoeffs' #Default: fischercoeffs | Name of fischer coefficient filer
EWC_IMPORTANCE = 1 #Default: ??? | EWC Importance Weighting Coefficient

### For Train
GAMMA = 0.9 #Default: 0.9 | Future reward decay parameter Gamma
EPS_START = 0.99 #Default: 0.99 | Starting Epsilon Value
EPS_END = 0.05 #Default: 0.05 | Final Epsilon value
EPS_DECAY = 300 #Default: 300 | Epsilon decay factor
TARGET_UPDATE = 5 #Default: 5 | How often to update target network to policy network
TRAIN_LR = 0.05 #Default: 0.05 | Learning rate during training
MEM_SIZE = 10000 #Default: 10000 | Size of RL replay memory
NUM_PHONES = 39   #Default: 36 (Japanese), 39 (English) | Number of phones in pretrained corpus **************************************
CONNECTION_LAYER = 'phone' #Default: phone | Layer at which to connect video game layers to pretrained layers
STIMULUS_REPS = 8 #Default: 8 | Number of times to repeat stimulus before moving to next trial
MOVE_SEPERATION = 1 #Deprecated
WAITTIME = 0 #Default: 0 | How long to wait before taking action
