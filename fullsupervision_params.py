N_TRIALS = 1024 #TODO: Update
testing_batch_size = 32
DATAFILE1 =  'states_WSJ_correct_fullsupervision_ls.txt'
DATAFILE2 =  'states_WSJ_correct_fullsupervision_rs.txt'
PRETRAIN_EPOCHS = 50

### FILE VARIABLES
PRETRAIN_MODELNAME_ADITIONS = '_originalattempt5' #extra suffix for the pretraining folder
TRAIN_MODELNAME_ADITIONS = 'fullsupervision_originalattempt5_standard' #extra suffix for the training folder
### General
OVERWRITE = False #Default: False | whether any files should be overwritten
EXPERIMENT = 'supervised_pretraining_GPJ/' #Default: supervised_pretraining_<CORPUS> | experiment folder
#ROOT = '/mnt/c/files/research/projects/vid_game/data/'
ROOT = '/fs/clip-realspeech/projects/vid_game/data/' #Default: /fs/clip-realspeech/projects/vid_game/data | root folder on clip cluster
PARAMS_FOLDER = 'params' #Default: params | folder to store intermediate parameter files

### For Pretrain
PHONES_FILE = 'pretrain_phones' #Default: pretrain_phones | file containing phone list for pretraining
PRETRAIN_SEGMENTS_FILE = 'pretrain_segments' #Default: pretrain_segments | file containing segments list for pretraining
PRETRAIN_ALIGNMENTS_FILE = 'pretrain_alignments' #Default: pretrain_alignments | file containing alignments list for pretraining
WAVS_FOLDER = 'wavs_GPJ/' #Default: wavs_<CORPUS>/ | folder containing pretraining wavs

### For Train
GAME_WAVS_FOLDER = 'wavs_WSJ/' #Default: wavs_WSJ/ (was wavs for English) | Folder storing audio files for training
MODEL_FOLDER = 'models/' #Default: models/  | name of model output folder
OUT_FOLDER = 'exp/'  #Default: exp/ | name of experiment run folder
LOSS_OUT_FILE = 'loss' #Default: loss | name of loss ouptut file
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
CONV_FREEZE = False #Default: False | Whether to freeze convolutional layers
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
LOSS_TYPE = 'standard' #Default: ewc | (or standard) Type of loss function to use during training
FISCHER_CORPUS='GPJ' #Default: Same as pretraining corpus | Corpus for which to calculate fischer coefficients
FISCHER_FILE = 'fischercoeffs' #Default: fischercoeffs | Name of fischer coefficient file
EWC_IMPORTANCE = 0.0005 #Default: ??? | EWC Importance Weighting Coefficient

### For Train
TRAIN_EPOCHS = 96000
TRAIN_LR = 0.01 #Default: 0.05 | Learning rate during training
NUM_PHONES = 36   #Default: 36 (Japanese), 39 (English) | Number of phones in pretrained corpus
