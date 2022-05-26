### Import Statements
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
from libs.ReplayMemory import *
from libs.DQN import *
from libs.Environment import *
import json
import time
import argparse
import os
import shutil
from libs.NN import *
import h5features as h5f


### Parser Arguments
# This loads in arguments from the command line such as the parameter file (which contains most parameters), the run
# number, whether we're looking at a model after pretraining or training, and the layer that we want to look at
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("-run_num")
parser.add_argument("-pretrain")
args = parser.parse_args()

# Set the pretraining parameter according to the command line
if args.pretrain == 'true':
    pretrain_model = True
else:
    pretrain_model = False

# This loads the parameters from file and sets them as global variables
with open(args.params_file, 'r') as f:
    all_params = json.load(f)
for key in all_params:
    globals()[key] = all_params[key]

print('parameters loaded from ' + args.params_file)


# Adds the run number to the modelname
if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run' + str(RUN_NUM)
    PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run' + str(RUN_NUM)

# Sets the correct model name based on whether we are looking at a model after pretraining or training
if pretrain_model:
    MODEL_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME # ADDED THE RUN1 TO MODEL_FOLDER AND MODELNAME
    OUT_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
    MODELNAME = PRETRAIN_MODELNAME
else:
    MODEL_FOLDER = OUT_FOLDER + TRAIN_MODELNAME
    OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'
    MODELNAME = TRAIN_MODELNAME


### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

### Set Model Start (this is used so we can add something to track how long the model is taking later)
tic = time.time()
running = True


for corpus in VALIDATION_COPORA:

    ### Step 1: Load data from data loader
    # Create an object SpeechDataLoader, as in pretrain_validation
    # The file names defined by the parameters (ie. VALIDATION_ALIGNMENTS_FILE) do not include the path to the file,
    # so you need to include ROOT before and '.txt' after (ie. ROOT + VALIDATION_ALIGNMENTS_FILE + '.txt.)
    # We'll use the VALIDATION files for these

    data = SpeechDataLoader(ROOT + VALIDATION_SEGMENTS_FILE + '_' + corpus + '.txt', ROOT + PHONES_FILE + '.txt', ROOT + VALIDATION_ALIGNMENTS_FILE+ '_' + corpus + '.txt', ROOT + WAVS_FOLDER , device) # Copied from pretrain_validation

    print('data loader created')

    # Use SpeechDataLoader.get_feature_dims() to get dimensions of input data and SpeechDataLoader.get_num_phones() to
    # get the number of phones used in this model.  Set them to the variables named below

    w, h = data.get_feature_dims()
    n_outputs = data.get_num_phones()

    print(int(n_outputs), 'outputs |', w, 'width |', h, 'height')

    # You can use len(SpeechDataLoader) to get the total amount of data we have in this data set.  Set to the varaible below

    n_datapoints = len(data)

    print('running for', str(n_datapoints))



    ### Step 2: Create network and load previous parameters
    # Create an instance of PhonConvNN - use dimensions of input data found above

    phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs, LAYERS).to(device)

    # Load in previous model.  You can use NN.PhonemeConvNN.load_state_dict to load parameters into model and
    # torch.load() to load the model itself.  You will need to map the location of the parameters to the device
    # set above and set strict=False to make sure that extra parameters that are created during training don't
    # cause problems.  An exmaple of this is in pretrain_validation

    MODEL_LOCATION= ROOT + MODEL_FOLDER + '/model_' + MODELNAME + '_final.pt'

    phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device), strict=False)

    # Set to train mode

    phoneme_classifier.eval()

    print('model loaded')



    ### Step 3: Collect network output, time and wav file from all tokens in dataset
    # Set batch size parameter

    testing_batch_size = 32

    # Calculate number of batches, by dividing total amount of data by batch size, call this n_batches

    n_batches = math.floor(n_datapoints / testing_batch_size)

    # Start list of wav_files, times and outputs,

    wav_names = []
    wav_times = []
    outputs = []

    # Iterate over total number of batches, for each
    for i_batch in range(n_batches):

        print('running batch:', str(i_batch))

        # Get wav files, you can use SpeechDataLoader.get_batch_phones

        wavs, labels, phones = data.get_batch_phones(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size)

        # You need to reshape this data using tensor.reshape to add a dummy second dimension equal to one
        # You can see what happens around line 79 of acousticgame_pretrain_validation.py.  You can get the current
        # size of the wavs vector using, wavs.size()

        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)

        # Transform the wav batch into mfccs using SpeechDataLoader.transform()

        wavs = data.transform(wavs)

        # Run wav files through network using the forward pass of the network

        predictions = phoneme_classifier(wavs)

        # Get wav_files and times using SpeechDataLoader.get_batch_wavname() and SpeechDataLoader.get_batch_time()
        # If these don't work entirely at first, type how you think they should be used and comment out the code for now
        # and we can troubleshoot later

        names = data.get_batch_wavname(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size)
        times = data.get_batch_time(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size)

        # Append each of these 3 to appropriate lists set before for loop

        wav_names += names
        wav_times += times
        outputs += predictions


    # Remaining datapoints not covered in i batches
    if n_datapoints - n_batches*testing_batch_size != 0:

        wavs, labels, phones = data.get_batch_phones(n_batches * testing_batch_size, n_datapoints)

        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)

        wavs = data.transform(wavs)

        predictions = phoneme_classifier(wavs)

        names = data.get_batch_wavname(n_batches * testing_batch_size, n_datapoints)
        times = data.get_batch_time(n_batches * testing_batch_size, n_datapoints)

        wav_names += names
        wav_times += times
        outputs += predictions


    # Result - should be three lists of equal length, one of vectors corresponding to a model output, one a corrsponding
    # list of wav names, and one a corresponding list of times where to find those wavs

    print(len(wav_names), "wav names |", len(wav_times), "wav times |", len(outputs), "outputs")



    ## Step 4: Create the h5f file and save it to the correct location

    FEATURES_OUTPATH = "/fs/clip-realspeech/projects/vid_game/abx/features/" # This can be put on the RunModels notebook

    with h5f.Writer(FEATURES_OUTPATH + TRAIN_MODELNAME + '_' + corpus + ".features") as writer:

        data = h5f.Data(wav_names, wav_times, outputs, check=True)

        writer.write(data, 'features')



