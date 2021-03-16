### Set Imports

N_TRIALS=50
OUT_LAYER = -1
NUM_PHONES = 39



from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
from ReplayMemory import *
from DQN import *
from Environment import *
import json
import time
import argparse
import os
import shutil
from NN import *


### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("-run_num")
args = parser.parse_args()



### Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)

for key in all_params:
    globals()[key] = all_params[key]

if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run'+str(RUN_NUM)
    PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run'+str(RUN_NUM)

DATAFILE = ROOT + 'states_new_realspeech.txt'
DATAFILE1 = DATAFILE
DATAFILE2 = DATAFILE

MODEL_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'
print('parameters loaded from '+args.params_file)


### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))


### Set Model Start
tic = time.time()
running=True

testing_batch_size = 32

if OVERWRITE:
    write_method = 'w'
else:
    write_method = 'x'
outfile = open(ROOT + OUT_FOLDER + RESULTS_FILE + '_abx_' +  PRETRAIN_MODELNAME  + '.txt', write_method)
outfile.close()
acoustic_params =('mfcc', 16000, 0.2, 400, 400, 160, 13, True)
transform_type, sr, phone_window_size, n_fft, spec_window_length, spec_window_hop, n_mfcc, log_mels = acoustic_params

data1 = GameDataLoader(DATAFILE1, ROOT + WAVS_FOLDER, device, transform_type=transform_type, sr=sr,
                      phone_window_size=phone_window_size,
                      n_fft=n_fft, spec_window_length=spec_window_length, spec_window_hop=spec_window_hop,
                      n_mfcc=n_mfcc, log_mels=log_mels)

data2 = GameDataLoader(DATAFILE2, ROOT + WAVS_FOLDER, device, transform_type=transform_type, sr=sr,
                      phone_window_size=phone_window_size,
                      n_fft=n_fft, spec_window_length=spec_window_length, spec_window_hop=spec_window_hop,
                      n_mfcc=n_mfcc, log_mels=log_mels)

print('data loader created')

w, h = data1.get_feature_dims()
n_outputs = int(NUM_PHONES)
print(int(n_outputs), 'outputs |', w, 'width |', h, 'height')

print('running for', str(N_TRIALS))

n_batches = math.floor(N_TRIALS / testing_batch_size)

MODEL_LOCATION= ROOT + MODEL_FOLDER + 'model_' + PRETRAIN_MODELNAME + '_final.pt'

phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device)  # TODO: Need to create classifier
phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device))
phoneme_classifier.eval()

results = (['ans1','ans2'], [0,0],[0,0])

pdist = nn.PairwiseDistance()

for trial in [1, 2]:
    for i_batch in range(n_batches):
        print('running batch:', str(i_batch))

        # Get raw waveforms from data and reshape for classifier
        batcha = torch.stack(data1.get_random_batch(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size))
        batchb = torch.stack(data2.get_random_batch(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size))

        if trial == 1:
            batchx = torch.stack(data1.get_random_batch(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size))
        if trial == 2:
            batchx = torch.stack(data2.get_random_batch(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size))

        # Generate Predictions
        predictionsa = phoneme_classifier.get_out_from_layer(batcha.reshape(batcha.size()[0], 1, batcha.size()[1], batcha.size()[2]), OUT_LAYER)
        predictionsb = phoneme_classifier.get_out_from_layer(batchb.reshape(batchb.size()[0], 1, batchb.size()[1], batchb.size()[2]), OUT_LAYER)

        predictionsx = phoneme_classifier.get_out_from_layer(batchx.reshape(batchx.size()[0], 1, batchx.size()[1], batchx.size()[2]), OUT_LAYER)

        axdis = pdist(predictionsx, predictionsa)
        bxdis = pdist(predictionsx, predictionsb)

        a_answer = sum(axdis < bxdis)
        b_answer = sum(bxdis < axdis)

        results[trial][0] += int(a_answer)
        results[trial][1] += int(b_answer)

print('done')