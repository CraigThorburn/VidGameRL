### Set Imports

N_TRIALS = 7
#OUT_LAYER = -2
# NUM_PHONES = 36
pretrain_model = True
ABX_FILE_OUT = 'abx'

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

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("-run_num")
parser.add_argument("-pretrain")
parser.add_argument("-layer")
parser.add_argument("-token1")
parser.add_argument("-token2")
args = parser.parse_args()

if args.pretrain == 'true':
    pretrain_model = True
else:
    pretrain_model = False

### Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)

OUT_LAYER = int(args.layer)

for key in all_params:
    globals()[key] = all_params[key]

if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run' + str(RUN_NUM)
    #PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run' + str(RUN_NUM)

token1=args.token1
token2=args.token2
DATAFILE = ROOT + 'states_new_realspeech.txt'
DATAFILE1 = ROOT + token1+'s.txt'
DATAFILE2 = ROOT + token2+'s.txt'

if pretrain_model:
    MODEL_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
    OUT_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
    MODELNAME = PRETRAIN_MODELNAME
else:
    MODEL_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'
    OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'
    MODELNAME = TRAIN_MODELNAME

print('parameters loaded from ' + args.params_file)

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

### Set Model Start
tic = time.time()
running = True

testing_batch_size = 32

if OVERWRITE:
    write_method = 'w'
else:
    write_method = 'x'
OUTFILE = ROOT + OUT_FOLDER + ABX_FILE_OUT + str(OUT_LAYER) + '_' +token1+token2+"_"+ MODELNAME + '.txt'
outfile = open(OUTFILE, write_method)
outfile.close()
acoustic_params = ('mfcc', 22050, 0.2, 400, 400, 160, 13, True)
transform_type, sr, phone_window_size, n_fft, spec_window_length, spec_window_hop, n_mfcc, log_mels = acoustic_params

data1 = GameDataLoader(DATAFILE1, ROOT + ABX_WAVS_FOLDER, device, transform_type=transform_type, sr=sr,
                       phone_window_size=phone_window_size,
                       n_fft=n_fft, spec_window_length=spec_window_length, spec_window_hop=spec_window_hop,
                       n_mfcc=n_mfcc, log_mels=log_mels)
print('data 1 loaded')
data2 = GameDataLoader(DATAFILE2, ROOT + ABX_WAVS_FOLDER, device, transform_type=transform_type, sr=sr,
                       phone_window_size=phone_window_size,
                       n_fft=n_fft, spec_window_length=spec_window_length, spec_window_hop=spec_window_hop,
                       n_mfcc=n_mfcc, log_mels=log_mels)
print('data 2 loaded')
print('data loader created')

w, h = data1.get_feature_dims()
n_outputs = int(NUM_PHONES)
print(int(n_outputs), 'outputs |', w, 'width |', h, 'height')

print('running for', str(N_TRIALS))

n_batches = math.floor(N_TRIALS / testing_batch_size)

MODEL_LOCATION = ROOT + MODEL_FOLDER + 'model_' + MODELNAME + '_final.pt'

phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device)  # TODO: Need to create classifier
phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device), strict=False)
phoneme_classifier.eval()
print('classifier loaded')
results = (['ans1', 'ans2'], [0, 0], [0, 0])

pdist = nn.PairwiseDistance()
print('starting calculations')
for trial in [1, 2]:
    for i in range(N_TRIALS):
        for j in range(N_TRIALS):
            for k in range(N_TRIALS):
                if trial == 1:
                    if i == k:
                        next
                elif trial == 2:
                    if j==k:
                        next
                print('running trial:', i,j,k)

            # Get raw waveforms from data and reshape for classifier
                a = torch.stack(data1.get_batch(i, i+1))
                b = torch.stack(data2.get_batch(j, j+1))

                if trial == 1:
                    x = torch.stack(
                        data1.get_batch(k, k+1))
                if trial == 2:
                    x = torch.stack(
                        data2.get_batch(k, k+1))

            # Generate Predictions
                predictionsa = phoneme_classifier.get_out_from_layer(
                    a.reshape(a.size()[0], 1, a.size()[1], a.size()[2]), OUT_LAYER)
                predictionsb = phoneme_classifier.get_out_from_layer(
                    b.reshape(b.size()[0], 1, b.size()[1], b.size()[2]), OUT_LAYER)

                predictionsx = phoneme_classifier.get_out_from_layer(
                    x.reshape(x.size()[0], 1, x.size()[1], x.size()[2]), OUT_LAYER)

                if len(predictionsx.size())==2:
                    pass

                elif len(predictionsx.size()) ==3:
                    predictionsx = predictionsx.reshape(testing_batch_size, predictionsx.size()[1] * predictionsx.size()[2])
                    predictionsa = predictionsa.reshape(testing_batch_size, predictionsa.size()[1] * predictionsa.size()[2])
                    predictionsa = predictionsa.reshape(testing_batch_size, predictionsa.size()[1] * predictionsa.size()[2])
                elif len(predictionsx.size()) ==4:
                    predictionsx = predictionsx.reshape(testing_batch_size, predictionsx.size()[1] * predictionsx.size()[2] * predictionsx.size()[3])
                    predictionsa = predictionsa.reshape(testing_batch_size, predictionsa.size()[1] * predictionsa.size()[2] * predictionsa.size()[3])
                    predictionsb = predictionsb.reshape(testing_batch_size, predictionsb.size()[1] * predictionsb.size()[2] * predictionsb.size()[3])

                else:
                    raise RuntimeError
                print(predictionsx.size())

                axdis = pdist(predictionsx, predictionsa)
                bxdis = pdist(predictionsx, predictionsb)

                a_answer = sum(axdis < bxdis)
                b_answer = sum(bxdis < axdis)

                results[trial][0] += int(a_answer)
                results[trial][1] += int(b_answer)

print('saving')
print(results)
outfile = open(OUTFILE, 'a+')
for l in results:
    outfile.write(str(l[0]) + '\t' + str(l[1]) + '\n')
outfile.close()
print('done')


