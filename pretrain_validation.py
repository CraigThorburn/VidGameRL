### Set Imports

# import math
# import random
# import numpy as np
# import matplotlib
#import matplotlib.pyplot as plt
import sys
# from collections import namedtuple
from itertools import count

import torch
# import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
import torchaudio
from ReplayMemory import *
from DQN import *
from Environment import *
from DataLoader import *
from NN import *
import json
import time
import argparse
import os
import shutil

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
args = parser.parse_args()

### Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)
for key in all_params:
    globals()[key] = all_params[key]


PRETRAIN_MODELNAME='pretrain_'+PRETRAIN_MODELNAME
print('parameters loaded from '+args.params_file)

MODEL_LOCATION= MODEL_PATH + PRETRAIN_MODELNAME + '_final.pt'
PRETRAIN_MODELNAME='pretrain_'+PRETRAIN_MODELNAME +'_test'
print('parameters loaded from '+args.params_file)



### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

data = SpeechDataLoader( ROOT+SEGMENTS_FILE, ROOT+PHONES_FILE, ROOT+ALIGNMENTS_FILE,
                         ROOT+WAVS_FOLDER) #TODO: Need file names here
print('data loader created')

n_datapoints = len(data)
print('running for',str(n_datapoints))

w, h = data.get_feature_dims()
n_outputs = int(data.get_num_phones())
print(int(n_outputs), 'outputs |',w, 'width |', h,'height')

if OVERWRITE:
    write_method = 'w'
else:
    write_method = 'x'
outfile = open(ROOT + REWARD_LIST_FILE + '_' + PRETRAIN_MODELNAME + '.txt', write_method)
outfile.close()

phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device) #TODO: Need to create classifier
phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device))
phoneme_classifier.eval()


### Set Model Start
tic = time.time()

wavs, labels, phones = data.get_batch_phones(0, n_datapoints)

wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)
wavs = data.transform(wavs)
labels =  torch.stack(labels).to(device)

# Generate Predictions
predictions = phoneme_classifier(wavs)

# Correct predictions
predicted_cats = predictions.max(1).indices
label_cats = labels.max(1).indices
correct_predictions = predicted_cats == label_cats
total_correct = sum(correct_predictions)

all_phones = data.get_phone_list()

phone_results = [(sum(predicted_cats[correct_predictions] == i), sum(label_cats == i)) for i in range(data.get_num_phones())]




print('model complete')
print('saving data')

### Save Final Outputs
outfile = open(ROOT + REWARD_LIST_FILE + '_' + PRETRAIN_MODELNAME + '.txt', 'a+')
for p in range(len(all_phones)):
    outfile.write(all_phones[p]+' '+str(int(phone_results[p][0]))+ ' ' + str(int(phone_results[p][1])) + '\n')


outfile.close()
print('data saved')

print('done')