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

shutil.copyfile(args.params_file, ROOT + PARAMS_FOLDER + '/' + PRETRAIN_MODELNAME + '.params')
print('parameter file moved to results location')

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

data = SpeechDataLoader( ROOT+SEGMENTS_FILE, ROOT+PHONES_FILE, ROOT+ALIGNMENTS_FILE,
                         ROOT+WAVS_FOLDER) #TODO: Need file names here
print('data loader created')

n_datapoints = len(data)
n_batches = math.floor(n_datapoints/PRETRAIN_BATCH_SIZE)
print('running for',str(n_datapoints), 'datapoints over',str(n_batches),'batches of size',str(PRETRAIN_BATCH_SIZE))

w = int((SAMPLE_RATE * WINDOW_SIZE /SPEC_WINDOW_LENGTH * 2) + 1)
h = N_FFT // 2 + 1
n_outputs = int(data.get_num_phones())
print(int(n_outputs), 'outputs |',w, 'width |', h,'height')

phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device) #TODO: Need to create classifier
phoneme_classifier.train()
optimizer = optim.SGD(phoneme_classifier.parameters(), lr = LR)



LOSSFILE = ROOT + 'exp/loss' + '_' + PRETRAIN_MODELNAME + '.txt'
loss_record = []

total_steps = PRETRAIN_EPOCHS*n_batches

### Set Model Start
tic = time.time()

for i_epoch in range(PRETRAIN_EPOCHS): #TODO: Need to define epochs
    print('starting epoch:',str(i_epoch))

    loss_record = loss_record + 'epoch' + str(i_epoch)

    data.randomize_data()

    for i_batch in range(n_batches):

        # Get raw waveforms from data and reshape for classifier
        wavs, labels= data.get_batch(i_batch*PRETRAIN_BATCH_SIZE, (i_batch+1)*PRETRAIN_BATCH_SIZE)
        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1])

        # Generate Predictions
        predictions = phoneme_classifier(wavs)

        # Loss calculated from predictions and true labels
        loss = F.smooth_l1_loss(predictions, torch.stack(labels))

        # Zero the optimizer
        optimizer.zero_grad()

        # Backpropogate loss and take optimizer step
        loss.backward()
        loss_record =  loss_record + ' ' + str(round(float(loss), 4))
        optimizer.step()

        ### Save Updates To File
        if i_batch % UPDATES == 0:
            outfile = open(LOSSFILE, 'a+')
            outfile.write(loss_record)
            loss_record = ''
            outfile.close()

            ### Print Timing Information
            toc = time.time()
            time_passed = toc - tic
            time_remaining = ((time_passed / (i_batch + (i_epoch*n_batches + 1))) * total_steps - time_passed) / 60
            print(
                'steps complete: ' + str(i_batch + (i_epoch*n_batches + 1)) + 'percent complete,  ' + str(
                    math.ceil((i_batch + (i_epoch*n_batches + 1)) / total_steps * 100)) + \
                ', time remaining: ' + str(int(time_remaining)) + ' minutes')

    ### Epoch is finished
    print('epoch finished')

    outfile = open(LOSSFILE, 'a+')
    outfile.write(loss_record)
    loss_record = ''
    outfile.close()

    ### Save Model Checkpoint
    torch.save(phoneme_classifier.state_dict(), ROOT + '/checkpoints/' + PRETRAIN_MODELNAME + '_epoch'+str(i_epoch)+'_model.pt')


print('model complete')
print('saving data')

### Save Final Outputs
outfile = open(LOSSFILE, 'a+')
outfile.write(loss_record)
outfile.close()
print('data saved')

### Save Final Model
print('saving model')
torch.save(phoneme_classifier.state_dict(), ROOT + '/models/' + PRETRAIN_MODELNAME + '_final.pt')
print('model saved')
print('done')