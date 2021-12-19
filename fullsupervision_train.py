### Set Imports
import os
os.chdir('/fs/clip-realspeech/projects/vid_game/software/VidGameRL/')

from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
from libs.ReplayMemory import *
from libs.DQN import *
from libs.Environment import *
from libs.NN import *
import json
import time
import argparse
import os
import shutil
import libs.Loss
from fullsupervision_params import *

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("-run_num")
args = parser.parse_args()

## Define Model Name From Arguments
with open(args.params_file, 'r') as f:
    all_params = json.load(f)

for key in all_params:
    globals()[key] = all_params[key]

if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run'+str(RUN_NUM)
    PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run'+str(RUN_NUM)

try:
    os.makedirs(ROOT + OUT_FOLDER + TRAIN_MODELNAME)
    print('created experiment output folder')
except OSError:
    print('experiment output folder exists')


MODEL_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

DATAFILE1 = ROOT + DATAFILE1
DATAFILE2 = ROOT + DATAFILE2

acoustic_params = ('mfcc', 16000, 0.2, 400, 400, 160, 13, True)
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


MODEL_LOCATION = ROOT + '/exp/'+PRETRAIN_MODELNAME + '/model_' + PRETRAIN_MODELNAME + '_final.pt'

model_params = torch.load(MODEL_LOCATION, map_location=device)
#model_params.pop('lin1.weight')
#model_params.pop('lin1.bias')
phoneme_classifier = PhonemeConvNN_extranodes(KERNEL, STRIDE, w, h, NUM_PHONES, extra_nodes=2).to(device)  # TODO: Need to create classifier
phoneme_classifier.load_state_dict(model_params, strict=False)
optimizer = optim.SGD(phoneme_classifier.parameters(), lr = 0.01)
phoneme_classifier.train()

print('classifier loaded')

print('running for', str(N_TRIALS*2))
n_batches = math.floor(N_TRIALS / testing_batch_size)*2

if LOSS_TYPE == 'ewc':
    precision_matrices,means = torch.load(ROOT + MODEL_FOLDER + FISCHER_FILE + '_' +  PRETRAIN_MODELNAME + '.txt', map_location=device)

    for p, n in phoneme_classifier.named_parameters():
        if p not in precision_matrices.keys():
            print('parameters without fischer coeffs:')
            print(p)
            precision_matrices[p] = torch.zeros(n.size())
            means[p] = torch.zeros(n.size())

    loss_class = libs.Loss.EWCLoss(means, precision_matrices, EWC_IMPORTANCE, device)

elif LOSS_TYPE == 'standard':
    loss_class = libs.Loss.StandardLoss(device)
else:
    raise NotImplementedError

loss_func = loss_class.calculate_loss

loss_record=''
LOSSFILE = ROOT + OUT_FOLDER + LOSS_OUT_FILE + '_' + TRAIN_MODELNAME + '.txt'

data1_transformed_states = data1.get_transformed_states()
data2_transformed_states = data2.get_transformed_states()
data_keys = [[d, 0] for d in data1_transformed_states.keys()] + [[d, 1] for d in data2_transformed_states.keys()]
random.shuffle(data_keys)

all_data_dicts = [data1_transformed_states, data2_transformed_states]
print(len(all_data_dicts))
BATCH_SIZE = testing_batch_size
tic = time.time()
for i_epoch in range(TRAIN_EPOCHS):

    for i_batch in range(n_batches):

        # Get raw waveforms from data and reshape for classifier
        wavs = torch.stack(
            [all_data_dicts[d[1]][d[0]] for d in data_keys[i_batch * BATCH_SIZE: (i_batch + 1) * BATCH_SIZE]]).to(
            device)
        labels = torch.zeros(BATCH_SIZE, 2).to(device)
        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1], wavs.size()[2]).to(device)

        for d in range(BATCH_SIZE):
            token = data_keys[(i_batch * BATCH_SIZE) + d]
            #  labels[d,0]=1#
            labels[d, token[1]] = 1
        #   print(labels)
        #  print(wavs.size())

        # labels =  torch.stack(labels).to(device)

        # Generate Predictions
        _, predictions = phoneme_classifier(wavs)
        #print(predictions)
        #print(labels)

        # Loss calculated from predictions and true labels
        loss = F.smooth_l1_loss(predictions, labels)  #
        # torch.nn.utils.clip_grad_norm_(phoneme_classifier.parameters(), 0.1)
        # loss_func(predictions, labels, phoneme_classifier.named_parameters())
       # print(loss)

        debug = False
        if debug:
            ##################################
            del wavs
            print()
            print('Tracing back tensors:')


            def getBack(var_grad_fn):
                print(var_grad_fn)
                for n in var_grad_fn.next_functions:
                    if n[0]:
                        try:
                            tensor = getattr(n[0], 'variable')
                            print(n[0])
                            print('Tensor with grad found:', tensor)
                            print(' - gradient:', tensor.grad)
                            print()
                        except AttributeError as e:
                            getBack(n[0])


            loss.backward()
            getBack(loss.grad_fn)
            ##################################

        # Zero the optimizer
        optimizer.zero_grad()

        # Backpropogate loss and take optimizer step
        loss.backward()
        optimizer.step()

        loss_record = loss_record + ' ' + str(round(float(loss), 4))

        ### Save Updates To File
        if i_batch % UPDATES == -1:
            outfile = open(LOSSFILE, 'a+')
            outfile.write(loss_record)
            loss_record = ''
            outfile.close()

            ### Print Timing Information
            toc = time.time()
            time_passed = toc - tic
            time_remaining = ((time_passed / (i_batch + (i_epoch * n_batches + 1))) * (n_batches*TRAIN_EPOCHS) - time_passed) / 60
            print(
                'steps complete: ' + str(i_batch + (i_epoch * n_batches + 1)) + ', percent complete: ' + str(
                    math.ceil((i_batch + (i_epoch * n_batches + 1)) / (n_batches*TRAIN_EPOCHS) * 100)) + \
                ', time remaining: ' + str(int(time_remaining)) + ' minutes')

print('model complete')
print('saving data')

### Save Final Outputs
outfile = open(LOSSFILE, 'a+')
outfile.write(loss_record)
outfile.close()
print('data saved')

### Save Final Model
print('saving model')
torch.save(phoneme_classifier.state_dict(), ROOT + OUT_FOLDER + 'model_' + TRAIN_MODELNAME + '_final.pt')
print('model saved')
print('done')