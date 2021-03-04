### Set Imports
import torch
import torch.optim as optim
import torch.nn.functional as F
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
print('parameters loaded from '+args.params_file)

try:
    os.makedirs(ROOT + OUT_FOLDER + PRETRAIN_MODELNAME)
    print('created experiment output folder')
except OSError:
    print('experiment output folder exists')

OUT_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'

print('parameters loaded from '+args.params_file)
shutil.copyfile(args.params_file, ROOT + OUT_FOLDER + 'params_'+PRETRAIN_MODELNAME + '.params')

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

data = SpeechDataLoader( ROOT+PRETRAIN_SEGMENTS_FILE + '.txt', ROOT+PHONES_FILE + '.txt', ROOT+PRETRAIN_ALIGNMENTS_FILE + '.txt',
                         ROOT+WAVS_FOLDER, device, spec_window_hop=SPEC_WINDOW_HOP, spec_window_length=SPEC_WINDOW_LENGTH) #TODO: Need file names here
print('data loader created')

n_datapoints = len(data)
n_batches = math.floor(n_datapoints/BATCH_SIZE)
print('running for',str(n_datapoints), 'datapoints over',str(n_batches),'batches of size',str(BATCH_SIZE))


w, h = data.get_feature_dims()
n_outputs = int(data.get_num_phones())
print(int(n_outputs), 'outputs |',w, 'width |', h,'height')

phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device) #TODO: Need to create classifier
phoneme_classifier.train()
optimizer = optim.SGD(phoneme_classifier.parameters(), lr = LR)



LOSSFILE = ROOT + OUT_FOLDER + LOSS_OUT_FILE + '_' + PRETRAIN_MODELNAME + '.txt'
loss_record = ''

total_steps = PRETRAIN_EPOCHS*n_batches
print(str(total_steps), 'total steps')
### Set Model Start
tic = time.time()

for i_epoch in range(PRETRAIN_EPOCHS): #TODO: Need to define epochs
    print('starting epoch:',str(i_epoch))

    loss_record = loss_record + '\nepoch' + str(i_epoch)

    data.randomize_data()

    for i_batch in range(n_batches):

        # Get raw waveforms from data and reshape for classifier
        wavs, labels= data.get_batch(i_batch*BATCH_SIZE, (i_batch+1)*BATCH_SIZE)
        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)

        wavs=data.transform(wavs)
        labels =  torch.stack(labels).to(device)

        # Generate Predictions
        predictions = phoneme_classifier(wavs)

        # Loss calculated from predictions and true labels
        loss = F.smooth_l1_loss(predictions, labels)

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
                'steps complete: ' + str(i_batch + (i_epoch*n_batches + 1)) + ', percent complete: ' + str(
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
torch.save(phoneme_classifier.state_dict(), ROOT + OUT_FOLDER + MODEL_FOLDER + PRETRAIN_MODELNAME + '_final.pt')
print('model saved')
print('done')