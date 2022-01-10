### Set Imports
import torch
from libs.DataLoader import *
from libs.NN import *
import json
import time
import argparse
from copy import deepcopy


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
    PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run' + str(RUN_NUM)

OUT_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'

print('parameters loaded from '+args.params_file)

MODEL_LOCATION= ROOT + OUT_FOLDER + 'model_' + PRETRAIN_MODELNAME + '_final.pt'



### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))


### Set Model Start
tic = time.time()
running=True

fischer_calculation_batch_size = 200



if OVERWRITE:
    write_method = 'w'
else:
    write_method = 'x'
outfile = open(ROOT + OUT_FOLDER + FISCHER_FILE + '_' +  PRETRAIN_MODELNAME + '.txt', write_method)
outfile.close()


data = SpeechDataLoader(ROOT + VALIDATION_SEGMENTS_FILE + '_' + FISCHER_CORPUS + '.txt', ROOT + PHONES_FILE + '.txt', ROOT + VALIDATION_ALIGNMENTS_FILE+ '_' + FISCHER_CORPUS + '.txt',
                        ROOT + WAVS_FOLDER , device)  # TODO: Need file names here
print('data loader created')

w, h = data.get_feature_dims()
n_outputs = int(data.get_num_phones())
print(int(n_outputs), 'outputs |', w, 'width |', h, 'height')

n_datapoints = len(data)
print('running for', str(n_datapoints))


phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device)  # TODO: Need to create classifier
phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device))



print('calculating Fischer coefficients')

model_params = {n: p for n, p in phoneme_classifier.named_parameters() if p.requires_grad}
precision_matrices = {}
for n, p in deepcopy(model_params).items():
    p.data.zero_()
    precision_matrices[n] = p.data

    phoneme_classifier.eval()

    # Get raw waveforms from data and reshape for classifier
#data.randomize_data()


n_batches = math.floor(n_datapoints / fischer_calculation_batch_size)

for i_batch in range(n_batches):

    wavs, labels, phones = data.get_batch_phones(i_batch * fischer_calculation_batch_size, (i_batch + 1) * fischer_calculation_batch_size)

    wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)
    wavs = data.transform(wavs)

#for inp in range(fischer_calculation_batch_size):
    phoneme_classifier.zero_grad()
   # input = wavs[inp,:,:].reshape(1,1,wavs.size()[2], wavs.size()[3])
    output = phoneme_classifier(input)#.flatten().to(device)
    label = output.max(1)[1].view(-1)
    #label = labels[inp].flatten().to(device)
    print(output.size(), label.size())
    loss = F.nll_loss(output,label)
    #loss = F.smooth_l1_loss(output, label)
    loss.backward()

    for n, p in phoneme_classifier.named_parameters():
        precision_matrices[n].data += p.grad.data ** 2 / n_batches

print('calculated diagonal of fischer information matrix')

means={}
for n, p in deepcopy(model_params).items():
    means[n] = p.data

print('fetched means')

torch.save([precision_matrices,means],ROOT + OUT_FOLDER + FISCHER_FILE + '_' +  PRETRAIN_MODELNAME + '.txt')

print('data saved')