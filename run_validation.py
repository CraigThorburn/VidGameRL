### Set Imports
import torch
from libs.DataLoader import *
from libs.NN import *
import json
import time
import argparse

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("-run_num")
parser.add_argument("-pretrain")
parser.add_argument("-layer")
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

if pretrain_model:
    MODEL_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
    OUT_FOLDER = OUT_FOLDER + PRETRAIN_MODELNAME + '/'
    MODELNAME = PRETRAIN_MODELNAME
else:
    MODEL_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'
    OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'
    MODELNAME = TRAIN_MODELNAME

print('parameters loaded from ' + args.params_file)

MODEL_LOCATION = ROOT + OUT_FOLDER + 'model_' + MODELNAME + '_final.pt'

### Set Computational Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device ' + str(device))

### Set Model Start
tic = time.time()
running=True

testing_batch_size = 32

for corpus in VALIDATION_COPORA:

    if OVERWRITE:
        write_method = 'w'
    else:
        write_method = 'x'
    outfile = open(ROOT + OUT_FOLDER + 'validation' + '_' +  MODELNAME + '_' + corpus + '.txt', write_method)
    outfile.close()


    data = SpeechDataLoader(ROOT + VALIDATION_SEGMENTS_FILE + '_' + corpus + '.txt', ROOT + PHONES_FILE + '.txt', ROOT + VALIDATION_ALIGNMENTS_FILE+ '_' + corpus + '.txt',
                            ROOT + WAVS_FOLDER , device)  # TODO: Need file names here
    print('data loader created')

    w, h = data.get_feature_dims()
    n_outputs = int(data.get_num_phones())
    print(int(n_outputs), 'outputs |', w, 'width |', h, 'height')

    n_datapoints = len(data)
    print('running for', str(n_datapoints))

    n_batches = math.floor(n_datapoints / testing_batch_size)
    if MODELTYPE == 'standard':
        phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device)  # TODO: Need to create classifier
        phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device), strict=False)
        phoneme_classifier.eval()
    elif MODELTYPE == 'extranodes':
        phoneme_classifier = PhonemeConvNN_extranodes(KERNEL, STRIDE, w, h, n_outputs, extra_nodes=EXTRA_NODES).to(
            device)  # TODO: Need to create classifier
        phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device), strict=False)
        phoneme_classifier.eval()

    results = torch.zeros((n_outputs, n_outputs))
    for i_batch in range(n_batches):
        print('running batch:', str(i_batch))

        # Get raw waveforms from data and reshape for classifier
        wavs, labels, phones = data.get_batch_phones(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size)
        #print(wavs.size(), len(labels))
        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)
        wavs = data.transform(wavs)
        labels =  torch.stack(labels).to(device)
        if MODELTYPE =='extranodes':
            labels = torch.cat((labels, torch.zeros((labels.size()[0], EXTRA_NODES), device=device)), 1)

        assert OUT_LAYER == -1, 'Out layer not valid for classification'

        # Generate Predictions
        predictions = phoneme_classifier.get_out_from_layer(wavs, OUT_LAYER, stacked_out = True )

        # Correct predictions
        predicted_cats = predictions.max(1).indices
        label_cats = labels.max(1).indices
        #print(label_cats.size(), predicted_cats.size())
        #correct_predictions = predicted_cats == label_cats
        #total_correct = sum(correct_predictions)

        #all_phones = data.get_phone_list()

        #phone_results = [(sum(predicted_cats[correct_predictions] == i), sum(label_cats == i)) for i in range(data.get_num_phones())]

        for b in range(label_cats.size()[0]):
            results[label_cats[b], predicted_cats[b]] += 1

        ### Save Final Outputs
    outfile = open(ROOT + OUT_FOLDER + 'validation' + '_' +  MODELNAME + '_' + corpus + '.txt', 'a+')
    outfile.write(''.join([p + ' ' for p in data.get_phone_list()])+ '\n')
    for p in range(len(results)):
        outfile.write(''.join([str(int(i))+' ' for i in results[p]]) + '\n')

    outfile.close()

    print('data saved')