### Set Imports
import torch
from DataLoader import *
from NN import *
import json
import time
import argparse

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

testing_batch_size = 32

for corpus in VALIDATION_COPORA:

    if OVERWRITE:
        write_method = 'w'
    else:
        write_method = 'x'
    outfile = open(ROOT + OUT_FOLDER + RESULTS_FILE + '_' +  PRETRAIN_MODELNAME + '_' + corpus + '.txt', write_method)
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

    phoneme_classifier = PhonemeConvNN(KERNEL, STRIDE, w, h, n_outputs).to(device)  # TODO: Need to create classifier
    phoneme_classifier.load_state_dict(torch.load(MODEL_LOCATION, map_location=device))
    phoneme_classifier.eval()



    results = torch.zeros((n_outputs, n_outputs))
    for i_batch in range(n_batches):
        print('running batch:', str(i_batch))

        # Get raw waveforms from data and reshape for classifier
        wavs, labels, phones = data.get_batch_phones(i_batch * testing_batch_size, (i_batch + 1) * testing_batch_size)

        wavs = wavs.reshape(wavs.size()[0], 1, wavs.size()[1]).to(device)
        wavs = data.transform(wavs)
        labels =  torch.stack(labels).to(device)

        # Generate Predictions
        predictions = phoneme_classifier(wavs)

        # Correct predictions
        predicted_cats = predictions.max(1).indices
        label_cats = labels.max(1).indices
        #correct_predictions = predicted_cats == label_cats
        #total_correct = sum(correct_predictions)

        #all_phones = data.get_phone_list()

        #phone_results = [(sum(predicted_cats[correct_predictions] == i), sum(label_cats == i)) for i in range(data.get_num_phones())]

        for b in range(testing_batch_size):
            results[label_cats[b], predicted_cats[b]] += 1

        ### Save Final Outputs
    outfile = open(ROOT + OUT_FOLDER + RESULTS_FILE + '_' +  PRETRAIN_MODELNAME + '_' + corpus + '.txt', 'a+')
    outfile.write(''.join([p + ' ' for p in data.get_phone_list()])+ '\n')
    for p in range(len(results)):
        outfile.write(''.join([str(int(i))+' ' for i in results[p]]) + '\n')

    outfile.close()

    print('data saved')

print('done')