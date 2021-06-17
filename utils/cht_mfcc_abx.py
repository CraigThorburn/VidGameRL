import torch
import os
import json
import numpy as np
from NN import *

pdist = nn.PairwiseDistance()

cht_mfccs = torch.load('/mnt/c/files/research/projects/vid_game/data/cht/cht.mfccs')

stims = list(cht_mfccs.keys())
all_results = []
for i in range(len(stims)):
    i_results = []
    for j in range(len(stims)):

        a = stims[i]
        x = stims[j]

        a_mfcc = torch.flatten(cht_mfccs[a]).reshape(1,819)
        x_mfcc = torch.flatten(cht_mfccs[x]).reshape(1,819)

        r = pdist(a_mfcc, x_mfcc)

        i_results.append(r)
    all_results.append(i_results)

outfile = open('/mnt/c/files/research/projects/vid_game/data/cht/cht_mfcc_dists.txt', 'w')
for l in all_results:
    outfile.write(''.join([str(float(i)) +' ' for i in l]) + '\n')
outfile.close()
print('done')



abx_correct = 0
abx_incorrect = 0
combinations = ['mr', 'mf', 'fr','ff','cr','cf']

critical_ax = 'a'
other_b = 'i'
for i in range(6):
    for j in range(6):
        if i == j:
            continue
        for k in range (6):

            a = critical_ax + combinations[i]
            x = critical_ax + combinations [j]

            b = other_b + combinations[k]

            a_mfcc = torch.flatten(cht_mfccs[a]).reshape(1, 819)
            b_mfcc = torch.flatten(cht_mfccs[b]).reshape(1, 819)
            x_mfcc = torch.flatten(cht_mfccs[x]).reshape(1, 819)

            ax = pdist(a_mfcc, x_mfcc)
            bx = pdist(b_mfcc, x_mfcc)

            if ax <= bx:
                abx_correct += 1
                success = 'success'
            else:
                abx_incorrect +=1
                success = 'fail'

            print('a:', a, 'b:', b, 'x:', x, 'ax:', str(ax), 'bx:', str(bx), 'difference', str(bx-ax),'outcome:', success )


print()
print('---------------------------------')
print('change')
critical_ax = 'i'
other_b = 'a'
for i in range(6):
    for j in range(6):
        if i == j:
            continue
        for k in range(6):

            a = critical_ax + combinations[i]
            x = critical_ax + combinations[j]

            b = other_b + combinations[k]

            a_mfcc = torch.flatten(cht_mfccs[a]).reshape(1, 819)
            b_mfcc = torch.flatten(cht_mfccs[b]).reshape(1, 819)
            x_mfcc = torch.flatten(cht_mfccs[x]).reshape(1, 819)

            ax = pdist(a_mfcc, x_mfcc)
            bx = pdist(b_mfcc, x_mfcc)



            if ax <= bx:
                abx_correct += 1
                success = 'success'
            else:
                abx_incorrect += 1
                success = 'fail'

            print('a:', a, 'b:', b, 'x:', x, 'ax:', str(ax), 'bx:', str(bx), 'difference', str(bx-ax),'outcome:', success )

outfile = open('/mnt/c/files/research/projects/vid_game/data/cht/cht_mfcc_dists.txt', 'w')
for l in all_results:
    outfile.write(''.join([str(float(i)) +' ' for i in l]) + '\n')
outfile.close()
print('done')








