
from params import *
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("param_save_file")
parser.add_argument("-run_num")
args = parser.parse_args()


ROOT = ROOT + EXPERIMENT
PRETRAIN_MODELNAME = 'pretrain_lr' + str(PRETRAIN_LR)+ '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_batchsize' + \
                     str(BATCH_SIZE) + '_epochs' + str(PRETRAIN_EPOCHS) + PRETRAIN_MODELNAME_ADITIONS

TRAIN_MODELNAME = 'experiment_' + GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay' + str(EPS_DECAY) + '_targetupdate' + str(TARGET_UPDATE) \
                + '_waittime' + str(WAITTIME) + '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_lr'+str(TRAIN_LR) + \
                 TRAIN_MODELNAME_ADITIONS


if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run'+str(RUN_NUM)
    PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run' + str(RUN_NUM)


filename = args.param_save_file
ps={}
f = open(filename, 'w')
for key in dir():
    try:
        if key not in ['argparse','json','ps','save_params', 'f', 'args', 'parser'] and key[0] != '_':
            ps[key] = globals()[key]
    except:
         pass
json.dump(ps,f)

f.close()
