

import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("param_save_file")
parser.add_argument("param_type")
parser.add_argument("-run_num")
args = parser.parse_args()

if args.param_type == 'cht':
    from cht_params import *
elif args.param_type == 'game':
    from game_params import *
elif args.param_type == 'acousticgame':
    from acousticgame_params import *
elif args.param_type == 'supervisedgame':
    from acousticgame_params import *
else:
    raise NotImplementedError


ROOT = ROOT + EXPERIMENT
if args.param_type == 'acousticgame':
    PRETRAIN_MODELNAME = 'pretrain_lr' + str(PRETRAIN_LR)+ '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_batchsize' + \
                     str(BATCH_SIZE) + '_epochs' + str(PRETRAIN_EPOCHS)+ PRETRAIN_MODELNAME_ADITIONS

elif args.param_type == 'supervisedgame':
    PRETRAIN_MODELNAME = 'pretrain_lr' + str(PRETRAIN_LR)+ '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_batchsize' + \
                     str(BATCH_SIZE) + '_epochs' + str(PRETRAIN_EPOCHS)+ PRETRAIN_MODELNAME_ADITIONS

TRAIN_MODELNAME = 'experiment_' + GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay' + str(EPS_DECAY) + '_targetupdate' + str(TARGET_UPDATE) \
                + '_waittime' + str(WAITTIME) + '_kernel' + str(KERNEL) + '_stride' + str(STRIDE) + '_lr'+str(TRAIN_LR) + '_freeze'+ str(CONV_FREEZE_LAYER)+'_freezetime' + str(FREEZE_LAYER_TIME) + \
                 TRAIN_MODELNAME_ADITIONS


if args.run_num:
    RUN_NUM = args.run_num
    TRAIN_MODELNAME = TRAIN_MODELNAME + '_run'+str(RUN_NUM)

    if args.param_type == 'acousticgame':
        PRETRAIN_MODELNAME = PRETRAIN_MODELNAME + '_run' + str(RUN_NUM)

    elif args.param_type == 'supervisedgame':
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
