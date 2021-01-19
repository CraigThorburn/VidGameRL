
from params import *
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("param_save_file")
parser.add_argument("-gamma")
parser.add_argument("-eps_start")
parser.add_argument("-eps_end")
parser.add_argument("-eps_decay")
parser.add_argument("-target_update")
parser.add_argument("-lr")
parser.add_argument("-updates")
parser.add_argument("-stimulus_reps")
parser.add_argument("-move_seperation")
parser.add_argument("-waittime")
parser.add_argument("-conv_size")
parser.add_argument("-kernel")
parser.add_argument("-stride")
parser.add_argument("-mem_size")
parser.add_argument("-token_type")
parser.add_argument("-game_mode")
parser.add_argument("-game_type")
parser.add_argument("-experiment")
parser.add_argument("-overwrite")
parser.add_argument("-run_num")
args = parser.parse_args()

if args.gamma:
    GAMMA=args.gamma

if args.eps_start:
    EPS_START = args.eps_start

if args.eps_end:
    EPS_END = args.eps_end

if args.eps_decay:
    EPS_DECAY = args.eps_decay

if args.target_update:
    TARGET_UPDATE = args.target_update

if args.lr:
    LR = args.lr

if args.updates:
   UPDATES = args.updates

if args.stimulus_reps:
    STIMULUS_REPS = args.stimulus_reps

if args.move_seperation:
    MOVE_SEPERATION = args.move_seperation

if args.waittime:
    WAITTIME = args.waittime

if args.conv_size:
    CONV_SIZE = args.conv_size

if args.kernel:
    KERNEL = args.kernel

if args.stride:
    STRIDE = args.stride

if args.mem_size:
    MEM_SIZE = args.mem_size

if args.token_type:
    TOKEN_TYPE = args.token_sype

if args.game_mode:
    GAME_MODE = args.game_mode

if args.game_type:
    GAME_TYPE = args.game_type

if args.experiment:
    EXPERIMENT = args.experiment

if args.overwrite:
    OVERWRITE = args.overwrite

MODELNAME = GAME_TYPE + '_gamma' + str(GAMMA) + '_epsdecay' + str(EPS_DECAY) + '_targetupdate' + str(TARGET_UPDATE) \
                + '_waittime' + str(WAITTIME) + '_convsize' + str(CONV_SIZE)

if args.run_num:
    RUN_NUM = args.run_num
    MODELNAME = MODELNAME + '_run'+str(RUN_NUM)



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
