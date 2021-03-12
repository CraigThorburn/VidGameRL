import argparse
import json
import os



### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("params_file", help="root directory")
parser.add_argument("is_test", help='is this this test results?')
parser.add_argument("-run_num")
parser.add_argument("-root", help="root directory")
parser.add_argument("-state", help="state file to use as input")
parser.add_argument("-episode", help="episode file to use as input")
parser.add_argument("-out", help="out file from experiment to use as input")
parser.add_argument("-results", help="where to output results file")
parser.add_argument("-overwrite", help="overwrite any existing output files")
parser.add_argument("-batch", help="process over batch. will treat results and out as folders")
args = parser.parse_args()

if args.batch.lower() == 'true':
    args.batch = True
else:
    args.batch=False


if args.params_file != 'NA':
    ### Define Model Name From Arguments



    with open(args.params_file, 'r') as f:
        all_params = json.load(f)

    for key in all_params:
        globals()[key] = all_params[key]

    if args.run_num:
        RUN_NUM = args.run_num
        TRAIN_MODELNAME = TRAIN_MODELNAME + '_run' + str(RUN_NUM)

    OUT_FOLDER = OUT_FOLDER + TRAIN_MODELNAME + '/'

    ep_inp = ROOT + EPISODE_FILE + '.txt'
    if args.is_test.lower() =='true':
        resultfiles = [ROOT + OUT_FOLDER + 'test_'+ RESULTS_FILE + '_' +  TRAIN_MODELNAME + '.txt']
        episodefiles = [ROOT + OUT_FOLDER + 'test_'+ STATE_OUT_FILE + '_' +  TRAIN_MODELNAME + '.txt']
        outfiles = [ROOT + OUT_FOLDER + 'test_'+ REWARD_OUT_FILE + '_' +  TRAIN_MODELNAME + '.txt']

        state_inp = ROOT + SIMPLE_STATE_TEST_FILE + '.txt'

    elif args.is_test.lower() =='false':
        resultfiles = [ROOT + OUT_FOLDER + 'train_'+ RESULTS_FILE + '_' +  TRAIN_MODELNAME + '.txt']
        episodefiles = [ROOT + OUT_FOLDER + 'train_'+ STATE_OUT_FILE + '_' +  TRAIN_MODELNAME + '.txt']
        outfiles = [ROOT + OUT_FOLDER + 'train_'+ REWARD_OUT_FILE + '_' +  TRAIN_MODELNAME + '.txt']

        state_inp = ROOT + SIMPLE_STATE_FILE + '.txt'


    else:
        raise NotImplementedError


    overwrite = OVERWRITE


else:
    state_inp = args.root + '/' + args.state
    overwrite = args.overwrite

    if args.batch:
        episodefiles = [args.root + '/' +args.episode + '/' + s for s in os.listdir(args.root + '/' + args.episode) if s[:5] == 'state']
        outfiles = [args.root + '/' +args.out + '/' + s for s in os.listdir(args.root + '/' + args.out) if s[:6] == 'reward']
        resultfiles = [args.root + '/' +args.results + '/results_' + s.split('expl_')[-1] for s in os.listdir(args.root + '/' + args.out)
                       if s[:6] == 'reward']
    else:
        outfiles = [args.root + '/' +args.out]
        resultfiles = [args.root + '/' +args.results]
        episodefiles = [args.root + '/' +args.episode]

with open(state_inp) as f:
    state_data = f.read().splitlines()
    header = state_data[0].split('\t')
state_list = [s.split('\t')[0] for s in state_data[1:]]

print(state_list)

iters = len(outfiles)

print(episodefiles)
print(outfiles)
print(resultfiles)

for i in range(iters):

    o = outfiles[i]
    r = resultfiles[i]
    e = episodefiles[i]
    print(o)

    with open(e) as f:
        episode_data = f.read().splitlines()

    with open(o) as f:
        out_data = f.read().splitlines()

    if overwrite:
        write_method = 'w'
    else:
        write_method = 'x'
    resultfile = open(r , write_method)
    resultfile.close()


    state_template = [0 for x in range(len(state_list))]

    results = []
    for l in range(len(episode_data)):
        ep_result = state_template.copy()
        ep = episode_data[l].split(' ')
        out = out_data[l].split(' ')
        for x in range(len(ep)-1):
            state = ep[x+1]
            reward = out[x+1]
            if state=='':
                pass
            ep_result[state_list.index(state)] += float(reward)
        results.append([ep[0]]+ ep_result)

    resultsfile = open(r, 'a+')

    resultsfile.write(''.join(['episode' + '\t']+[s +'\t' for s in state_list] + ['\n']))
    for r in results:
        resultsfile.write(''.join([str(i)+'\t' for i in r]))
        resultsfile.write('\n')

    resultsfile.close()

print('finished')