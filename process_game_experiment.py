import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("root", help="root directory")
parser.add_argument("state", help="state file to use as input")
parser.add_argument("episode", help="episode file to use as input")
parser.add_argument("out", help="out file from experiment to use as input")
parser.add_argument("results", help="where to output results file")
parser.add_argument("-overwrite", help="overwrite any existing output files")
parser.add_argument("-batch", help="process over batch. will treat results and out as folders")
args = parser.parse_args()




with open(args.root+ '/' + args.state) as f:
    state_data = f.read().splitlines()
    header = state_data[0].split('\t')





state_list = [s.split('\t')[0] for s in state_data[1:]]





if args.batch:
    episodefiles = [args.episode + '/' + s for s in os.listdir(args.root+'/'+args.episode) if s[:5] == 'state' ]
    outfiles = [args.out + '/' + s for s in os.listdir(args.root+'/'+args.out) if s[:6] == 'reward' ]
    resultfiles = [args.results + '/results_' + s.split('expl_')[-1] for s in os.listdir(args.root+'/'+args.out)if s[:6] == 'reward' ]
else:
    outfiles = [args.out]
    resultfiles = [args.results]
    episodefiles = [args.episode]

iters = len(outfiles)

print(episodefiles)
print(outfiles)
print(resultfiles)

for i in range(iters):

    o = outfiles[i]
    r = resultfiles[i]
    e = episodefiles[i]
    print(o)

    with open(args.root + '/' + e) as f:
        episode_data = f.read().splitlines()

    with open(args.root+ '/' +o) as f:
        out_data = f.read().splitlines()

    if args.overwrite:
        write_method = 'w'
    else:
        write_method = 'x'
    resultfile = open(args.root+'/' +r , write_method)
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

    resultsfile = open(args.root+'/' +r, 'a+')

    resultsfile.write(''.join(['episode' + '\t']+[s +'\t' for s in state_list] + ['\n']))
    for r in results:
        resultsfile.write(''.join([str(i)+'\t' for i in r]))
        resultsfile.write('\n')

    resultsfile.close()

print('finished')